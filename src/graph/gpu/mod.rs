pub mod program;
use super::*;
use crate::funcs::GpuFunctionGroup;
use program::{Brand, Buffer, Device, Program, ProgramError};
use std::collections::HashMap;

pub enum GeneralBuffer {
    Float(Buffer<f32>),
    Usize(Buffer<usize>),
}

pub struct GpuTensor {
    mirror: GeneralTensor, // Mirror of GPU on CPU
    buffer: Option<GeneralBuffer>,
    is_sync: bool,
}

impl GeneralBuffer {
    fn new(prog: &Program, t: &GeneralTensor) -> Result<Self, GraphError> {
        let mut buff = match t {
            GeneralTensor::Float(t) => GeneralBuffer::Float(prog.create_buffer::<f32>(t.size())?),
            GeneralTensor::Usize(t) => GeneralBuffer::Usize(prog.create_buffer::<usize>(t.size())?),
        };
        buff.write_from(t)?;
        Ok(buff)
    }
    fn length(&self) -> usize {
        match self {
            GeneralBuffer::Float(b) => b.length(),
            GeneralBuffer::Usize(b) => b.length(),
        }
    }
    fn write_from(&mut self, t: &GeneralTensor) -> Result<(), GraphError> {
        match self {
            GeneralBuffer::Float(b) => match t {
                GeneralTensor::Float(t) => {
                    b.write_from(t.blob())?;
                }
                GeneralTensor::Usize(_) => {
                    return Err(GraphError::IncompatibleTypes);
                }
            },
            GeneralBuffer::Usize(b) => match t {
                GeneralTensor::Float(_) => {
                    return Err(GraphError::IncompatibleTypes);
                }
                GeneralTensor::Usize(t) => {
                    b.write_from(t.blob())?;
                }
            },
        }
        Ok(())
    }
    fn read_into(&self, t: &mut GeneralTensor) -> Result<(), GraphError> {
        match self {
            GeneralBuffer::Float(b) => match t {
                GeneralTensor::Float(t) => {
                    let mut blob = vec![0.; t.size()];
                    b.read_into(&mut blob)?;
                    *t = Tensor::raw(t.shape(), blob)?;
                }
                GeneralTensor::Usize(_) => {
                    return Err(GraphError::IncompatibleTypes);
                }
            },
            GeneralBuffer::Usize(b) => match t {
                GeneralTensor::Float(_) => {
                    return Err(GraphError::IncompatibleTypes);
                }
                GeneralTensor::Usize(t) => {
                    let mut blob = vec![0; t.size()];
                    b.read_into(&mut blob)?;
                    *t = Tensor::raw(t.shape(), blob)?;
                }
            },
        }
        Ok(())
    }
}

impl<'a> program::KernelArgument<'a> for &'a GeneralBuffer {
    fn push(&self, kernel: &mut program::Kernel<'a>) {
        match self {
            GeneralBuffer::Float(b) => {
                b.push(kernel);
            }
            GeneralBuffer::Usize(b) => {
                b.push(kernel);
            }
        }
    }
}

#[derive(Clone)]
pub struct GpuComputation {
    computation: Computation,
    gpu_function: GpuFunctionGroup,
}

pub struct CompiledGraph {
    program: Program,
    comp_buffers: HashMap<TensorId, Vec<GeneralBuffer>>,
}

pub struct GpuGraph {
    device: Device,
    program: Option<CompiledGraph>,
    tensors: Vec<GpuTensor>,
    grads: Vec<GpuTensor>,
    params: Vec<TensorId>,
    names: Vec<String>,
    computations: BTreeMap<TensorId, GpuComputation>,
    optimizer_state: HashMap<String, GpuTensor>,
    optimizer_step: usize,
}

impl GpuGraph {
    pub fn new() -> Result<Self, GraphError> {
        let device = Device::by_brand(Brand::Nvidia)?[0].clone();
        Ok(Self {
            device,
            tensors: Default::default(),
            grads: Default::default(),
            computations: Default::default(),
            names: Default::default(),
            params: Default::default(),
            optimizer_state: Default::default(),
            optimizer_step: 0,
            program: None,
        })
    }
    pub fn get(&self, id: TensorId) -> Result<&GpuTensor, GraphError> {
        self.tensors.get(id).ok_or(GraphError::TensorNotFound(id))
    }
    pub fn compile(&mut self) -> Result<(), GraphError> {
        if self.program.is_some() {
            return Ok(());
        }
        let mut src = String::new();
        for comp in self.computations.values() {
            for func in comp.gpu_function.forward_funcs.iter() {
                src = src + &func.source_code;
            }
            for func in comp.gpu_function.funcs.iter() {
                src = src + &func.source_code;
            }
        }
        src += "
        __kernel void optimizer(__global float *param, __global float *grad, __global float *m, __global float *v,  float learning_rate, ulong step, ulong n) {
            uint id = get_global_id(0);
            param += id;
            grad += id;
            m += id;
            v += id;
            float beta1 = 0.9;
            float beta2 = 0.999;
            float weight_decay = 0.01;
            if(id < n) {
                *param = *param - *param * learning_rate * weight_decay;
                *m = beta1 * (*m) + (1 - beta1) * (*grad);
                *v = beta2 * (*v) + (1 - beta2) * (*grad) * (*grad);
                float m_hat = *m / (1.0 - pow(beta1, step + 1));
                float v_hat = *v / (1.0 - pow(beta2, step + 1));
                float v_hat_sqrt_inv = learning_rate / (sqrt(v_hat) + 1e-8);
                *param = *param - m_hat * v_hat_sqrt_inv;
            }
        }
        ";
        let prog = Program::from_opencl(&self.device, &src)?;

        let mut comp_buffers = HashMap::new();

        for (id, comp) in self.computations.iter() {
            comp_buffers.entry(*id).or_insert_with(|| {
                comp.gpu_function
                    .shared_buffers
                    .iter()
                    .map(|sz| {
                        prog.create_buffer::<f32>(*sz)
                            .map(|b| GeneralBuffer::Float(b))
                    })
                    .collect::<Result<Vec<_>, ProgramError>>()
                    .unwrap()
            });
        }

        let mut optimizer_state = HashMap::new();
        for p in self.params.to_vec() {
            let t = self.tensors.get(p).unwrap().mirror.shape().to_vec();
            let m_val = GeneralTensor::Float(Tensor::zeros(&t));
            let v_val = GeneralTensor::Float(Tensor::zeros(&t));
            let m = GpuTensor {
                buffer: Some(GeneralBuffer::new(&prog, &m_val)?),
                mirror: m_val,
                is_sync: true,
            };
            let v = GpuTensor {
                buffer: Some(GeneralBuffer::new(&prog, &v_val)?),
                mirror: v_val,
                is_sync: true,
            };
            optimizer_state.insert(format!("{}_m", self.name_of(p)?), m);
            optimizer_state.insert(format!("{}_v", self.name_of(p)?), v);
        }
        for (v, g) in self.tensors.iter_mut().zip(self.grads.iter_mut()) {
            v.buffer = Some(GeneralBuffer::new(&prog, &v.mirror)?);
            g.buffer = Some(GeneralBuffer::new(&prog, &g.mirror)?);
            v.is_sync = true;
            g.is_sync = true;
        }
        self.program = Some(CompiledGraph {
            program: prog,
            comp_buffers,
        });
        self.optimizer_state = optimizer_state;
        Ok(())
    }
}

impl GpuGraph {
    pub fn fetch_grad(&mut self, tensor_id: TensorId) -> Result<&Tensor<f32>, GraphError> {
        let gt = self.grads.get_mut(tensor_id).unwrap();
        if !gt.is_sync {
            gt.buffer
                .as_mut()
                .ok_or(GraphError::NotReady)?
                .read_into(&mut gt.mirror)?;
            gt.is_sync = true;
        }
        Ok(gt.mirror.as_float()?)
    }
}

impl Graph for GpuGraph {
    fn alloc_usize(&mut self, t: Tensor<usize>, name: String) -> Result<TensorId, GraphError> {
        self.grads.push(GpuTensor {
            buffer: None,
            mirror: GeneralTensor::Float(Tensor::zeros(t.shape())),
            is_sync: false,
        });
        self.tensors.push(GpuTensor {
            buffer: None,
            mirror: GeneralTensor::Usize(t),
            is_sync: false,
        });
        self.names.push(name);
        let id = self.tensors.len() - 1;
        Ok(id)
    }
    fn load_usize<T: TensorOps<usize>>(
        &mut self,
        tensor_id: TensorId,
        tensor: &T,
    ) -> Result<(), GraphError> {
        self.compile()?;
        let gt = self.tensors.get_mut(tensor_id).unwrap();
        gt.mirror = GeneralTensor::Usize(tensor.view().into());
        gt.buffer
            .as_mut()
            .ok_or(GraphError::NotReady)?
            .write_from(&gt.mirror)?;
        gt.is_sync = true;
        Ok(())
    }
    fn alloc(
        &mut self,
        t: Tensor<f32>,
        is_param: bool,
        name: String,
    ) -> Result<TensorId, GraphError> {
        self.grads.push(GpuTensor {
            buffer: None,
            mirror: GeneralTensor::Float(Tensor::zeros(t.shape())),
            is_sync: false,
        });
        self.tensors.push(GpuTensor {
            buffer: None,
            mirror: GeneralTensor::Float(t),
            is_sync: false,
        });
        self.names.push(name);
        let id = self.tensors.len() - 1;
        if is_param {
            self.params.push(id);
        }
        Ok(id)
    }
    fn load<T: TensorOps<f32>>(
        &mut self,
        tensor_id: TensorId,
        tensor: &T,
    ) -> Result<(), GraphError> {
        self.compile()?;
        let gt = self.tensors.get_mut(tensor_id).unwrap();
        gt.mirror = GeneralTensor::Float(tensor.view().into());
        gt.buffer
            .as_mut()
            .ok_or(GraphError::NotReady)?
            .write_from(&gt.mirror)?;
        gt.is_sync = true;
        Ok(())
    }
    fn load_grad<T: TensorOps<f32>>(
        &mut self,
        tensor_id: TensorId,
        tensor: &T,
    ) -> Result<(), GraphError> {
        self.compile()?;
        let gt = self.grads.get_mut(tensor_id).unwrap();
        gt.mirror = GeneralTensor::Float(tensor.view().into());
        gt.buffer
            .as_mut()
            .ok_or(GraphError::NotReady)?
            .write_from(&gt.mirror)?;
        gt.is_sync = true;
        Ok(())
    }
    fn zero_grad(&mut self) -> Result<(), GraphError> {
        for gt in self.grads.iter_mut() {
            gt.mirror = GeneralTensor::Float(Tensor::zeros(gt.mirror.shape()));
            gt.buffer
                .as_mut()
                .ok_or(GraphError::NotReady)?
                .write_from(&gt.mirror)?;
            gt.is_sync = true;
        }
        Ok(())
    }
    fn name_of(&self, id: TensorId) -> Result<&String, GraphError> {
        self.names.get(id).ok_or(GraphError::TensorNotFound(id))
    }
    fn get(&self, id: TensorId) -> Result<&GeneralTensor, GraphError> {
        let gt = self.tensors.get(id).unwrap();
        if !gt.is_sync {
            return Err(GraphError::NotReady);
        }
        Ok(&gt.mirror)
    }
    fn get_grad(&self, id: TensorId) -> Result<&Tensor<f32>, GraphError> {
        let gt = self.grads.get(id).unwrap();
        if !gt.is_sync {
            return Err(GraphError::NotReady);
        }
        Ok(gt.mirror.as_float()?)
    }
    fn backward_all(
        &mut self,
        id: TensorId,
        loss_fn: Box<dyn Loss>,
        _limit: Option<usize>,
    ) -> Result<f32, GraphError> {
        self.compile()?;

        self.fetch(id, false)?;
        let output = Graph::get(self, id)?;
        let (loss, grad) = loss_fn.run(output)?;
        let mean_coeff = 1. / loss.size() as f32;

        self.load_grad(id, &(&grad * &Tensor::scalar(mean_coeff))?)?;

        let program = self.program.as_mut().ok_or(GraphError::NotReady)?;

        for (id, c) in self.computations.clone().iter().rev() {
            let inps = c
                .computation
                .inps
                .iter()
                .map(|id| {
                    self.tensors[*id]
                        .buffer
                        .as_ref()
                        .ok_or(GraphError::NotReady)
                })
                .collect::<Result<Vec<_>, GraphError>>()?;
            let inp_grads = c
                .computation
                .inps
                .iter()
                .map(|id| self.grads[*id].buffer.as_ref().ok_or(GraphError::NotReady))
                .collect::<Result<Vec<_>, GraphError>>()?;
            let out = self.tensors[*id]
                .buffer
                .as_ref()
                .ok_or(GraphError::NotReady)?;
            let out_grad = self.grads[*id]
                .buffer
                .as_ref()
                .ok_or(GraphError::NotReady)?;

            let buffs = program.comp_buffers.get(id).ok_or(GraphError::NotReady)?;

            for k in c.gpu_function.funcs.iter() {
                let mut kern = program.program.create_kernel(
                    &k.kernel_name,
                    k.global_work_size,
                    k.local_work_size,
                );
                kern = kern.arg(out);
                kern = kern.arg(out_grad);
                for buff in buffs.iter() {
                    kern = kern.arg(buff);
                }
                for (inp, grad) in inps.iter().cloned().zip(inp_grads.iter().cloned()) {
                    kern = kern.arg(inp);
                    kern = kern.arg(grad);
                }
                kern.run()?;
            }

            for inp in c.computation.inps.iter() {
                self.grads.get_mut(*inp).unwrap().is_sync = false;
            }
        }

        Ok(loss.mean())
    }
    fn forward(&mut self, training: bool) -> Result<(), GraphError> {
        self.compile()?;
        let program = self.program.as_mut().ok_or(GraphError::NotReady)?;
        for (out, c) in self.computations.iter() {
            let inps = c
                .computation
                .inps
                .iter()
                .map(|id| self.tensors.get(*id).ok_or(GraphError::TensorNotFound(*id)))
                .collect::<Result<Vec<_>, GraphError>>()?;
            let batches = self.tensors.get(*out).unwrap().mirror.shape()[0];
            let out_tensor = self
                .tensors
                .get(*out)
                .ok_or(GraphError::TensorNotFound(*out))?;

            let buffs = program.comp_buffers.get(out).ok_or(GraphError::NotReady)?;

            for func in c.gpu_function.forward_funcs.iter() {
                let local_work_size = func.local_work_size;
                let mut global_work_size = if training {
                    func.global_work_size
                } else {
                    func.global_work_size / batches
                };
                global_work_size +=
                    (local_work_size - (global_work_size % local_work_size)) % local_work_size;

                let mut kern = program.program.create_kernel(
                    &func.kernel_name,
                    global_work_size,
                    local_work_size,
                );
                kern = kern.arg(out_tensor.buffer.as_ref().ok_or(GraphError::NotReady)?);
                for buff in buffs.iter() {
                    kern = kern.arg(buff);
                }
                for inp in inps.iter() {
                    kern = kern.arg(inp.buffer.as_ref().ok_or(GraphError::NotReady)?);
                }
                kern.run()?;
            }

            let gt = self.tensors.get_mut(*out).unwrap();
            gt.is_sync = false;
            /*gt.buffer
                .as_mut()
                .ok_or(GraphError::NotReady)?
                .read_into(&mut gt.mirror)?;
            println!("{}: {} ({})", &format!("{:?}", c.computation.func)[..3], beg.elapsed().as_millis(), c.forward.global_work_size);*/
        }
        Ok(())
    }
    fn call(
        &mut self,
        mut f: Box<dyn Function>,
        tensor_ids: &[TensorId],
    ) -> Result<TensorId, GraphError> {
        let (tensors, shapes): (Vec<&GeneralTensor>, Vec<Vec<usize>>) = tensor_ids
            .iter()
            .map(|id| {
                self.get(*id)
                    .map(|gt| (&gt.mirror, gt.mirror.shape().to_vec()))
            })
            .collect::<Result<Vec<_>, GraphError>>()?
            .into_iter()
            .unzip();
        let out = f.run(&tensors, false)?;
        let child = self.alloc(out, false, "".into())?;
        let gpu_function = f.gpu_impl(child, &shapes);

        self.computations.insert(
            child,
            GpuComputation {
                computation: Computation {
                    func: f,
                    inps: tensor_ids.to_vec(),
                },
                gpu_function,
            },
        );
        self.program = None; // Needs recompile
        Ok(child)
    }
    fn optimize<O: Optimizer>(
        &mut self,
        _optimizer: &O, // TODO: Generate OpenCL code with this
        learning_rate: f32,
    ) -> Result<(), GraphError> {
        self.compile()?;

        for p in self.params.iter() {
            self.tensors.get_mut(*p).unwrap().is_sync = false;
        }

        let program = self.program.as_mut().ok_or(GraphError::NotReady)?;

        let tens: Vec<(
            &GeneralBuffer,
            &GeneralBuffer,
            &GeneralBuffer,
            &GeneralBuffer,
        )> = self
            .tensors
            .iter_mut()
            .enumerate()
            .filter(|(id, _)| self.params.contains(id))
            .map(|(id, params)| {
                let name = self.names.get(id).ok_or(GraphError::TensorNotFound(id))?;
                let grad = self.grads.get(id).ok_or(GraphError::TensorNotFound(id))?;
                let m = self
                    .optimizer_state
                    .get(&format!("{}_m", name))
                    .unwrap()
                    .buffer
                    .as_ref()
                    .unwrap();
                let v = self
                    .optimizer_state
                    .get(&format!("{}_v", name))
                    .unwrap()
                    .buffer
                    .as_ref()
                    .unwrap();
                Ok((
                    params.buffer.as_ref().ok_or(GraphError::NotReady)?,
                    grad.buffer.as_ref().ok_or(GraphError::NotReady)?,
                    m,
                    v,
                ))
            })
            .collect::<Result<Vec<_>, GraphError>>()?;
        for (param, grad, m, v) in tens {
            let works = param.length();
            let local_work_size = 32;
            let global_work_size =
                works + ((local_work_size - (works % local_work_size)) % local_work_size);
            let mut kern = program
                .program
                .create_kernel("optimizer", global_work_size, 32);
            kern = kern.arg(param);
            kern = kern.arg(grad);
            kern = kern.arg(m);
            kern = kern.arg(v);
            kern = kern.arg(learning_rate);
            kern = kern.arg(self.optimizer_step);
            kern = kern.arg(works);
            kern.run()?;
        }
        self.optimizer_step += 1;
        Ok(())
    }
    fn fetch(&mut self, tensor_id: TensorId, grad: bool) -> Result<(), GraphError> {
        self.compile()?;
        let gt = self.tensors.get_mut(tensor_id).unwrap();
        if !gt.is_sync {
            gt.buffer
                .as_mut()
                .ok_or(GraphError::NotReady)?
                .read_into(&mut gt.mirror)?;
            gt.is_sync = true;
        }
        if grad {
            let gg = self.grads.get_mut(tensor_id).unwrap();
            if !gg.is_sync {
                gg.buffer
                    .as_mut()
                    .ok_or(GraphError::NotReady)?
                    .read_into(&mut gg.mirror)?;
                gg.is_sync = true;
            }
        }
        Ok(())
    }
    fn params(&self) -> &[TensorId] {
        &self.params
    }
    fn optimizer_step(&self) -> usize {
        self.optimizer_step
    }
    fn get_optimizer_state(&self) -> Result<OptimizerState, GraphError> {
        let mut result = HashMap::new();
        for p in self.params.iter() {
            let name = self.name_of(*p)?;
            let key_m = format!("{}_m", name);
            let key_v = format!("{}_v", name);
            let m = self.optimizer_state.get(&key_m).unwrap();
            let v = self.optimizer_state.get(&key_v).unwrap();
            let mut m_val = GeneralTensor::Float(Tensor::zeros(m.mirror.shape()));
            let mut v_val = GeneralTensor::Float(Tensor::zeros(v.mirror.shape()));
            m.buffer.as_ref().unwrap().read_into(&mut m_val)?;
            v.buffer.as_ref().unwrap().read_into(&mut v_val)?;
            result.insert(key_m, m_val.as_float()?.clone());
            result.insert(key_v, v_val.as_float()?.clone());
        }
        Ok(OptimizerState {
            step: self.optimizer_step,
            state: result,
        })
    }
    fn set_optimizer_state(&mut self, state: &OptimizerState) -> Result<(), GraphError> {
        self.optimizer_step = state.step;
        for p in self.params.iter() {
            let name = self.name_of(*p)?;
            let key_m = format!("{}_m", name);
            let key_v = format!("{}_v", name);

            let m = self
                .optimizer_state
                .get_mut(&key_m)
                .unwrap()
                .buffer
                .as_mut()
                .unwrap();
            let m_val = GeneralTensor::Float(state.state.get(&key_m).unwrap().clone());
            m.write_from(&m_val)?;
            drop(m);

            let v = self
                .optimizer_state
                .get_mut(&key_v)
                .unwrap()
                .buffer
                .as_mut()
                .unwrap();
            let v_val = GeneralTensor::Float(state.state.get(&key_v).unwrap().clone());
            v.write_from(&v_val)?;
        }
        Ok(())
    }
}
