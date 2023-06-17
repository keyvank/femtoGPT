pub mod program;
use super::*;
use crate::funcs::{GpuFunction, GpuFunctionGroup};
use program::{Brand, Buffer, Device, Program, ProgramError};

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
    forward: GpuFunction,
    backward: GpuFunctionGroup,
}

pub struct CompiledGraph {
    program: Program,
}

pub struct GpuGraph {
    device: Device,
    program: Option<CompiledGraph>,
    tensors: Vec<GpuTensor>,
    grads: Vec<GpuTensor>,
    names: Vec<String>,
    computations: BTreeMap<TensorId, GpuComputation>,
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
            src = src + &comp.forward.source_code;
            for func in comp.backward.funcs.iter() {
                src = src + &func.source_code;
            }
        }
        let prog = Program::from_opencl(&self.device, &src)?;
        for (v, g) in self.tensors.iter_mut().zip(self.grads.iter_mut()) {
            v.buffer = Some(GeneralBuffer::new(&prog, &v.mirror)?);
            g.buffer = Some(GeneralBuffer::new(&prog, &g.mirror)?);
            v.is_sync = true;
            g.is_sync = true;
        }
        self.program = Some(CompiledGraph { program: prog });
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
            mirror: GeneralTensor::Usize(Tensor::zeros(t.shape())),
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
    fn alloc(&mut self, t: Tensor<f32>, name: String) -> Result<TensorId, GraphError> {
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
    fn name_of(&self, _id: TensorId) -> Result<&String, GraphError> {
        unimplemented!();
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
            let buffs = c
                .backward
                .shared_buffers
                .iter()
                .map(|sz| program.program.create_buffer::<f32>(*sz))
                .collect::<Result<Vec<_>, ProgramError>>()?;

            for k in c.backward.funcs.iter() {
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

        /*Ok(loss.mean())*/
        //unimplemented!();
        Ok(0.)
    }
    fn forward(&mut self, _training: bool) -> Result<(), GraphError> {
        self.compile()?;
        let program = self.program.as_mut().ok_or(GraphError::NotReady)?;
        for (out, c) in self.computations.iter() {
            let inps = c
                .computation
                .inps
                .iter()
                .map(|id| self.tensors.get(*id).ok_or(GraphError::TensorNotFound(*id)))
                .collect::<Result<Vec<_>, GraphError>>()?;
            let out_tensor = self
                .tensors
                .get(*out)
                .ok_or(GraphError::TensorNotFound(*out))?;

            let mut kern = program.program.create_kernel(
                &c.forward.kernel_name,
                c.forward.global_work_size,
                c.forward.local_work_size,
            );
            kern = kern.arg(out_tensor.buffer.as_ref().ok_or(GraphError::NotReady)?);
            for inp in inps.iter() {
                kern = kern.arg(inp.buffer.as_ref().ok_or(GraphError::NotReady)?);
            }
            kern.run()?;

            self.tensors.get_mut(*out).unwrap().is_sync = false;
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
        let child = self.alloc(out, "".into())?;
        let forward = f.gpu_run(child, &shapes);
        let backward = f.gpu_grad(child, &shapes);

        self.computations.insert(
            child,
            GpuComputation {
                computation: Computation {
                    func: f,
                    inps: tensor_ids.to_vec(),
                },
                forward,
                backward,
            },
        );
        self.program = None; // Needs recompile
        Ok(child)
    }
    fn optimize<O: Optimizer>(
        &mut self,
        _opt: &mut O,
        _params: &HashSet<TensorId>,
        _learning_rate: f32,
    ) -> Result<(), GraphError> {
        unimplemented!();
    }
    fn fetch(&mut self, tensor_id: TensorId, grad: bool) -> Result<(), GraphError> {
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
}
