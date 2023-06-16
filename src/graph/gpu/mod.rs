pub mod program;
use super::*;
use crate::funcs::{GpuFunction, GpuFunctionGroup};
use program::{Brand, Buffer, Device, Program, ProgramError};

pub struct GpuTensor {
    mirror: GeneralTensor, // Mirror of GPU on CPU
    buffer: Option<Buffer<f32>>,
    is_sync: bool,
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
            let mut v_buff = prog.create_buffer(v.mirror.size())?;
            let mut g_buff = prog.create_buffer(g.mirror.size())?;
            v_buff.write_from(v.mirror.blob())?;
            g_buff.write_from(g.mirror.blob())?;
            v.buffer = Some(v_buff);
            g.buffer = Some(g_buff);
            v.is_sync = true;
            g.is_sync = true;
        }
        self.program = Some(CompiledGraph { program: prog });
        Ok(())
    }
    pub fn call(
        &mut self,
        mut f: Box<dyn Function>,
        tensor_ids: &[TensorId],
    ) -> Result<TensorId, GraphError> {
        let (tensors, shapes): (Vec<&Tensor<f32>>, Vec<Vec<usize>>) = tensor_ids
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
}

impl GpuGraph {
    pub fn fetch(&mut self, tensor_id: TensorId) -> Result<&GeneralTensor, GraphError> {
        let gt = self.tensors.get_mut(tensor_id).unwrap();
        if !gt.is_sync {
            match gt.mirror {
                GeneralTensor::Float(mirror) => {
                    let mut read = vec![0.; gt.mirror.size()];
                    gt.buffer
                        .as_ref()
                        .ok_or(GraphError::NotReady)?
                        .read_into(&mut read)?;
                    gt.mirror = GeneralTensor::Float(Tensor::raw(mirror.shape(), read)?);
                }
                GeneralTensor::Usize(mirror) => {
                    let mut read = vec![0; gt.mirror.size()];
                    gt.buffer
                        .as_ref()
                        .ok_or(GraphError::NotReady)?
                        .read_into(&mut read)?;
                    gt.mirror = GeneralTensor::Usize(Tensor::raw(mirror.shape(), read)?);
                }
            }
            gt.is_sync = true;
        }
        Ok(&gt.mirror)
    }
    pub fn fetch_grad(&mut self, tensor_id: TensorId) -> Result<&Tensor<f32>, GraphError> {
        let gt = self.grads.get_mut(tensor_id).unwrap();
        if !gt.is_sync {
            let mut read = vec![0.; gt.mirror.as_float()?.size()];
            gt.buffer
                .as_ref()
                .ok_or(GraphError::NotReady)?
                .read_into(&mut read)?;
            gt.mirror = GeneralTensor::Float(Tensor::raw(gt.mirror.as_float()?.shape(), read)?);
            gt.is_sync = true;
        }
        Ok(gt.mirror.as_float()?)
    }
}

impl Graph for GpuGraph {
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
            .write_from(gt.mirror.as_float()?.blob())?;
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
            .write_from(gt.mirror.as_float()?.blob())?;
        gt.is_sync = true;
        Ok(())
    }
    fn zero_grad(&mut self) {
        unimplemented!();
    }
    fn name_of(&self, _id: TensorId) -> Result<&String, GraphError> {
        unimplemented!();
    }
    fn get(&self, _id: TensorId) -> Result<&GeneralTensor, GraphError> {
        unimplemented!();
    }
    fn get_grad(&self, _id: TensorId) -> Result<&Tensor<f32>, GraphError> {
        unimplemented!();
    }
    fn backward_all(
        &mut self,
        id: TensorId,
        _loss_fn: Box<dyn Loss>,
        _limit: Option<usize>,
    ) -> Result<f32, GraphError> {
        self.compile()?;

        let output_shape = self.get(id)?.mirror.as_float()?.shape().to_vec();
        self.load_grad(id, &Tensor::ones(&output_shape))?;

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
        _f: Box<dyn Function>,
        _tensor_ids: &[TensorId],
    ) -> Result<TensorId, GraphError> {
        unimplemented!();
    }
    fn optimize<O: Optimizer>(
        &mut self,
        _opt: &mut O,
        _params: &HashSet<TensorId>,
        _learning_rate: f32,
    ) -> Result<(), GraphError> {
        unimplemented!();
    }
}
