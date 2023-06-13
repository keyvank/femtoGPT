pub mod program;
use super::*;

use program::{Brand, Buffer, Device, Program};

pub struct GpuTensor {
    mirror: Tensor<f32>, // Mirror of GPU on CPU
    buffer: Buffer<f32>,
    is_sync: bool,
}

pub struct GpuGraph {
    program: Program,
    tensors: Vec<GpuTensor>,
    grads: Vec<GpuTensor>,
    names: Vec<String>,
    computations: BTreeMap<TensorId, Computation>,
}

impl GpuGraph {
    pub fn new() -> Result<Self, GraphError> {
        let device = Device::by_brand(Brand::Nvidia)?[0].clone();
        let src = r#"
            __kernel void add(__global float* out, uint sz, __global float* a, __global float* b) {
                uint id = get_global_id(0);
                if(id < sz) {
                    out[id] = a[id] + b[id];
                }
            }
        "#;
        let program = Program::from_opencl(&device, src)?;
        Ok(Self {
            tensors: Default::default(),
            grads: Default::default(),
            computations: Default::default(),
            names: Default::default(),
            program,
        })
    }
    pub fn alloc(&mut self, t: Tensor<f32>, name: String) -> Result<TensorId, GraphError> {
        self.grads.push(GpuTensor {
            buffer: self.program.create_buffer::<f32>(t.size())?,
            mirror: Tensor::zeros(t.shape()),
            is_sync: false,
        });
        self.tensors.push(GpuTensor {
            buffer: self.program.create_buffer::<f32>(t.size())?,
            mirror: Tensor::zeros(t.shape()),
            is_sync: false,
        });
        self.names.push(name);
        let id = self.tensors.len() - 1;
        self.load(id, &t)?;
        Ok(id)
    }
    pub fn load<T: TensorOps<f32>>(
        &mut self,
        tensor_id: TensorId,
        tensor: &T,
    ) -> Result<(), GraphError> {
        let gt = self.tensors.get_mut(tensor_id).unwrap();
        gt.mirror = tensor.view().into();
        gt.buffer.write_from(gt.mirror.blob())?;
        gt.is_sync = true;
        Ok(())
    }
    pub fn fetch(&mut self, tensor_id: TensorId) -> Result<&Tensor<f32>, GraphError> {
        let gt = self.tensors.get_mut(tensor_id).unwrap();
        if !gt.is_sync {
            let mut read = Vec::new();
            gt.buffer.read_into(&mut read)?;
            gt.mirror = Tensor::raw(gt.mirror.shape(), read)?;
            gt.is_sync = true;
        }
        Ok(&gt.mirror)
    }
    pub fn get(&self, id: TensorId) -> Result<&GpuTensor, GraphError> {
        self.tensors.get(id).ok_or(GraphError::TensorNotFound(id))
    }
    pub fn call(
        &mut self,
        mut f: Box<dyn Function>,
        tensor_ids: &[TensorId],
    ) -> Result<TensorId, GraphError> {
        let tensors = tensor_ids
            .iter()
            .map(|id| self.get(*id).map(|gt| &gt.mirror))
            .collect::<Result<Vec<_>, GraphError>>()?;
        let out = f.run(&tensors, false)?;
        let child = self.alloc(out, "".into())?;
        self.computations.insert(
            child,
            Computation {
                func: f,
                inps: tensor_ids.to_vec(),
            },
        );
        Ok(child)
    }
    pub fn forward(&mut self, _training: bool) -> Result<(), GraphError> {
        for (out, c) in self.computations.iter_mut() {
            let inps = c
                .inps
                .iter()
                .map(|id| self.tensors.get(*id).ok_or(GraphError::TensorNotFound(*id)))
                .collect::<Result<Vec<_>, GraphError>>()?;
            let out_tensor = self
                .tensors
                .get(*out)
                .ok_or(GraphError::TensorNotFound(*out))?;

            let works = out_tensor.mirror.size();
            let local_work_size = 32;
            let global_work_size =
                works + ((local_work_size - (works % local_work_size)) % local_work_size);
            let mut kern = self
                .program
                .create_kernel("add", global_work_size, local_work_size);
            kern = kern.arg(&out_tensor.buffer);
            kern = kern.arg(works as u32);
            for inp in inps.iter() {
                kern = kern.arg(&inp.buffer);
            }
            kern.run()?;
        }
        Ok(())
    }
}
