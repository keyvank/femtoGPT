use super::Function;
use crate::tensor::*;

#[cfg(feature = "gpu")]
use super::{GpuFunction, TensorId};

#[derive(Debug, Clone)]
pub struct Relu;
impl Relu {
    pub fn new() -> Box<dyn Function> {
        Box::new(Self {})
    }
}
impl Function for Relu {
    fn run(&mut self, inps: &[&Tensor<f32>], _training: bool) -> Result<Tensor<f32>, TensorError> {
        Ok(inps[0].map_values(|f| if f > 0. { f } else { 0.01 * f }))
    }
    fn grad(
        &self,
        inps: &[&Tensor<f32>],
        out_grad: &Tensor<f32>,
    ) -> Result<Vec<Tensor<f32>>, TensorError> {
        let der = inps[0].map_values(|f| if f > 0. { 1. } else { 0.01 });
        Ok(vec![(&der * out_grad)?])
    }
    fn clone_box(&self) -> Box<dyn Function> {
        Box::new(self.clone())
    }

    #[cfg(feature = "gpu")]
    fn gpu_source_code(&self, out_id: TensorId, inps: &[Vec<usize>]) -> GpuFunction {
        let works = inps[0].iter().fold(1, |a, b| a * b);
        let source_code = format!(
            "__kernel void calc_{out_id}(
                            __global float* out,
                            __global float* a) {{
            uint id = get_global_id(0);
            if(id < {works}) {{
                float val = a[id];
                out[id] = val > 0. ? val : 0.;
            }}
        }}"
        );

        let local_work_size = 32;
        let global_work_size =
            works + ((local_work_size - (works % local_work_size)) % local_work_size);

        GpuFunction {
            source_code,
            kernel_name: format!("calc_{}", out_id),
            local_work_size,
            global_work_size,
        }
    }
}
