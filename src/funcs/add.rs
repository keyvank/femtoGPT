use super::Function;
use crate::tensor::*;

#[cfg(feature = "gpu")]
use super::{gpu, GpuFunction, TensorId};

#[derive(Debug, Clone)]
pub struct Add;
impl Add {
    pub fn new() -> Box<dyn Function> {
        Box::new(Self {})
    }
}
impl Function for Add {
    fn run(&mut self, inps: &[&Tensor<f32>], _training: bool) -> Result<Tensor<f32>, TensorError> {
        inps[0] + inps[1]
    }
    fn grad(
        &self,
        _inps: &[&Tensor<f32>],
        out_grad: &Tensor<f32>,
    ) -> Result<Vec<Tensor<f32>>, TensorError> {
        Ok(vec![out_grad.clone(), out_grad.clone()])
    }
    fn clone_box(&self) -> Box<dyn Function> {
        Box::new(self.clone())
    }

    #[cfg(feature = "gpu")]
    fn gpu_run(&self, out_id: TensorId, inps: &[Vec<usize>]) -> GpuFunction {
        gpu::add::gpu_run(out_id, inps)
    }
}
