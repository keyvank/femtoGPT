use super::Function;
use crate::tensor::*;

#[cfg(feature = "gpu")]
use super::{gpu, GpuFunction, TensorId};

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
    fn gpu_run(&self, out_id: TensorId, inps: &[Vec<usize>]) -> GpuFunction {
        gpu::relu::gpu_run(out_id, inps)
    }
}
