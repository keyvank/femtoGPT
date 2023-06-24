use super::Function;
use crate::tensor::*;

#[cfg(feature = "gpu")]
use super::{gpu, GpuFunctionGroup, TensorId};

#[derive(Debug, Clone)]
pub struct Relu;
impl Relu {
    pub fn new() -> Box<dyn Function> {
        Box::new(Self {})
    }
}
impl Function for Relu {
    fn run(
        &mut self,
        inps: &[&GeneralTensor],
        _training: bool,
    ) -> Result<Tensor<f32>, TensorError> {
        let inps = inps
            .iter()
            .map(|t| t.as_float())
            .collect::<Result<Vec<_>, TensorError>>()?;
        Ok(inps[0].map_values(|f| if f > 0. { f } else { 0.01 * f }))
    }
    fn grad(
        &self,
        inps: &[&GeneralTensor],
        out_grad: &Tensor<f32>,
    ) -> Result<Vec<Tensor<f32>>, TensorError> {
        let inps = inps
            .iter()
            .map(|t| t.as_float())
            .collect::<Result<Vec<_>, TensorError>>()?;
        let der = inps[0].map_values(|f| if f > 0. { 1. } else { 0.01 });
        Ok(vec![(&der * out_grad)?])
    }
    fn clone_box(&self) -> Box<dyn Function> {
        Box::new(self.clone())
    }

    #[cfg(feature = "gpu")]
    fn gpu_grad(&self, out_id: TensorId, inps: &[Vec<usize>]) -> GpuFunctionGroup {
        gpu::relu::gpu_grad(out_id, inps)
    }
}
