use super::Function;
use crate::tensor::*;

#[cfg(feature = "gpu")]
use super::{gpu, GpuFunctionGroup, TensorId};

#[derive(Debug, Clone)]
pub struct Coeff {
    coeff: f32,
}
impl Coeff {
    pub fn new(coeff: f32) -> Box<dyn Function> {
        Box::new(Self { coeff })
    }
}
impl Function for Coeff {
    fn run(
        &mut self,
        inps: &[&GeneralTensor],
        _training: bool,
    ) -> Result<Tensor<f32>, TensorError> {
        Ok(inps[0].as_float()?.map_values(|f| f * self.coeff))
    }
    fn grad(
        &self,
        _inps: &[&GeneralTensor],
        out_grad: &Tensor<f32>,
    ) -> Result<Vec<Tensor<f32>>, TensorError> {
        Ok(vec![out_grad.map_values(|d| d * self.coeff)])
    }
    fn clone_box(&self) -> Box<dyn Function> {
        Box::new(self.clone())
    }

    #[cfg(feature = "gpu")]
    fn gpu_grad(&self, out_id: TensorId, inps: &[Vec<usize>]) -> GpuFunctionGroup {
        gpu::coeff::gpu_grad(out_id, inps, self.coeff)
    }
}
