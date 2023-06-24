use super::Function;
use crate::tensor::*;

#[cfg(feature = "gpu")]
use super::{gpu, GpuFunctionGroup, TensorId};

#[derive(Debug, Clone)]
pub struct MatMul;
impl MatMul {
    pub fn new() -> Box<dyn Function> {
        Box::new(Self {})
    }
}
impl Function for MatMul {
    fn run(
        &mut self,
        inps: &[&GeneralTensor],
        _training: bool,
    ) -> Result<Tensor<f32>, TensorError> {
        let inps = inps
            .iter()
            .map(|t| t.as_float())
            .collect::<Result<Vec<_>, TensorError>>()?;
        inps[0] ^ inps[1]
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
        Ok(vec![
            (out_grad ^ &inps[1].transpose()?)?,
            (&inps[0].transpose()? ^ out_grad)?,
        ])
    }
    fn clone_box(&self) -> Box<dyn Function> {
        Box::new(self.clone())
    }

    #[cfg(feature = "gpu")]
    fn gpu_impl(&self, out_id: TensorId, inps: &[Vec<usize>]) -> GpuFunctionGroup {
        gpu::matmul::gpu_impl(out_id, inps)
    }
}
