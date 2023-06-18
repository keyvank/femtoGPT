use super::Function;
use crate::tensor::*;

#[cfg(feature = "gpu")]
use super::{gpu, GpuFunction, GpuFunctionGroup, TensorId};

#[derive(Debug, Clone)]
pub struct Transpose {}
impl Transpose {
    pub fn new() -> Box<dyn Function> {
        Box::new(Self {})
    }
}

impl Function for Transpose {
    fn run(
        &mut self,
        inps: &[&GeneralTensor],
        _training: bool,
    ) -> Result<Tensor<f32>, TensorError> {
        let inps = inps
            .iter()
            .map(|t| t.as_float())
            .collect::<Result<Vec<_>, TensorError>>()?;
        inps[0].transpose()
    }
    fn grad(
        &self,
        _inps: &[&GeneralTensor],
        out_grad: &Tensor<f32>,
    ) -> Result<Vec<Tensor<f32>>, TensorError> {
        Ok(vec![out_grad.transpose()?])
    }
    fn clone_box(&self) -> Box<dyn Function> {
        Box::new(self.clone())
    }

    #[cfg(feature = "gpu")]
    fn gpu_run(&self, out_id: TensorId, inps: &[Vec<usize>]) -> GpuFunction {
        gpu::transpose::gpu_run(out_id, inps)
    }

    #[cfg(feature = "gpu")]
    fn gpu_grad(&self, out_id: TensorId, inps: &[Vec<usize>]) -> GpuFunctionGroup {
        gpu::transpose::gpu_grad(out_id, inps)
    }
}
