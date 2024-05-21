use super::Function;
use crate::tensor::*;

#[cfg(feature = "gpu")]
use super::{gpu, GpuFunction, TensorId};

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
        // 这段代码的作用，是将out_grad中的每个元素乘以self.coeff
        Ok(vec![out_grad.map_values(|d| d * self.coeff)])
    }
    fn clone_box(&self) -> Box<dyn Function> {
        Box::new(self.clone())
    }
    //gpu_impl的作用是将CPU上的数据转移到GPU上
    #[cfg(feature = "gpu")]//cfg是条件编译的关键字，feature是一个特性，gpu是一个特性的名字，如果有gpu这个特性，那么就会编译这段代码
    fn gpu_impl(&self, out_id: TensorId, inps: &[Vec<usize>]) -> GpuFunction {
        gpu::coeff::gpu_impl(out_id, inps, self.coeff)
    }
}
