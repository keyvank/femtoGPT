use super::Function;
use crate::tensor::*;

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
}
