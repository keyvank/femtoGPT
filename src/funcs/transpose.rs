use super::Function;
use crate::tensor::*;

#[derive(Debug, Clone)]
pub struct Transpose {}
impl Transpose {
    pub fn new() -> Box<dyn Function> {
        Box::new(Self {})
    }
}

impl Function for Transpose {
    fn run(&mut self, inps: &[&Tensor<f32>], _training: bool) -> Result<Tensor<f32>, TensorError> {
        Ok(inps[0].transpose())
    }
    fn grad(
        &self,
        _inps: &[&Tensor<f32>],
        out_grad: &Tensor<f32>,
    ) -> Result<Vec<Tensor<f32>>, TensorError> {
        Ok(vec![out_grad.transpose()])
    }
    fn clone_box(&self) -> Box<dyn Function> {
        Box::new(self.clone())
    }
}
