use super::{Function, Tensor};

#[derive(Debug)]
pub struct Cat {}
impl Cat {
    pub fn new() -> Box<dyn Function> {
        Box::new(Self {})
    }
}

impl Function for Cat {
    fn run(&mut self, inps: &[&Tensor<f32>], _training: bool) -> Tensor<f32> {
        Tensor::cat(inps)
    }
    fn grad(&self, inps: &[&Tensor<f32>], out_grad: &Tensor<f32>) -> Vec<Tensor<f32>> {
        Tensor::split(out_grad, inps.len())
    }
}
