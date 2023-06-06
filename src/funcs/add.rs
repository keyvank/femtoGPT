use super::{Function, Tensor};

#[derive(Debug, Clone)]
pub struct Add;
impl Add {
    pub fn new() -> Box<dyn Function> {
        Box::new(Self {})
    }
}
impl Function for Add {
    fn run(&mut self, inps: &[&Tensor<f32>], _training: bool) -> Tensor<f32> {
        inps[0] + inps[1]
    }
    fn grad(&self, _inps: &[&Tensor<f32>], out_grad: &Tensor<f32>) -> Vec<Tensor<f32>> {
        vec![out_grad.clone(), out_grad.clone()]
    }
    fn clone_box(&self) -> Box<dyn Function> {
        Box::new(self.clone())
    }
}
