use super::{Function, Tensor, TensorOps};

#[derive(Debug)]
pub struct Transpose {}
impl Transpose {
    pub fn new() -> Box<dyn Function> {
        Box::new(Self {})
    }
}

impl Function for Transpose {
    fn run(&mut self, inps: &[&Tensor<f32>]) -> Tensor<f32> {
        inps[0].transpose()
    }
    fn grad(
        &self,
        _inps: &[&Tensor<f32>],
        _out: &Tensor<f32>,
        out_grad: &Tensor<f32>,
    ) -> Vec<Tensor<f32>> {
        vec![out_grad.transpose()]
    }
}
