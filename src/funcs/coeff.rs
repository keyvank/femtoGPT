use super::{Function, Tensor, TensorOps};

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
    fn run(&mut self, inps: &[&Tensor<f32>], _training: bool) -> Tensor<f32> {
        inps[0].map_values(|f| f * self.coeff)
    }
    fn grad(&self, _inps: &[&Tensor<f32>], out_grad: &Tensor<f32>) -> Vec<Tensor<f32>> {
        vec![out_grad.map_values(|d| d * self.coeff)]
    }
    fn clone_box(&self) -> Box<dyn Function> {
        Box::new(self.clone())
    }
}
