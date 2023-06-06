use super::{Function, Tensor, TensorOps};

#[derive(Debug, Clone)]
pub struct Relu;
impl Relu {
    pub fn new() -> Box<dyn Function> {
        Box::new(Self {})
    }
}
impl Function for Relu {
    fn run(&mut self, inps: &[&Tensor<f32>], _training: bool) -> Tensor<f32> {
        inps[0].map_values(|f| if f > 0. { f } else { 0.01 * f })
    }
    fn grad(&self, inps: &[&Tensor<f32>], out_grad: &Tensor<f32>) -> Vec<Tensor<f32>> {
        let der = inps[0].map_values(|f| if f > 0. { 1. } else { 0.01 });
        vec![&der * out_grad]
    }
    fn clone_box(&self) -> Box<dyn Function> {
        Box::new(self.clone())
    }
}
