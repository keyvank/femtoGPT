use super::{Function, Tensor, TensorOps};

#[derive(Debug)]
pub struct MatMul;
impl MatMul {
    pub fn new() -> Box<dyn Function> {
        Box::new(Self {})
    }
}
impl Function for MatMul {
    fn run(&mut self, inps: &[&Tensor<f32>], _training: bool) -> Tensor<f32> {
        inps[0] ^ inps[1]
    }
    fn grad(
        &self,
        inps: &[&Tensor<f32>],
        _out: &Tensor<f32>,
        out_grad: &Tensor<f32>,
    ) -> Vec<Tensor<f32>> {
        vec![
            out_grad ^ &inps[1].transpose(),
            &inps[0].transpose() ^ out_grad,
        ]
    }
}
