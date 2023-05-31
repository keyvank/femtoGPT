use super::{Function, Tensor, TensorOps};
use rayon::prelude::*;

#[derive(Debug)]
pub struct Softmax;
impl Softmax {
    pub fn new() -> Box<dyn Function> {
        Box::new(Self {})
    }
}
impl Function for Softmax {
    fn run(&mut self, inps: &[&Tensor<f32>], _training: bool) -> Tensor<f32> {
        let result = inps[0].map(1, |l| {
            let max = l
                .blob()
                .iter()
                .fold(f32::NEG_INFINITY, |a, b| f32::max(a, *b));
            let sum = l
                .map_values(|f| (f - max).exp())
                .inners()
                .iter()
                .map(|t| t.scalar())
                .sum::<f32>();
            l.map_values(|f| (f - max).exp() / sum)
        });
        result
    }
    fn grad(
        &self,
        _inps: &[&Tensor<f32>],
        out: &Tensor<f32>,
        out_grad: &Tensor<f32>,
    ) -> Vec<Tensor<f32>> {
        let jacobian = out.map(1, |l| {
            let n = l.shape()[0];
            Tensor::jacobian(n, |i, j| {
                let si = l.blob()[i];
                if i == j {
                    si * (1. - si)
                } else {
                    let sj = l.blob()[j];
                    -si * sj
                }
            })
        });

        let out = &jacobian ^ &out_grad.unsqueeze(-1);
        vec![out.squeeze(-1).into()]
    }
}
