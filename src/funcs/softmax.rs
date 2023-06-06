use super::{Function, Tensor, TensorOps};

#[derive(Debug, Clone)]
pub struct Softmax {
    out: Tensor<f32>,
}
impl Softmax {
    pub fn new() -> Box<dyn Function> {
        Box::new(Self {
            out: Tensor::scalar(0.),
        })
    }
}
impl Function for Softmax {
    fn run(&mut self, inps: &[&Tensor<f32>], _training: bool) -> Tensor<f32> {
        self.out = inps[0].map(1, |l| {
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

        self.out.clone()
    }
    fn grad(&self, _inps: &[&Tensor<f32>], out_grad: &Tensor<f32>) -> Vec<Tensor<f32>> {
        let grad_inp0 = self
            .out
            .keep_right(1)
            .inners()
            .iter()
            .zip(out_grad.keep_right(1).inners().iter())
            .map(|(l, o)| {
                let n = l.shape()[0];
                let mut data = vec![0.; n];
                for i in 0..n {
                    let si = l.blob()[i];
                    let mut sum = 0.;
                    for j in 0..n {
                        let sj = l.blob()[j];
                        sum += (if i == j { si * (1. - si) } else { -si * sj }) * o.blob()[j];
                    }
                    data[i] = sum;
                }
                data
            })
            .flatten()
            .collect::<Vec<_>>();
        vec![Tensor::raw(out_grad.shape(), grad_inp0)]
    }
    fn clone_box(&self) -> Box<dyn Function> {
        Box::new(self.clone())
    }
}
