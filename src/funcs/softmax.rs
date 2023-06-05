use super::{Function, Tensor, TensorOps};

#[derive(Debug)]
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
        let jacobian = self.out.map(1, |l| {
            let n = l.shape()[0];
            let mut data = vec![0.; n * n];
            for i in 0..n {
                let si = l.blob()[i];
                for j in 0..i {
                    let sj = l.blob()[j];
                    data[i * n + j] = -si * sj;
                    data[j * n + i] = data[i * n + j];
                }
                data[i * n + i] = si * (1. - si);
            }
            Tensor::raw(&[n, n], data)
        });

        let out = &jacobian ^ &out_grad.unsqueeze(-1);
        vec![out.squeeze(-1).into()]
    }
}
