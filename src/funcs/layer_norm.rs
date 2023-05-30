use super::{Function, Tensor, TensorOps};
use rayon::prelude::*;

#[derive(Debug)]
pub struct LayerNorm {
    norm: Tensor<f32>,
}
impl LayerNorm {
    pub fn new() -> Box<dyn Function> {
        Box::new(Self {
            norm: Tensor::scalar(0.),
        })
    }
}

const EPSILON: f32 = 1e-5;

impl Function for LayerNorm {
    fn run(&mut self, inps: &[&Tensor<f32>], _training: bool) -> Tensor<f32> {
        self.norm = inps[0].map(1, |l| {
            let avg = l.blob().iter().sum::<f32>() / l.size() as f32;
            let var = (l.blob().iter().map(|f| (f - avg).powi(2)).sum::<f32>() / l.size() as f32
                + EPSILON)
                .sqrt();
            Tensor::raw(
                l.shape(),
                l.blob().iter().map(|v| (v - avg) / var).collect(),
            )
        });
        &(&self.norm * inps[1]) + inps[2]
    }
    fn grad(
        &self,
        inps: &[&Tensor<f32>],
        _out: &Tensor<f32>,
        out_grad: &Tensor<f32>,
    ) -> Vec<Tensor<f32>> {
        let jacobian = inps[0].map(1, |l| {
            let n = l.size();
            let nf = n as f32;
            let avg = l.blob().iter().sum::<f32>() / nf;
            let sigma2 = l.blob().iter().map(|f| (f - avg).powi(2)).sum::<f32>() / nf + EPSILON;
            let sigma = sigma2.sqrt();
            Tensor::raw(
                &[n, n],
                (0..n * n)
                    .into_par_iter()
                    .map(|work| {
                        let i = work / n;
                        let j = work % n;
                        if i == j {
                            let a = l.blob()[i];
                            ((1. - 1. / nf) * sigma - (a - avg).powi(2) / sigma / nf) / sigma2
                        } else {
                            let a = l.blob()[i];
                            let b = l.blob()[j];
                            (-1. / nf * sigma - (b - avg) * (a - avg) / sigma / nf) / sigma2
                        }
                    })
                    .collect(),
            )
        });
        let inp0_out = &(inps[1] * &jacobian) ^ &out_grad.unsqueeze(-1);
        vec![
            inp0_out.squeeze(-1).into(),
            out_grad * &self.norm,
            out_grad.clone(),
        ]
    }
}
