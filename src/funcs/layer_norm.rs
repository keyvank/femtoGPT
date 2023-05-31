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
            let size_inv = 1. / l.size() as f32;
            let avg = l.blob().iter().sum::<f32>() * size_inv;
            let var = (l.blob().iter().map(|f| (f - avg).powi(2)).sum::<f32>() * size_inv
                + EPSILON)
                .sqrt();
            let var_inv = 1. / var;
            Tensor::raw(
                l.shape(),
                l.blob().iter().map(|v| (v - avg) * var_inv).collect(),
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
            let nf_inv = 1. / nf;
            let avg = l.blob().iter().sum::<f32>() / nf;
            let sigma2 = l.blob().iter().map(|f| (f - avg).powi(2)).sum::<f32>() / nf + EPSILON;
            let sigma2_inv = 1. / sigma2;
            let sigma = sigma2.sqrt();
            let sigma_inv = 1. / sigma;
            Tensor::raw(
                &[n, n],
                (0..n * n)
                    .into_par_iter()
                    .map(|work| {
                        let i = work / n;
                        let a = l.blob()[i];
                        let j = work % n;
                        if i == j {
                            ((1. - nf_inv) * sigma - (a - avg).powi(2) * sigma_inv * nf_inv)
                                * sigma2_inv
                        } else {
                            let b = l.blob()[j];
                            (-nf_inv * sigma - (b - avg) * (a - avg) * sigma_inv * nf_inv)
                                * sigma2_inv
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
