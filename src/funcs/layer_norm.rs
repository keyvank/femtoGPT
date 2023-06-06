use super::{Function, Tensor, TensorOps};

#[derive(Debug, Clone)]
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
    fn grad(&self, inps: &[&Tensor<f32>], out_grad: &Tensor<f32>) -> Vec<Tensor<f32>> {
        let jacobian = inps[0].map(1, |l| {
            let inp1_blob = inps[1].blob();
            let n = l.size();
            let n_inv = 1. / n as f32;
            let avg = l.blob().iter().sum::<f32>() * n_inv;
            let sigma2 = l.blob().iter().map(|f| (f - avg).powi(2)).sum::<f32>() * n_inv + EPSILON;
            let sigma2_inv = 1. / sigma2;
            let sigma = sigma2.sqrt();
            let sigma_inv = 1. / sigma;
            let mut data = vec![0.; n * n];
            for i in 0..n {
                let a = l.blob()[i];
                let inp1_i = inp1_blob[i];
                for j in 0..i {
                    let b = l.blob()[j];
                    let val =
                        (-n_inv * sigma - (b - avg) * (a - avg) * sigma_inv * n_inv) * sigma2_inv;
                    data[i * n + j] = val * inp1_blob[j];
                    data[j * n + i] = val * inp1_i;
                }
                let val =
                    ((1. - n_inv) * sigma - (a - avg).powi(2) * sigma_inv * n_inv) * sigma2_inv;
                data[i * n + i] = val * inp1_i;
            }
            Tensor::raw(&[n, n], data)
        });
        let inp0_out = &jacobian ^ &out_grad.unsqueeze(-1);
        vec![
            inp0_out.squeeze(-1).into(),
            out_grad * &self.norm,
            out_grad.clone(),
        ]
    }
    fn clone_box(&self) -> Box<dyn Function> {
        Box::new(self.clone())
    }
}
