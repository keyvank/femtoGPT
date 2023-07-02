use super::Function;
use crate::tensor::*;

#[cfg(feature = "gpu")]
use super::{gpu, GpuFunction, TensorId};

#[derive(Debug, Clone)]
pub struct CrossEntropy {
    exp_output: Tensor<f32>,
}
impl CrossEntropy {
    pub fn new() -> Box<dyn Function> {
        Box::new(Self {
            exp_output: Tensor::scalar(0.),
        })
    }
}
impl Function for CrossEntropy {
    fn run(
        &mut self,
        inps: &[&GeneralTensor],
        _training: bool,
    ) -> Result<Tensor<f32>, TensorError> {
        let inp = inps[0].as_float()?;
        let target = inps[1].as_usize()?;

        self.exp_output = inp.map(1, |o| Ok(o.map_values(|f| f.exp())))?;

        Tensor::raw(
            target.shape(),
            inp.keep_right(1)?
                .inners()
                .iter()
                .zip(target.blob().iter())
                .zip(self.exp_output.keep_right(1)?.inners().iter())
                .map(|((o, t), o_exps)| {
                    let sum = o_exps.blob().iter().sum::<f32>();
                    let loss = sum.ln() - o.blob()[*t];
                    loss
                })
                .collect(),
        )
    }
    fn grad(
        &self,
        inps: &[&GeneralTensor],
        out_grad: &Tensor<f32>,
    ) -> Result<Vec<Tensor<f32>>, TensorError> {
        let inp = inps[0].as_float()?;
        let target = inps[1].as_usize()?;

        let classes = inp.shape()[inp.dim() - 1];

        Ok(vec![Tensor::raw(
            inp.shape(),
            self.exp_output
                .keep_right(1)?
                .inners()
                .iter()
                .zip(target.blob().iter())
                .zip(out_grad.blob().iter())
                .map(|((o_exps, t), g)| {
                    let o_exps = o_exps.blob();
                    let sum = o_exps.iter().sum::<f32>();
                    let sum_inv = 1. / sum;

                    let grad = (0..classes)
                        .map(|c| {
                            let val = o_exps[c];
                            (if *t == c {
                                val * sum_inv - 1.0
                            } else {
                                val * sum_inv
                            }) * g
                        })
                        .collect::<Vec<_>>();

                    grad
                })
                .flatten()
                .collect(),
        )?])
    }
    fn clone_box(&self) -> Box<dyn Function> {
        Box::new(self.clone())
    }

    #[cfg(feature = "gpu")]
    fn gpu_impl(&self, out_id: TensorId, inps: &[Vec<usize>]) -> GpuFunction {
        gpu::crossentropy::gpu_impl(out_id, inps)
    }
}
