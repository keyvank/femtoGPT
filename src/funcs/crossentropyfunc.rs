use super::Function;
use crate::tensor::*;

#[cfg(feature = "gpu")]
use super::{GpuFunction, TensorId};

#[derive(Debug, Clone)]
pub struct CrossEntropyFunc;
impl CrossEntropyFunc {
    pub fn new() -> Box<dyn Function> {
        Box::new(Self {})
    }
}
impl Function for CrossEntropyFunc {
    fn run(
        &mut self,
        inps: &[&GeneralTensor],
        _training: bool,
    ) -> Result<Tensor<f32>, TensorError> {
        let inp = inps[0].as_float()?;
        let target = inps[1].as_usize()?;

        Tensor::raw(
            target.shape(),
            inp.keep_right(1)?
                .inners()
                .iter()
                .zip(target.blob().iter())
                .map(|(o, t)| {
                    let o_exps = o.blob().iter().map(|f| f.exp()).collect::<Vec<_>>();
                    let sum = o_exps.iter().sum::<f32>();
                    let loss = sum.ln() - o.blob()[*t];
                    loss
                })
                .collect(),
        )
    }
    fn grad(
        &self,
        inps: &[&GeneralTensor],
        _out_grad: &Tensor<f32>,
    ) -> Result<Vec<Tensor<f32>>, TensorError> {
        let inp = inps[0].as_float()?;
        let target = inps[1].as_usize()?;

        let classes = inp.shape()[inp.dim() - 1];

        Ok(vec![Tensor::raw(
            inp.shape(),
            inp.keep_right(1)?
                .inners()
                .iter()
                .zip(target.blob().iter())
                .map(|(o, t)| {
                    let o_exps = o.blob().iter().map(|f| f.exp()).collect::<Vec<_>>();
                    let sum = o_exps.iter().sum::<f32>();
                    let sum_inv = 1. / sum;

                    let grad = (0..classes)
                        .map(|c| {
                            let val = o_exps[c];
                            if *t == c {
                                val * sum_inv - 1.0
                            } else {
                                val * sum_inv
                            }
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
    fn gpu_impl(&self, _out_id: TensorId, _inps: &[Vec<usize>]) -> GpuFunction {
        unimplemented!()
    }
}
