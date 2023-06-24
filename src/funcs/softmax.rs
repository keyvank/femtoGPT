use super::Function;
use crate::tensor::*;

#[cfg(feature = "gpu")]
use super::{gpu, GpuFunction, TensorId};

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
    fn run(
        &mut self,
        inps: &[&GeneralTensor],
        _training: bool,
    ) -> Result<Tensor<f32>, TensorError> {
        let inps = inps
            .iter()
            .map(|t| t.as_float())
            .collect::<Result<Vec<_>, TensorError>>()?;
        self.out = inps[0].map(1, |l| {
            let max = l
                .blob()
                .iter()
                .fold(f32::NEG_INFINITY, |a, b| f32::max(a, *b));
            let sum = l.blob().iter().map(|f| (f - max).exp()).sum::<f32>();
            Ok(l.map_values(|f| (f - max).exp() / sum))
        })?;

        Ok(self.out.clone())
    }
    fn grad(
        &self,
        _inps: &[&GeneralTensor],
        out_grad: &Tensor<f32>,
    ) -> Result<Vec<Tensor<f32>>, TensorError> {
        let grad_inp0 = self
            .out
            .keep_right(1)?
            .inners()
            .iter()
            .zip(out_grad.keep_right(1)?.inners().iter())
            .map(|(l, o)| {
                let l_blob = l.blob();
                let o_blob = o.blob();
                let n = l.shape()[0];
                let mut data = vec![0.; n];
                for i in 0..n {
                    let si = l_blob[i];
                    let mut sum = 0.;
                    for j in 0..n {
                        let sj = l_blob[j];
                        sum += (if i == j { si * (1. - si) } else { -si * sj }) * o_blob[j];
                    }
                    data[i] = sum;
                }
                data
            })
            .flatten()
            .collect::<Vec<_>>();
        Ok(vec![Tensor::raw(out_grad.shape(), grad_inp0)?])
    }
    fn clone_box(&self) -> Box<dyn Function> {
        Box::new(self.clone())
    }

    #[cfg(feature = "gpu")]
    fn gpu_impl(&self, out_id: TensorId, inps: &[Vec<usize>]) -> GpuFunction {
        gpu::softmax::gpu_impl(out_id, inps)
    }
}
