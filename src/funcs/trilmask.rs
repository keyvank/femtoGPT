use super::Function;
use crate::tensor::*;

#[cfg(feature = "gpu")]
use super::{gpu, GpuFunction, GpuFunctionGroup, TensorId};

#[derive(Debug, Clone)]
pub struct TrilMask {
    n: usize,
    value: f32,
}
impl TrilMask {
    pub fn new(n: usize, value: f32) -> Box<dyn Function> {
        Box::new(Self { n, value })
    }
}

impl Function for TrilMask {
    fn run(
        &mut self,
        inps: &[&GeneralTensor],
        _training: bool,
    ) -> Result<Tensor<f32>, TensorError> {
        inps[0].as_float()?.map(2, |t| {
            let t_blob = t.blob();
            let mut dat = Vec::with_capacity(self.n * self.n);
            for i in 0..self.n {
                for j in 0..self.n {
                    dat.push(if j <= i {
                        t_blob[i * self.n + j]
                    } else {
                        self.value
                    });
                }
            }
            Ok(Tensor::raw(&[self.n, self.n], dat)?)
        })
    }
    fn grad(
        &self,
        _inps: &[&GeneralTensor],
        out_grad: &Tensor<f32>,
    ) -> Result<Vec<Tensor<f32>>, TensorError> {
        Ok(vec![out_grad.map(2, |t| {
            let t_blob = t.blob();
            let mut dat = Vec::with_capacity(self.n * self.n);
            for i in 0..self.n {
                for j in 0..self.n {
                    dat.push(if j <= i { t_blob[i * self.n + j] } else { 0. });
                }
            }
            Ok(Tensor::raw(&[self.n, self.n], dat)?)
        })?])
    }
    fn clone_box(&self) -> Box<dyn Function> {
        Box::new(self.clone())
    }

    #[cfg(feature = "gpu")]
    fn gpu_run(&self, out_id: TensorId, inps: &[Vec<usize>]) -> GpuFunction {
        gpu::trilmask::gpu_run(out_id, inps, self.n, self.value)
    }

    #[cfg(feature = "gpu")]
    fn gpu_grad(&self, out_id: TensorId, inps: &[Vec<usize>]) -> GpuFunctionGroup {
        gpu::trilmask::gpu_grad(out_id, inps, self.n)
    }
}
