use super::Function;
use crate::tensor::*;

#[cfg(feature = "gpu")]
use super::{gpu, GpuFunction, TensorId};

#[derive(Debug, Clone)]
pub struct Dropout {
    mask: Tensor<f32>,
    rate: f32,
}
impl Dropout {
    pub fn new(rate: f32) -> Box<dyn Function> {
        Box::new(Self {
            rate,
            mask: Tensor::scalar(1.),
        })
    }
}

impl Function for Dropout {
    fn run(&mut self, inps: &[&GeneralTensor], training: bool) -> Result<Tensor<f32>, TensorError> {
        let inp = inps[0].as_float()?;
        Ok(if training {
            let mut rng = rand::thread_rng();
            let rnd = Tensor::<f32>::rand_range(&mut rng, 0., 1.0, inp.shape());
            let scale = 1. / (1. - self.rate);
            self.mask = rnd.map_values(|v| if v > self.rate { scale } else { 0. });
            (inp * &self.mask)?
        } else {
            self.mask = Tensor::scalar(1.);
            inp.clone()
        })
    }
    fn grad(
        &self,
        _inps: &[&GeneralTensor],
        out_grad: &Tensor<f32>,
    ) -> Result<Vec<Tensor<f32>>, TensorError> {
        Ok(vec![(out_grad * &self.mask)?])
    }
    fn clone_box(&self) -> Box<dyn Function> {
        Box::new(self.clone())
    }

    #[cfg(feature = "gpu")]
    fn gpu_impl(&self, out_id: TensorId, inps: &[Vec<usize>]) -> GpuFunction {
        gpu::dropout::gpu_impl(out_id, inps)
    }
}
