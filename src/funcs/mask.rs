use super::{Function, Tensor, TensorMutOps, TensorOps};
use rayon::prelude::*;

#[derive(Debug)]
pub struct Mask {
    mask: Tensor<f32>,
    value: f32,
}
impl Mask {
    pub fn new(mask: Tensor<bool>, value: f32) -> Box<dyn Function> {
        Box::new(Self {
            mask: (&mask).into(),
            value,
        })
    }
}

impl Function for Mask {
    fn run(&mut self, inps: &[&Tensor<f32>], _training: bool) -> Tensor<f32> {
        inps[0].map(self.mask.dim(), |t| {
            let dat = t
                .blob()
                .par_iter()
                .zip(self.mask.blob().par_iter())
                .map(|(v, m)| if *m == 1. { self.value } else { *v })
                .collect::<Vec<_>>();
            Tensor::raw(t.shape(), dat)
        })
    }
    fn grad(
        &self,
        _inps: &[&Tensor<f32>],
        _out: &Tensor<f32>,
        out_grad: &Tensor<f32>,
    ) -> Vec<Tensor<f32>> {
        vec![out_grad * &self.mask]
    }
}
