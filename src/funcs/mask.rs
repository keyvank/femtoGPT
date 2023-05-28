use super::{Function, Tensor, TensorMutOps, TensorOps};

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
    fn run(&mut self, inps: &[&Tensor<f32>]) -> Tensor<f32> {
        let mut reshaped = self.mask.shape().to_vec();
        reshaped.insert(0, 0);
        let mut result = inps[0].clone();
        for mut t in result.reshape_mut(&reshaped).iter_mut() {
            for (r, mask) in t.blob_mut().iter_mut().zip(self.mask.blob().iter()) {
                if *mask == 1. {
                    *r = self.value;
                }
            }
        }
        result
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
