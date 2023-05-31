use super::{Function, Tensor, TensorOps};

#[derive(Debug)]
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
    fn run(&mut self, inps: &[&Tensor<f32>], training: bool) -> Tensor<f32> {
        if training {
            let mut rng = rand::thread_rng();
            let rnd = Tensor::<f32>::rand_range(&mut rng, 0., 1.0, inps[0].shape());
            let scale = 1. / (1. - self.rate);
            self.mask = rnd.map_values(|v| if v > self.rate { scale } else { 0. });
            inps[0] * &self.mask
        } else {
            self.mask = Tensor::scalar(1.);
            inps[0].clone()
        }
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
