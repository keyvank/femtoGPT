use super::Function;
use crate::tensor::*;

const SQRT_2_OVER_PI: f32 = 0.7978845608;
const GELU_CONST: f32 = 0.044715;

fn gelu(x: f32) -> f32 {
    0.5 * x * ((SQRT_2_OVER_PI * (x + GELU_CONST * x.powi(3))).tanh() + 1.)
}

fn gelu_prime(x: f32) -> f32 {
    let x3 = x.powi(3);
    let v = SQRT_2_OVER_PI * x + SQRT_2_OVER_PI * GELU_CONST * x3;
    let sech_2 = 1. / v.cosh().powi(2);
    0.5 + (0.398942 * x + 0.0535161 * x3) * sech_2 + 0.5 * v.tanh()
}

#[derive(Debug, Clone)]
pub struct Gelu;
impl Gelu {
    pub fn new() -> Box<dyn Function> {
        Box::new(Self {})
    }
}
impl Function for Gelu {
    fn run(&mut self, inps: &[&Tensor<f32>], _training: bool) -> Result<Tensor<f32>, TensorError> {
        Ok(inps[0].map_values(gelu))
    }
    fn grad(
        &self,
        inps: &[&Tensor<f32>],
        out_grad: &Tensor<f32>,
    ) -> Result<Vec<Tensor<f32>>, TensorError> {
        let der = inps[0].map_values(gelu_prime);
        Ok(vec![(&der * out_grad)?])
    }
    fn clone_box(&self) -> Box<dyn Function> {
        Box::new(self.clone())
    }
}
