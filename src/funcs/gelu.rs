use super::Function;
use crate::tensor::*;

#[cfg(feature = "gpu")]
use super::{gpu, GpuFunction, TensorId};

const SQRT_2_OVER_PI: f32 = 0.7978845608;
const GELU_CONST: f32 = 0.044715;

/// Computes the Gaussian Error Linear Unit (GELU) activation function.
///
/// GELU is a smooth, non-linear activation function that has been shown to work
/// well in modern deep learning models, particularly in transformer architectures.
/// It approximates the input multiplied by the cumulative distribution function
/// of a Gaussian distribution.
///
/// The function uses the tanh-based approximation for computational efficiency:
/// ```text
/// GELU(x) = 0.5 * x * (tanh(sqrt(2/π) * (x + 0.044715 * x³)) + 1)
/// ```
fn gelu(x: f32) -> f32 {
    0.5 * x * ((SQRT_2_OVER_PI * (x + GELU_CONST * x.powi(3))).tanh() + 1.)
}

fn gelu_prime(x: f32) -> f32 {
    let x2 = x * x;
    let x3 = x2 * x;
    let v = SQRT_2_OVER_PI * x + SQRT_2_OVER_PI * GELU_CONST * x3;
    let v_prime = SQRT_2_OVER_PI + 3. * SQRT_2_OVER_PI * GELU_CONST * x2;
    let sech_2 = 1. / v.cosh().powi(2);
    0.5 * (1. + v.tanh() + x * sech_2 * v_prime)
}

#[derive(Debug, Clone)]
pub struct Gelu;
impl Gelu {
    pub fn new() -> Box<dyn Function> {
        Box::new(Self {})
    }
}
impl Function for Gelu {
    fn run(
        &mut self,
        inps: &[&GeneralTensor],
        _training: bool,
    ) -> Result<Tensor<f32>, TensorError> {
        Ok(inps[0].as_float()?.map_values(gelu))
    }
    fn grad(
        &self,
        inps: &[&GeneralTensor],
        out_grad: &Tensor<f32>,
    ) -> Result<Vec<Tensor<f32>>, TensorError> {
        let der = inps[0].as_float()?.map_values(gelu_prime);
        Ok(vec![(&der * out_grad)?])
    }
    fn clone_box(&self) -> Box<dyn Function> {
        Box::new(self.clone())
    }

    #[cfg(feature = "gpu")]
    fn gpu_impl(&self, out_id: TensorId, inps: &[Vec<usize>]) -> GpuFunction {
        gpu::gelu::gpu_impl(out_id, inps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // For calculating high-accuracy numerical derivative
    fn gelu_f64(x: f64) -> f64 {
        0.5 * x * ((SQRT_2_OVER_PI as f64 * (x + GELU_CONST as f64 * x.powi(3))).tanh() + 1.)
    }

    #[test]
    fn test_derivative() {
        const EPSILON: f64 = 1e-5;

        let mut x = -5.;
        while x < 5. {
            let numeric = ((gelu_f64(x as f64 + EPSILON) - gelu_f64(x as f64 - EPSILON))
                / EPSILON
                / 2.) as f32;
            let symbolic = gelu_prime(x);
            let diff = numeric / symbolic;
            x += 0.01;
            assert!(diff > 0.99 && diff < 1.01);
        }
    }
}
