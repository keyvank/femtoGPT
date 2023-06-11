use serde::{Deserialize, Serialize};

use crate::tensor::{Tensor, TensorError, TensorOps};
use rayon::prelude::*;

pub trait Optimizer: Serialize + serde::de::DeserializeOwned {
    fn step_num(&self) -> usize;
    fn step(
        &mut self,
        params: Vec<&mut Tensor<f32>>,
        grads: Vec<&Tensor<f32>>,
        learning_rate: f32,
    ) -> Result<(), TensorError>;
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Naive {
    t: usize,
}

impl Naive {
    pub fn new() -> Self {
        Self { t: 0 }
    }
}

// Simplest optimizer possible, just reduce gradients
// from params with a learning_rate coefficient
impl Optimizer for Naive {
    fn step_num(&self) -> usize {
        self.t
    }
    fn step(
        &mut self,
        params: Vec<&mut Tensor<f32>>,
        grads: Vec<&Tensor<f32>>,
        learning_rate: f32,
    ) -> Result<(), TensorError> {
        for (param, grad) in params.into_iter().zip(grads.into_iter()) {
            *param = (&*param + &(grad * &Tensor::scalar(-learning_rate))?)?;
        }
        Ok(())
    }
}

const EPSILON: f32 = 1e-8;

#[derive(Clone, Serialize, Deserialize)]
pub struct AdamW {
    beta1: f32,
    beta2: f32,
    weight_decay: f32,
    m: Vec<Tensor<f32>>,
    v: Vec<Tensor<f32>>,
    t: usize,
}

impl AdamW {
    pub fn new() -> Self {
        Self {
            beta1: 0.9,
            beta2: 0.999,
            weight_decay: 0.01,
            m: Default::default(),
            v: Default::default(),
            t: 0,
        }
    }
}

// Adam optimizer with weight decay!
// https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
impl Optimizer for AdamW {
    fn step_num(&self) -> usize {
        self.t
    }
    fn step(
        &mut self,
        params: Vec<&mut Tensor<f32>>,
        grads: Vec<&Tensor<f32>>,
        learning_rate: f32,
    ) -> Result<(), TensorError> {
        if self.m.len() == 0 || self.v.len() == 0 {
            self.m = vec![Tensor::scalar(0.); params.len()];
            self.v = vec![Tensor::scalar(0.); params.len()];
        }
        params
            .into_par_iter()
            .zip(grads.into_par_iter())
            .zip(self.m.par_iter_mut())
            .zip(self.v.par_iter_mut())
            .map(|(((param, grad), m), v)| {
                // Weight decay
                *param =
                    (&*param - &(&*param * &Tensor::scalar(learning_rate * self.weight_decay))?)?;

                *m = (&(&Tensor::scalar(self.beta1) * &*m)?
                    + &(&Tensor::scalar(1. - self.beta1) * grad)?)?;
                *v = (&(&Tensor::scalar(self.beta2) * &*v)?
                    + &(&(&Tensor::scalar(1. - self.beta2) * grad)? * grad)?)?;
                let m_hat =
                    (&*m * &Tensor::scalar(1. / (1. - self.beta1.powi(self.t as i32 + 1))))?;
                let v_hat =
                    (&*v * &Tensor::scalar(1. / (1. - self.beta2.powi(self.t as i32 + 1))))?;

                let v_hat_sqrt_inv = v_hat.map_values(|f| learning_rate / (f.sqrt() + EPSILON));

                *param = (&*param - &(&m_hat * &v_hat_sqrt_inv)?)?;
                Ok(())
            })
            .collect::<Result<Vec<()>, TensorError>>()?;
        self.t += 1;
        Ok(())
    }
}
