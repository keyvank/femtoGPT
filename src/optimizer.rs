use serde::{Deserialize, Serialize};

use crate::tensor::{Tensor, TensorError, TensorOps};
use rayon::prelude::*;
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OptimizerState {
    pub step: usize,
    pub state: HashMap<String, Tensor<f32>>,
}

pub trait Optimizer: Clone + Serialize + serde::de::DeserializeOwned {
    fn step(
        &self,
        params: HashMap<String, (&mut Tensor<f32>, &Tensor<f32>)>,
        optimizer_state: &mut OptimizerState,
        learning_rate: f32,
    ) -> Result<(), TensorError>;
}

const EPSILON: f32 = 1e-8;

#[derive(Clone, Serialize, Deserialize)]
pub struct AdamW {
    beta1: f32,
    beta2: f32,
    weight_decay: f32,
}

impl AdamW {
    pub fn new() -> Self {
        Self {
            beta1: 0.9,
            beta2: 0.999,
            weight_decay: 0.01,
        }
    }
}

// Adam optimizer with weight decay!
// https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
impl Optimizer for AdamW {
    fn step(
        &self,
        params: HashMap<String, (&mut Tensor<f32>, &Tensor<f32>)>,
        optimizer_state: &mut OptimizerState,
        learning_rate: f32,
    ) -> Result<(), TensorError> {
        for (name, m, v) in params
            .into_par_iter()
            .map(|(name, (param, grad))| {
                let m_key = format!("{}_m", name);
                let v_key = format!("{}_v", name);
                let mut m = optimizer_state
                    .state
                    .get(&m_key)
                    .cloned()
                    .unwrap_or(Tensor::zeros(param.shape()));
                let mut v = optimizer_state
                    .state
                    .get(&v_key)
                    .cloned()
                    .unwrap_or(Tensor::zeros(param.shape()));

                // Weight decay
                *param =
                    (&*param - &(&*param * &Tensor::scalar(learning_rate * self.weight_decay))?)?;

                m = (&(&Tensor::scalar(self.beta1) * &m)?
                    + &(&Tensor::scalar(1. - self.beta1) * grad)?)?;
                v = (&(&Tensor::scalar(self.beta2) * &v)?
                    + &(&(&Tensor::scalar(1. - self.beta2) * grad)? * grad)?)?;

                let m_hat = (&m
                    * &Tensor::scalar(
                        1. / (1. - self.beta1.powi(optimizer_state.step as i32 + 1)),
                    ))?;
                let v_hat = (&v
                    * &Tensor::scalar(
                        1. / (1. - self.beta2.powi(optimizer_state.step as i32 + 1)),
                    ))?;

                let v_hat_sqrt_inv = v_hat.map_values(|f| learning_rate / (f.sqrt() + EPSILON));

                *param = (&*param - &(&m_hat * &v_hat_sqrt_inv)?)?;
                Ok((name, m, v))
            })
            .collect::<Result<Vec<_>, TensorError>>()?
        {
            let m_key = format!("{}_m", name);
            let v_key = format!("{}_v", name);
            optimizer_state.state.insert(m_key, m);
            optimizer_state.state.insert(v_key, v);
        }
        optimizer_state.step += 1;
        Ok(())
    }
}
