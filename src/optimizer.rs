use serde::{Deserialize, Serialize};

use crate::tensor::{Tensor, TensorOps};
use std::collections::HashMap;
pub trait Optimizer: Serialize + serde::de::DeserializeOwned {
    fn step(&mut self, params: Vec<&mut Tensor<f32>>, grads: Vec<&Tensor<f32>>);
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Naive {
    learning_rate: f32,
}

impl Naive {
    pub fn new(learning_rate: f32) -> Self {
        Self { learning_rate }
    }
}

impl Optimizer for Naive {
    fn step(&mut self, params: Vec<&mut Tensor<f32>>, grads: Vec<&Tensor<f32>>) {
        for (param, grad) in params.into_iter().zip(grads.into_iter()) {
            *param = &*param + &(grad * &Tensor::scalar(-self.learning_rate));
        }
    }
}

const EPSILON: f32 = 1e-8;

#[derive(Clone, Serialize, Deserialize)]
pub struct AdamW {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    weight_decay: f32,
    m_v: HashMap<usize, (Tensor<f32>, Tensor<f32>)>,
    t: usize,
}

impl AdamW {
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            weight_decay: 0.01,
            m_v: HashMap::new(),
            t: 1,
        }
    }
}

impl Optimizer for AdamW {
    fn step(&mut self, params: Vec<&mut Tensor<f32>>, grads: Vec<&Tensor<f32>>) {
        println!("{}", self.t);
        for (t, (param, grad)) in params.into_iter().zip(grads.into_iter()).enumerate() {
            let mv = self
                .m_v
                .entry(t)
                .or_insert((Tensor::scalar(0.), Tensor::scalar(0.)));

            // Weight decay
            *param = &*param - &(&*param * &Tensor::scalar(self.learning_rate * self.weight_decay));

            mv.0 =
                &(&Tensor::scalar(self.beta1) * &mv.0) + &(&Tensor::scalar(1. - self.beta1) * grad);
            mv.1 = &(&Tensor::scalar(self.beta2) * &mv.1)
                + &(&(&Tensor::scalar(1. - self.beta2) * grad) * grad);
            let m_hat = &mv.0 * &Tensor::scalar(1. / (1. - self.beta1.powi(self.t as i32)));
            let v_hat = &mv.1 * &Tensor::scalar(1. / (1. - self.beta2.powi(self.t as i32)));

            let v_hat_sqrt_inv = v_hat.map_values(|f| self.learning_rate / (f.sqrt() + EPSILON));

            *param = &*param - &(&m_hat * &v_hat_sqrt_inv);
        }
        self.t += 1;
    }
}
