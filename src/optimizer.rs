use serde::{Deserialize, Serialize};

use crate::tensor::{Tensor, TensorError, TensorOps};
use rayon::prelude::*;
use std::collections::HashMap;

/// Tracks the state of an optimizer across training steps.
///
/// This struct maintains the current step count and stores optimizer-specific
/// state tensors (such as momentum and velocity for Adam-based optimizers).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OptimizerState {
    pub step: usize,
    pub state: HashMap<String, Tensor<f32>>,
}

/// GPU-specific optimizer implementation details.
///
/// Contains the OpenCL kernel code and metadata needed to execute
/// optimizer steps on GPU hardware.
#[cfg(feature = "gpu")]
#[derive(Clone, Debug)]
pub struct GpuOptimizer {
    pub extra_buffers: HashMap<String, usize>,
    pub source_code: String,
    pub kernel_name: String,
}

/// Trait for implementing optimization algorithms.
///
/// This trait defines the interface for different optimizer implementations,
/// such as Adam, SGD, or other gradient-based optimization methods.
pub trait Optimizer: Clone + Serialize + serde::de::DeserializeOwned {
    /// Performs a single optimization step to update model parameters.
    ///
    /// # Arguments
    ///
    /// * `params` - A map of parameter names to tuples of (mutable parameter tensor, gradient tensor)
    /// * `optimizer_state` - Mutable reference to the optimizer state, tracking steps and internal buffers
    /// * `learning_rate` - The learning rate controlling the magnitude of parameter updates
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on success, or a `TensorError` if computation fails.
    fn step(
        &self,
        params: HashMap<String, (&mut Tensor<f32>, &Tensor<f32>)>,
        optimizer_state: &mut OptimizerState,
        learning_rate: f32,
    ) -> Result<(), TensorError>;

    #[cfg(feature = "gpu")]
    fn gpu_impl(&self, params: &HashMap<String, Vec<usize>>) -> GpuOptimizer;
}

/// Small constant used for numerical stability to prevent division by zero.
const EPSILON: f32 = 1e-8;

/// AdamW optimizer with weight decay regularization.
///
/// Implements the AdamW (Adam with decoupled Weight decay regularization) algorithm.
/// See: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
///
/// This optimizer combines adaptive learning rates (like Adam) with explicit
/// L2 weight decay, which improves generalization compared to L2 regularization.
#[derive(Clone, Serialize, Deserialize)]
pub struct AdamW {
    beta1: f32,
    beta2: f32,
    weight_decay: f32,
}

/// Creates a new AdamW optimizer with default hyperparameters.
///
/// Default values:
/// - beta1: 0.9
/// - beta2: 0.999
/// - weight_decay: 0.01
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

    #[cfg(feature = "gpu")]
    fn gpu_impl(&self, params: &HashMap<String, Vec<usize>>) -> GpuOptimizer {
        let source_code = "
        __kernel void optimizer(__global float *param,
                                __global float *grad,
                                __global float *m,
                                __global float *v,
                                float learning_rate,
                                ulong step,
                                ulong n) {
            uint id = get_global_id(0);
            param += id;
            grad += id;
            m += id;
            v += id;
            float beta1 = 0.9;
            float beta2 = 0.999;
            float weight_decay = 0.01;
            if(id < n) {
                *param = *param - *param * learning_rate * weight_decay;
                *m = beta1 * (*m) + (1 - beta1) * (*grad);
                *v = beta2 * (*v) + (1 - beta2) * (*grad) * (*grad);
                float m_hat = *m / (1.0 - pow(beta1, step + 1));
                float v_hat = *v / (1.0 - pow(beta2, step + 1));
                float v_hat_sqrt_inv = learning_rate / (sqrt(v_hat) + 1e-8);
                *param = *param - m_hat * v_hat_sqrt_inv;
            }
        }"
        .into();
        GpuOptimizer {
            source_code,
            extra_buffers: params
                .iter()
                .map(|(k, v)| {
                    let sz = v.iter().fold(1, |a, b| a * b);
                    vec![(k.clone() + "_m", sz), (k.clone() + "_v", sz)]
                })
                .flatten()
                .collect(),
            kernel_name: "optimizer".into(),
        }
    }
}
