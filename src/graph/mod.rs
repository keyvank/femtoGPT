#[cfg(feature = "gpu")]
pub mod gpu;

use crate::funcs::{Function, Loss};
use crate::optimizer::{Optimizer, OptimizerState};
use crate::tensor::*;
use std::collections::{BTreeMap, HashMap};
use thiserror::Error;

pub type TensorId = usize;

pub trait Graph {
    fn alloc(
        &mut self,
        t: Tensor<f32>,
        is_param: bool,
        name: String,
    ) -> Result<TensorId, GraphError>;
    fn alloc_usize(&mut self, t: Tensor<usize>, name: String) -> Result<TensorId, GraphError>;
    fn params(&self) -> &[TensorId];
    fn load<T: TensorOps<f32>>(
        &mut self,
        tensor_id: TensorId,
        tensor: &T,
    ) -> Result<(), GraphError>;
    fn load_usize<T: TensorOps<usize>>(
        &mut self,
        tensor_id: TensorId,
        tensor: &T,
    ) -> Result<(), GraphError>;
    fn load_grad<T: TensorOps<f32>>(
        &mut self,
        tensor_id: TensorId,
        tensor: &T,
    ) -> Result<(), GraphError>;
    fn zero_grad(&mut self) -> Result<(), GraphError>;
    fn name_of(&self, id: TensorId) -> Result<&String, GraphError>;
    fn fetch(&mut self, id: TensorId, grad: bool) -> Result<(), GraphError>;
    fn get(&self, id: TensorId) -> Result<&GeneralTensor, GraphError>;
    fn get_grad(&self, id: TensorId) -> Result<&Tensor<f32>, GraphError>;
    fn backward_all(
        &mut self,
        id: TensorId,
        loss_fn: Box<dyn Loss>,
        limit: Option<usize>,
    ) -> Result<f32, GraphError>;
    fn forward(&mut self, training: bool) -> Result<(), GraphError>;
    fn call(
        &mut self,
        f: Box<dyn Function>,
        tensor_ids: &[TensorId],
    ) -> Result<TensorId, GraphError>;
    fn optimize<O: Optimizer>(
        &mut self,
        optimizer: &O,
        learning_rate: f32,
    ) -> Result<(), GraphError>;
    fn optimizer_step(&self) -> usize;
    fn get_optimizer_state(&self) -> Result<OptimizerState, GraphError>;
    fn set_optimizer_state(&mut self, state: &OptimizerState) -> Result<(), GraphError>;
}

unsafe impl Send for CpuGraph {}
unsafe impl Sync for CpuGraph {}

struct Computation {
    inps: Vec<TensorId>,
    func: Box<dyn Function>,
}

impl Clone for Computation {
    fn clone(&self) -> Self {
        Self {
            inps: self.inps.clone(),
            func: self.func.clone_box(),
        }
    }
}

unsafe impl Send for Computation {}
unsafe impl Sync for Computation {}

#[derive(Clone)]
pub struct CpuGraph {
    tensors: Vec<GeneralTensor>,
    grads: Vec<Tensor<f32>>,
    names: Vec<String>,
    params: Vec<TensorId>,
    computations: BTreeMap<TensorId, Computation>,
    optimizer_state: OptimizerState,
}

#[derive(Error, Debug)]
pub enum GraphError {
    #[error("tensor error: {0}")]
    TensorError(#[from] TensorError),
    #[error("tensor with id {0} not found")]
    TensorNotFound(usize),
    #[error("graph is not ready!")]
    NotReady,
    #[error("tensor types incompatible!")]
    IncompatibleTypes,

    #[cfg(feature = "gpu")]
    #[error("gpu error: {0}")]
    GpuError(#[from] gpu::program::ProgramError),
}

#[cfg(feature = "gpu")]
impl From<ocl::Error> for GraphError {
    fn from(error: ocl::Error) -> Self {
        GraphError::GpuError(gpu::program::ProgramError::from(error))
    }
}

impl CpuGraph {
    fn add_grad<T: TensorOps<f32>>(&mut self, id: TensorId, add: T) -> Result<(), GraphError> {
        // Usize tensors do not have gradient
        if self.get(id)?.as_float().is_err() {
            return Ok(());
        }

        let shape = self.get(id)?.as_float()?.shape().to_vec();
        let grad = self
            .grads
            .get_mut(id)
            .ok_or(GraphError::TensorNotFound(id))?;
        if add.dim() >= shape.len() {
            for t in add.keep_right(shape.len())?.inners().iter() {
                *grad = (&*grad + t)?;
            }
        } else {
            *grad = (&*grad + &add.view())?;
        }
        Ok(())
    }
}

impl Graph for CpuGraph {
    fn alloc_usize(&mut self, t: Tensor<usize>, name: String) -> Result<TensorId, GraphError> {
        self.grads.push(Tensor::zeros(t.shape()));
        self.tensors.push(GeneralTensor::Usize(t));
        self.names.push(name);
        Ok(self.tensors.len() - 1)
    }
    fn alloc(
        &mut self,
        t: Tensor<f32>,
        is_param: bool,
        name: String,
    ) -> Result<TensorId, GraphError> {
        self.grads.push(Tensor::zeros(t.shape()));
        self.tensors.push(GeneralTensor::Float(t));
        self.names.push(name);
        let id = self.tensors.len() - 1;
        if is_param {
            self.params.push(id);
        }
        Ok(id)
    }
    fn load<T: TensorOps<f32>>(
        &mut self,
        tensor_id: TensorId,
        tensor: &T,
    ) -> Result<(), GraphError> {
        self.tensors[tensor_id] = GeneralTensor::Float(tensor.view().into());
        Ok(())
    }
    fn load_usize<T: TensorOps<usize>>(
        &mut self,
        tensor_id: TensorId,
        tensor: &T,
    ) -> Result<(), GraphError> {
        self.tensors[tensor_id] = GeneralTensor::Usize(tensor.view().into());
        Ok(())
    }
    fn load_grad<T: TensorOps<f32>>(
        &mut self,
        tensor_id: TensorId,
        tensor: &T,
    ) -> Result<(), GraphError> {
        self.grads[tensor_id] = tensor.view().into();
        Ok(())
    }
    fn zero_grad(&mut self) -> Result<(), GraphError> {
        self.grads.iter_mut().for_each(|t| {
            t.fill(0.);
        });
        Ok(())
    }

    fn name_of(&self, id: TensorId) -> Result<&String, GraphError> {
        self.names.get(id).ok_or(GraphError::TensorNotFound(id))
    }
    fn get(&self, id: TensorId) -> Result<&GeneralTensor, GraphError> {
        self.tensors.get(id).ok_or(GraphError::TensorNotFound(id))
    }
    fn get_grad(&self, id: TensorId) -> Result<&Tensor<f32>, GraphError> {
        self.grads.get(id).ok_or(GraphError::TensorNotFound(id))
    }
    fn backward_all(
        &mut self,
        id: TensorId,
        loss_fn: Box<dyn Loss>,
        limit: Option<usize>,
    ) -> Result<f32, GraphError> {
        let output = self.get(id)?;
        let (loss, grad) = loss_fn.run(output)?;
        let mean_coeff = 1. / loss.size() as f32;
        self.add_grad(id, (&grad * &Tensor::scalar(mean_coeff))?)?;

        for (i, (id, comp)) in self.computations.clone().iter().rev().enumerate() {
            if let Some(limit) = limit {
                if i >= limit {
                    break;
                }
            }
            let inps = comp
                .inps
                .iter()
                .map(|id| &self.tensors[*id])
                .collect::<Vec<_>>();
            let grad_out = &self.grads[*id];
            let grads = comp.func.grad(&inps, grad_out)?;
            for (id, grad) in comp.inps.clone().into_iter().zip(grads.into_iter()) {
                self.add_grad(id, grad)?;
            }
        }

        Ok(loss.mean())
    }
    fn forward(&mut self, training: bool) -> Result<(), GraphError> {
        for (out, c) in self.computations.iter_mut() {
            let tensors = c
                .inps
                .iter()
                .map(|id| self.tensors.get(*id).ok_or(GraphError::TensorNotFound(*id)))
                .collect::<Result<Vec<_>, GraphError>>()?;
            let result = c.func.run(&tensors, training)?;
            self.tensors[*out] = GeneralTensor::Float(result);
        }
        Ok(())
    }
    fn call(
        &mut self,
        mut f: Box<dyn Function>,
        tensor_ids: &[TensorId],
    ) -> Result<TensorId, GraphError> {
        let tensors = tensor_ids
            .iter()
            .map(|id| self.get(*id))
            .collect::<Result<Vec<_>, GraphError>>()?;
        let out = f.run(&tensors, false)?;
        let child = self.alloc(out, false, "".into())?;
        self.computations.insert(
            child,
            Computation {
                func: f,
                inps: tensor_ids.to_vec(),
            },
        );
        Ok(child)
    }
    fn optimize<O: Optimizer>(
        &mut self,
        optimizer: &O,
        learning_rate: f32,
    ) -> Result<(), GraphError> {
        let pg = self
            .tensors
            .iter_mut()
            .enumerate()
            .filter(|(id, _)| self.params.contains(id))
            .map(|(id, params)| {
                let name = self
                    .names
                    .get(id)
                    .cloned()
                    .ok_or(GraphError::TensorNotFound(id))?;
                let grad = self.grads.get(id).ok_or(GraphError::TensorNotFound(id))?;
                Ok((name, (params.as_float_mut()?, grad)))
            })
            .collect::<Result<HashMap<String, (&mut Tensor<f32>, &Tensor<f32>)>, GraphError>>()?;
        optimizer.step(pg, &mut self.optimizer_state, learning_rate)?;
        Ok(())
    }
    fn fetch(&mut self, _tensor_id: TensorId, _grad: bool) -> Result<(), GraphError> {
        // All tensors are ready by default in a CPU graph!
        Ok(())
    }
    fn params(&self) -> &[TensorId] {
        &self.params
    }
    fn optimizer_step(&self) -> usize {
        self.optimizer_state.step
    }
    fn get_optimizer_state(&self) -> Result<OptimizerState, GraphError> {
        Ok(self.optimizer_state.clone())
    }
    fn set_optimizer_state(&mut self, state: &OptimizerState) -> Result<(), GraphError> {
        self.optimizer_state = state.clone();
        Ok(())
    }
}

impl CpuGraph {
    pub fn new() -> Self {
        Self {
            tensors: Default::default(),
            grads: Default::default(),
            computations: Default::default(),
            params: Default::default(),
            names: Default::default(),
            optimizer_state: Default::default(),
        }
    }
}
