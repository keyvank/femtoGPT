use crate::funcs::{Function, Loss};
use crate::optimizer::Optimizer;
use crate::tensor::*;
use rand::Rng;
use std::collections::{BTreeMap, HashSet};

pub type TensorId = usize;

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
pub struct Graph {
    tensors: BTreeMap<TensorId, Tensor<f32>>,
    grads: BTreeMap<TensorId, Tensor<f32>>,
    computations: BTreeMap<TensorId, Computation>,
}

impl Graph {
    pub fn new() -> Self {
        Self {
            tensors: Default::default(),
            grads: Default::default(),
            computations: Default::default(),
        }
    }
    pub fn alloc_rand<R: Rng>(&mut self, rng: &mut R, shape: &[usize]) -> TensorId {
        self.alloc(Tensor::<f32>::rand(rng, shape))
    }
    fn alloc(&mut self, t: Tensor<f32>) -> TensorId {
        let id = self.tensors.len();
        self.tensors.insert(id, t);
        id
    }
    pub fn load<T: TensorOps<f32>>(&mut self, tensor_id: TensorId, tensor: &T) {
        self.tensors.insert(tensor_id, tensor.view().into());
    }
    pub fn load_grad<T: TensorOps<f32>>(&mut self, tensor_id: TensorId, tensor: &T) {
        self.grads.insert(tensor_id, tensor.view().into());
    }
    pub fn zero_grad(&mut self) {
        self.grads.clear();
    }
    pub fn add_grad<T: TensorOps<f32>>(&mut self, id: TensorId, add: T) -> Result<(), TensorError> {
        let shape = self.get(id).shape().to_vec();
        let grad = self.grads.entry(id).or_insert(Tensor::zeros(&shape));
        if add.dim() >= shape.len() {
            for t in add.keep_right(shape.len())?.inners().iter() {
                *grad = (&*grad + t)?;
            }
        } else {
            *grad = (&*grad + &add.view())?;
        }
        Ok(())
    }
    pub fn get(&self, id: TensorId) -> &Tensor<f32> {
        self.tensors.get(&id).expect("Tensor not found!")
    }
    pub fn get_grad(&self, id: TensorId) -> &Tensor<f32> {
        self.grads.get(&id).expect("Tensor not found!")
    }
    pub fn backward_all(
        &mut self,
        id: TensorId,
        loss_fn: Box<dyn Loss>,
    ) -> Result<f32, TensorError> {
        let output = self.get(id);
        let (loss, grad) = loss_fn.run(output)?;
        let mean_coeff = 1. / loss.size() as f32;
        self.add_grad(id, (&grad * &Tensor::scalar(mean_coeff))?)?;

        for (id, comp) in self.computations.clone().iter().rev() {
            for inp in comp.inps.iter() {
                let shape = self.get(*inp).shape().to_vec();
                self.grads
                    .entry(*inp)
                    .or_insert(Tensor::<f32>::zeros(&shape));
            }
            let inps = comp
                .inps
                .iter()
                .map(|id| &self.tensors[id])
                .collect::<Vec<_>>();
            let grad_out = &self.grads[&id];
            let grads = comp.func.grad(&inps, grad_out)?;
            for (id, grad) in comp.inps.clone().into_iter().zip(grads.into_iter()) {
                self.add_grad(id, grad)?;
            }
        }

        Ok(loss.mean())
    }
    pub fn forward(&mut self, training: bool) -> Result<(), TensorError> {
        for (out, c) in self.computations.iter_mut() {
            let tensors = c
                .inps
                .iter()
                .map(|id| self.tensors.get(id).expect("Tensor not found!"))
                .collect::<Vec<_>>();
            let result = c.func.run(&tensors, training)?;
            self.tensors.insert(*out, result);
        }
        Ok(())
    }
    pub fn call(
        &mut self,
        mut f: Box<dyn Function>,
        tensor_ids: &[TensorId],
    ) -> Result<TensorId, TensorError> {
        let tensors = tensor_ids
            .iter()
            .map(|id| self.tensors.get(id).expect("Tensor not found!"))
            .collect::<Vec<_>>();
        let out = f.run(&tensors, false)?;
        let child = self.alloc(out);
        self.computations.insert(
            child,
            Computation {
                func: f,
                inps: tensor_ids.to_vec(),
            },
        );
        Ok(child)
    }
    pub fn optimize<O: Optimizer>(
        &mut self,
        opt: &mut O,
        params: &HashSet<TensorId>,
        learning_rate: f32,
    ) -> Result<(), TensorError> {
        let (params, grads): (Vec<&mut Tensor<f32>>, Vec<&Tensor<f32>>) = self
            .tensors
            .iter_mut()
            .filter(|(id, _)| params.contains(id))
            .map(|(id, params)| {
                let grad = self.grads.get(id).expect("Tensor not found!");
                (params, grad)
            })
            .collect::<Vec<_>>()
            .into_iter()
            .unzip();
        opt.step(params, grads, learning_rate)?;
        Ok(())
    }
}
