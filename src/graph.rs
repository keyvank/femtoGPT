use crate::funcs::{Function, Loss};
use crate::optimizer::Optimizer;
use crate::tensor::*;
use rand::Rng;
use std::collections::{HashMap, HashSet};

pub type TensorId = usize;

struct Shape {
    is_batched: bool,
    shape: Vec<usize>,
}

impl Shape {
    fn matches(&self, shape: &[usize]) -> bool {
        if self.is_batched {
            shape.len() == self.shape.len() + 1 && shape[1..] == self.shape
        } else {
            shape == self.shape
        }
    }
}

struct Computation {
    inps: Vec<TensorId>,
    out: TensorId,
    func: Box<dyn Function>,
}

pub struct Graph {
    tensors: HashMap<TensorId, Tensor<f32>>,
    grads: HashMap<TensorId, Tensor<f32>>,
    computations: Vec<Computation>,
    shapes: HashMap<TensorId, Shape>,
}

impl Graph {
    pub fn new() -> Self {
        Self {
            tensors: Default::default(),
            grads: Default::default(),
            computations: Default::default(),
            shapes: Default::default(),
        }
    }
    pub fn alloc_param<R: Rng>(&mut self, rng: &mut R, shape: &[usize]) -> TensorId {
        let id = self.alloc(Tensor::<f32>::rand(rng, shape));
        self.shapes.insert(
            id,
            Shape {
                is_batched: false,
                shape: shape.to_vec(),
            },
        );
        id
    }
    pub fn alloc_input(&mut self, shape: &[usize]) -> TensorId {
        let mut final_shape = shape.to_vec();
        final_shape.insert(0, 1);
        let id = self.alloc(Tensor::<f32>::rand(&mut rand::thread_rng(), &final_shape));
        self.shapes.insert(
            id,
            Shape {
                is_batched: true,
                shape: shape.to_vec(),
            },
        );
        id
    }
    fn alloc(&mut self, t: Tensor<f32>) -> TensorId {
        let id = self.tensors.len();
        self.tensors.insert(id, t);
        id
    }
    pub fn load<T: TensorOps<f32>>(&mut self, tensor_id: TensorId, tensor: &T) {
        if let Some(shape) = self.shapes.get(&tensor_id) {
            assert!(shape.matches(tensor.shape()));
        }
        self.tensors.insert(tensor_id, tensor.view().into());
    }
    pub fn zero_grad(&mut self) {
        self.grads.clear();
    }
    pub fn add_grad<T: TensorOps<f32>>(&mut self, id: TensorId, add: T) {
        let mut shape = self.get(id).shape().to_vec();
        let grad = self.grads.entry(id).or_insert(Tensor::zeros(&shape));
        if add.dim() >= shape.len() {
            shape.insert(0, 0);
            for t in add.reshape(&shape).inners().iter() {
                *grad = &*grad + t;
            }
        } else {
            *grad = &*grad + &add.view();
        }
    }
    pub fn get(&self, id: TensorId) -> &Tensor<f32> {
        self.tensors.get(&id).expect("Tensor not found!")
    }
    pub fn backward(&mut self, id: TensorId) {
        if let Some(comp) = self.computations.iter().find(|c| c.out == id) {
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
            let result_out = &self.tensors[&id];
            let grad_out = &self.grads[&id];
            let grads = comp.func.grad(&inps, result_out, grad_out);
            for (id, grad) in comp.inps.clone().into_iter().zip(grads.into_iter()) {
                self.add_grad(id, grad);
            }
        }
    }
    pub fn backward_all(&mut self, id: TensorId, loss_fn: Box<dyn Loss>) -> Tensor<f32> {
        let output = self.get(id);
        let (loss, grad) = loss_fn.run(output);
        self.add_grad(id, grad);

        let backward_order = self
            .computations
            .iter()
            .map(|c| c.out)
            .rev()
            .collect::<Vec<_>>();
        for id in backward_order {
            self.backward(id);
        }

        loss
    }
    pub fn gradient_check(&mut self, id: TensorId) {
        const EPSILON: f32 = 1e-5;
        for comp in self.computations.iter_mut() {
            for i in 0..comp.inps.len() {
                let mut inps = comp
                    .inps
                    .iter()
                    .map(|id| {
                        Tensor::<f32>::rand_range(
                            &mut rand::thread_rng(),
                            -0.1,
                            0.1,
                            self.tensors[id].shape(),
                        )
                    })
                    .collect::<Vec<_>>();
                let inp_val = inps[i].clone();

                inps[i] = (&inp_val + &Tensor::scalar(EPSILON));
                let out_plus = comp.func.run(&inps.iter().collect::<Vec<_>>(), false);

                inps[i] = (&inp_val - &Tensor::scalar(EPSILON));
                let out_minus = comp.func.run(&inps.iter().collect::<Vec<_>>(), false);

                let numeric = &(&out_plus - &out_minus) * &Tensor::scalar(0.5 / EPSILON);
                inps[i] = inp_val.clone();
                let out = comp.func.run(&inps.iter().collect::<Vec<_>>(), false);
                let symbolic = comp.func.grad(
                    &inps.iter().collect::<Vec<_>>(),
                    &out,
                    &Tensor::ones(out.shape()),
                )[i]
                    .clone();

                let num_sum = numeric.blob().iter().sum::<f32>();
                let sym_sum = symbolic.blob().iter().sum::<f32>();
                let diff = num_sum / sym_sum;
            }
        }
    }
    pub fn forward(&mut self, training: bool) {
        for c in self.computations.iter_mut() {
            let tensors = c
                .inps
                .iter()
                .map(|id| self.tensors.get(id).expect("Tensor not found!"))
                .collect::<Vec<_>>();
            let result = c.func.run(&tensors, training);
            self.tensors.insert(c.out, result);
        }
    }
    pub fn call(&mut self, mut f: Box<dyn Function>, tensor_ids: &[TensorId]) -> TensorId {
        let tensors = tensor_ids
            .iter()
            .map(|id| self.tensors.get(id).expect("Tensor not found!"))
            .collect::<Vec<_>>();
        let out = f.run(&tensors, false);
        let child = self.alloc(out);
        self.computations.push(Computation {
            func: f,
            out: child,
            inps: tensor_ids.to_vec(),
        });
        child
    }
    pub fn optimize(&mut self, opt: &mut Box<dyn Optimizer>, params: &HashSet<TensorId>) {
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
        opt.step(params, grads);
    }
}
