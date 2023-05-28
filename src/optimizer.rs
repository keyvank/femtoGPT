use crate::tensor::Tensor;

pub trait Optimizer {
    fn step(&self, params: Vec<&mut Tensor<f32>>, grads: Vec<&Tensor<f32>>);
}

pub struct NaiveOptimizer {
    learning_rate: f32,
}

impl NaiveOptimizer {
    pub fn new(learning_rate: f32) -> Box<dyn Optimizer> {
        Box::new(Self { learning_rate })
    }
}

impl Optimizer for NaiveOptimizer {
    fn step(&self, params: Vec<&mut Tensor<f32>>, grads: Vec<&Tensor<f32>>) {
        for (param, grad) in params.into_iter().zip(grads.into_iter()) {
            *param = &*param + &(grad * &Tensor::scalar(-self.learning_rate));
        }
    }
}
