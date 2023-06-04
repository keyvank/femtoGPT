use super::{Loss, Tensor, TensorOps};
use rayon::prelude::*;

#[derive(Debug)]
pub struct CrossEntropy {
    classes: usize,
    target: Tensor<usize>,
}
impl CrossEntropy {
    pub fn new(classes: usize, target: Tensor<usize>) -> Box<dyn Loss> {
        Box::new(Self { classes, target })
    }
}

impl Loss for CrossEntropy {
    fn run(&self, inp: &Tensor<f32>) -> (Tensor<f32>, Tensor<f32>) {
        let grad_shape = inp.shape().to_vec();
        let mut loss_shape = grad_shape.clone();
        loss_shape.pop();
        let (loss, grad): (Vec<Tensor<f32>>, Vec<Tensor<f32>>) = inp
            .keep_right(1)
            .inners()
            .par_iter()
            .zip(self.target.blob().par_iter())
            .map(|(o, t)| {
                let o_exps = o.blob().iter().map(|f| f.exp()).collect::<Vec<_>>();
                let sum = o_exps.iter().sum::<f32>();
                let loss = sum.ln() - o.blob()[*t];
                let sum_inv = 1. / sum;

                let grad = (0..self.classes)
                    .map(|c| {
                        let val = o_exps[c];
                        if *t == c {
                            val * sum_inv - 1.0
                        } else {
                            val * sum_inv
                        }
                    })
                    .collect::<Vec<_>>();

                (Tensor::scalar(loss), Tensor::vector(&grad))
            })
            .unzip();

        (
            Tensor::stack(&loss).reshape(&loss_shape).into(),
            Tensor::stack(&grad).reshape(&grad_shape).into(),
        )
    }
}
