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
                let sum = o.blob().iter().map(|f| f.exp()).sum::<f32>();
                let loss = sum.ln() - o.get(*t).scalar();

                let grad = (0..self.classes)
                    .map(|c| {
                        let val = o.get(c).scalar().exp();
                        if *t == c {
                            val / sum - 1.0
                        } else {
                            val / sum
                        }
                    })
                    .collect::<Vec<_>>();

                (Tensor::scalar(loss), Tensor::vector(&grad))
            })
            .collect::<Vec<_>>()
            .into_iter()
            .unzip();

        (
            Tensor::stack(&loss).reshape(&loss_shape).into(),
            Tensor::stack(&grad).reshape(&grad_shape).into(),
        )
    }
}
