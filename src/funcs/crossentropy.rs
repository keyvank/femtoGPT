use super::Loss;
use crate::tensor::*;

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
    fn run(&self, inp: &Tensor<f32>) -> Result<(Tensor<f32>, Tensor<f32>), TensorError> {
        let grad_shape = inp.shape().to_vec();
        let mut loss_shape = grad_shape.clone();
        loss_shape.pop();
        let (loss, grad): (Vec<f32>, Vec<Vec<f32>>) = inp
            .keep_right(1)?
            .inners()
            .iter()
            .zip(self.target.blob().iter())
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

                (loss, grad)
            })
            .unzip();

        Ok((
            Tensor::raw(&loss_shape, loss),
            Tensor::raw(&grad_shape, grad.into_iter().flatten().collect()),
        ))
    }
}
