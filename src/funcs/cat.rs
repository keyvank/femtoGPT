use super::Function;
use crate::tensor::*;

#[derive(Debug, Clone)]
pub struct Cat {}
impl Cat {
    pub fn new() -> Box<dyn Function> {
        Box::new(Self {})
    }
}

impl Function for Cat {
    fn run(&mut self, inps: &[&Tensor<f32>], _training: bool) -> Result<Tensor<f32>, TensorError> {
        let first_input = inps.get(0).ok_or(TensorError::UnexpectedShape)?; // TODO: Better error?
        let shape = first_input.shape().to_vec();
        if !inps.iter().all(|t| t.shape() == shape) {
            return Err(TensorError::UnexpectedShape);
        }
        let each_sz = first_input.size();
        let group_size = shape.last().ok_or(TensorError::UnexpectedShape)?;
        let mut offset = 0;
        let mut data = Vec::with_capacity(each_sz * inps.len());

        while offset < each_sz {
            for inp in inps {
                data.extend(&inp.blob()[offset..offset + group_size]);
            }
            offset += group_size;
        }

        let mut target_shape = shape.clone();
        target_shape[shape.len() - 1] = target_shape[shape.len() - 1] * inps.len();
        Tensor::raw(&target_shape, data)
    }
    fn grad(
        &self,
        inps: &[&Tensor<f32>],
        out_grad: &Tensor<f32>,
    ) -> Result<Vec<Tensor<f32>>, TensorError> {
        let cnt = inps.len();
        let last_dim = *out_grad
            .shape()
            .last()
            .ok_or(TensorError::UnexpectedShape)?;
        let group_size = last_dim / cnt;
        if group_size * cnt != last_dim {
            return Err(TensorError::UnexpectedShape);
        }
        let mut result = vec![Vec::new(); cnt];
        let mut offset = 0;
        let out_grad_blob = out_grad.blob();
        while offset < out_grad.size() {
            for i in 0..cnt {
                result[i].extend(&out_grad_blob[offset..offset + group_size]);
                offset += group_size;
            }
        }
        let mut target_shape = out_grad.shape().to_vec();
        target_shape[out_grad.dim() - 1] = target_shape[out_grad.dim() - 1] / cnt;
        result
            .into_iter()
            .map(|d| Tensor::raw(&target_shape, d))
            .collect()
    }
    fn clone_box(&self) -> Box<dyn Function> {
        Box::new(self.clone())
    }
}
