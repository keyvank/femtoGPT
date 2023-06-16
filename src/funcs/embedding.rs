use super::Function;
use crate::tensor::*;

#[derive(Debug, Clone)]
pub struct Embedding;
impl Embedding {
    pub fn new() -> Box<dyn Function> {
        Box::new(Self {})
    }
}
impl Function for Embedding {
    fn run(
        &mut self,
        inps: &[&GeneralTensor],
        _training: bool,
    ) -> Result<Tensor<f32>, TensorError> {
        let inp = inps[0].as_usize()?;
        let emb = inps[1].as_float()?;
        if emb.dim() != 2 {
            return Err(TensorError::UnexpectedShape);
        }
        inp.map(0, |s| Ok(emb.get(s.scalar()?)?.into()))
    }
    fn grad(
        &self,
        _inps: &[&GeneralTensor],
        _out_grad: &Tensor<f32>,
    ) -> Result<Vec<Tensor<f32>>, TensorError> {
        Ok(vec![Tensor::scalar(0.), Tensor::scalar(0.)])
    }
    fn clone_box(&self) -> Box<dyn Function> {
        Box::new(self.clone())
    }
}
