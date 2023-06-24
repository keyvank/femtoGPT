use super::Function;
use crate::tensor::*;

#[cfg(feature = "gpu")]
use super::{gpu, GpuFunctionGroup, TensorId};

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
        inps: &[&GeneralTensor],
        out_grad: &Tensor<f32>,
    ) -> Result<Vec<Tensor<f32>>, TensorError> {
        let inp = inps[0].as_usize()?;
        let mut grad = Tensor::<f32>::zeros(inps[1].as_float()?.shape());
        for (ch, embed) in inp
            .blob()
            .iter()
            .zip(out_grad.keep_right(1)?.inners().iter())
        {
            let mut row = grad.get_mut(*ch)?;
            row.set((&row.view() + embed)?)?;
        }
        Ok(vec![Tensor::scalar(0.), grad])
    }
    fn clone_box(&self) -> Box<dyn Function> {
        Box::new(self.clone())
    }

    #[cfg(feature = "gpu")]
    fn gpu_impl(&self, out_id: TensorId, inps: &[Vec<usize>]) -> GpuFunctionGroup {
        gpu::embedding::gpu_impl(out_id, inps)
    }
}
