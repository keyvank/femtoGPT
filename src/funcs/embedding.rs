use super::Function;
use crate::tensor::*;

#[cfg(feature = "gpu")]
use super::{gpu, GpuFunction, TensorId};

/// A simple **embedding layer** that maps integer indices to float vectors.
///
/// The first input tensor contains indices (`usize`) and the second contains
/// the embedding matrix (`f32`).
#[derive(Debug, Clone)]
pub struct Embedding;
impl Embedding {
    /// Creates a new `Embedding` layer wrapped in a `Box<dyn Function>`.
    ///
    /// # Example
    /// ```example
    /// let embed = Embedding::new();
    /// ```
    pub fn new() -> Box<dyn Function> {
        Box::new(Self {})
    }
}

impl Function for Embedding {
    /// Performs a forward pass of the embedding layer.
    ///
    /// # Arguments
    /// * `inps` - A slice of references to `GeneralTensor`s. Expects:
    ///     1. A tensor of indices (`usize`)
    ///     2. A tensor of embedding vectors (`f32`) with shape `[vocab_size, embedding_dim]`
    /// * `_training` - Indicates if the function is in training mode (not used here)
    ///
    /// # Returns
    /// A `Tensor<f32>` containing the embedded vectors for the given indices.
    ///
    /// # Errors
    /// Returns `TensorError::UnexpectedShape` if the embedding matrix is not 2D.
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

    /// Computes gradients with respect to the input and embedding matrix.
    ///
    /// The gradient with respect to indices is a scalar zero (indices don't need gradients
    /// as they are discrete). The gradient with respect to the embedding matrix is computed
    /// by accumulating the output gradients at rows corresponding to the input indices.
    ///
    /// # Arguments
    /// * `inps` - A slice of references to the input tensors (indices and embedding matrix)
    /// * `out_grad` - The gradient tensor with respect to the output
    ///
    /// # Returns
    /// A `Vec<Tensor<f32>>` containing:
    ///     1. Gradient w.r.t. indices (always zero scalar)
    ///     2. Gradient w.r.t. embedding matrix (updated rows only)
    ///
    /// # Errors
    /// Returns `TensorError` if tensor operations fail during gradient computation.
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

    /// Creates a boxed clone of this embedding layer.
    ///
    /// # Returns
    /// A new `Box<dyn Function>` containing a clone of this `Embedding`.
    fn clone_box(&self) -> Box<dyn Function> {
        Box::new(self.clone())
    }

    #[cfg(feature = "gpu")]
    fn gpu_impl(&self, out_id: TensorId, inps: &[Vec<usize>]) -> GpuFunction {
        gpu::embedding::gpu_impl(out_id, inps)
    }
}
