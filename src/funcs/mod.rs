#[cfg(feature = "gpu")]
use crate::graph::TensorId;

#[cfg(feature = "gpu")]
mod gpu;

#[cfg(feature = "gpu")]
pub use gpu::{GpuFunction, KernelCall, SharedBuffer};

mod add;
mod cat;
mod coeff;
mod crossentropy;
mod crossentropyfunc;
mod dropout;
mod embedding;
mod gelu;
mod layer_norm;
mod matmul;
mod relu;
mod softmax;
mod transpose;
mod trilmask;

pub use add::*;
pub use cat::*;
pub use coeff::*;
pub use crossentropy::*;
pub use crossentropyfunc::*;
pub use dropout::*;
pub use embedding::*;
pub use gelu::*;
pub use layer_norm::*;
pub use matmul::*;
pub use relu::*;
pub use softmax::*;
pub use transpose::*;
pub use trilmask::*;

use super::tensor::*;

pub trait Function: std::fmt::Debug {
    fn clone_box(&self) -> Box<dyn Function>;
    fn run(&mut self, inps: &[&GeneralTensor], training: bool) -> Result<Tensor<f32>, TensorError>;
    fn grad(
        &self,
        inps: &[&GeneralTensor],
        out_grad: &Tensor<f32>,
    ) -> Result<Vec<Tensor<f32>>, TensorError>;

    #[cfg(feature = "gpu")]
    fn gpu_impl(&self, _out_id: TensorId, _inp_shapes: &[Vec<usize>]) -> GpuFunction;
}

pub trait Loss: std::fmt::Debug {
    fn run(&self, inp: &GeneralTensor) -> Result<(Tensor<f32>, Tensor<f32>), TensorError>;
}
