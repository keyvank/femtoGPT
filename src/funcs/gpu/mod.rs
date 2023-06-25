pub mod add;
pub mod cat;
pub mod coeff;
pub mod dropout;
pub mod embedding;
pub mod layer_norm;
pub mod matmul;
pub mod relu;
pub mod softmax;
pub mod transpose;
pub mod trilmask;

use crate::graph::TensorId;

#[derive(Clone, Debug)]
pub enum SharedBuffer {
    Float(usize),
    Usize(usize),
}

#[derive(Clone, Debug)]
pub struct GpuFunction {
    pub shared_buffers: Vec<SharedBuffer>,
    pub forward_funcs: Vec<KernelCall>,
    pub backward_funcs: Vec<KernelCall>,
}

#[derive(Clone, Debug)]
pub struct KernelCall {
    pub source_code: String,
    pub kernel_name: String,
    pub global_work_size: usize,
    pub local_work_size: usize,
}
