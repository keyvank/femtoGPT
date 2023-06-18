pub mod add;
pub mod cat;
pub mod coeff;
pub mod embedding;
pub mod layer_norm;
pub mod matmul;
pub mod relu;
pub mod softmax;
pub mod trilmask;

use crate::graph::TensorId;

#[derive(Clone, Debug)]
pub struct GpuFunctionGroup {
    pub shared_buffers: Vec<usize>,
    pub funcs: Vec<GpuFunction>,
}

#[derive(Clone, Debug)]
pub struct GpuFunction {
    pub source_code: String,
    pub kernel_name: String,
    pub global_work_size: usize,
    pub local_work_size: usize,
}
