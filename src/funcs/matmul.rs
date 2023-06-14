use super::Function;
use crate::tensor::*;

#[cfg(feature = "gpu")]
use super::{GpuFunction, TensorId};

#[derive(Debug, Clone)]
pub struct MatMul;
impl MatMul {
    pub fn new() -> Box<dyn Function> {
        Box::new(Self {})
    }
}
impl Function for MatMul {
    fn run(&mut self, inps: &[&Tensor<f32>], _training: bool) -> Result<Tensor<f32>, TensorError> {
        inps[0] ^ inps[1]
    }
    fn grad(
        &self,
        inps: &[&Tensor<f32>],
        out_grad: &Tensor<f32>,
    ) -> Result<Vec<Tensor<f32>>, TensorError> {
        Ok(vec![
            (out_grad ^ &inps[1].transpose()?)?,
            (&inps[0].transpose()? ^ out_grad)?,
        ])
    }
    fn clone_box(&self) -> Box<dyn Function> {
        Box::new(self.clone())
    }

    #[cfg(feature = "gpu")]
    fn gpu_source_code(&self, out_id: TensorId, inps: &[Vec<usize>]) -> GpuFunction {
        let inp0_size = inps[0][..inps[0].len() - 2].iter().fold(1, |a, b| a * b);
        let inp1_size = inps[1][..inps[1].len() - 2].iter().fold(1, |a, b| a * b);
        assert_eq!(inps[0][inps[0].len() - 1], inps[1][inps[1].len() - 2]);
        let m = inps[0][inps[0].len() - 2];
        let n = inps[0][inps[0].len() - 1];
        let p = inps[1][inps[1].len() - 1];
        let mp = m * p;
        let works = std::cmp::max(inp0_size, inp1_size);
        println!("{}", works);
        let source_code = format!(
            "__kernel void calc_{out_id}(
                            __global float* out,
                            __global float* a,
                            __global float* b) {{
            uint id = get_global_id(0);
            uint id_a = id % {inp0_size};
            uint id_b = id % {inp1_size};
            if(id < {works}) {{
                out += {mp} * id;
                for(uint i = 0; i < {mp}; i++) out[i] = 0.;
                for(uint i = 0; i < {m}; i++) {{
                    for(uint k = 0; k < {n}; k++) {{
                        for(uint j = 0; j < {p}; j++) {{
                            out[i * {p} + j] += a[i * {n} + k] * b[{p} * k + j];
                        }}
                    }}
                }}
            }}
        }}"
        );

        let local_work_size = 32;
        let global_work_size =
            works + ((local_work_size - (works % local_work_size)) % local_work_size);

        GpuFunction {
            source_code,
            kernel_name: format!("calc_{}", out_id),
            local_work_size,
            global_work_size,
        }
    }
}
