use super::Function;
use crate::tensor::*;

#[cfg(feature = "gpu")]
use super::{GpuFunction, TensorId};

#[derive(Debug, Clone)]
pub struct Softmax {
    out: Tensor<f32>,
}
impl Softmax {
    pub fn new() -> Box<dyn Function> {
        Box::new(Self {
            out: Tensor::scalar(0.),
        })
    }
}
impl Function for Softmax {
    fn run(&mut self, inps: &[&Tensor<f32>], _training: bool) -> Result<Tensor<f32>, TensorError> {
        self.out = inps[0].map(1, |l| {
            let max = l
                .blob()
                .iter()
                .fold(f32::NEG_INFINITY, |a, b| f32::max(a, *b));
            let sum = l.blob().iter().map(|f| (f - max).exp()).sum::<f32>();
            Ok(l.map_values(|f| (f - max).exp() / sum))
        })?;

        Ok(self.out.clone())
    }
    fn grad(
        &self,
        _inps: &[&Tensor<f32>],
        out_grad: &Tensor<f32>,
    ) -> Result<Vec<Tensor<f32>>, TensorError> {
        let grad_inp0 = self
            .out
            .keep_right(1)?
            .inners()
            .iter()
            .zip(out_grad.keep_right(1)?.inners().iter())
            .map(|(l, o)| {
                let l_blob = l.blob();
                let o_blob = o.blob();
                let n = l.shape()[0];
                let mut data = vec![0.; n];
                for i in 0..n {
                    let si = l_blob[i];
                    let mut sum = 0.;
                    for j in 0..n {
                        let sj = l_blob[j];
                        sum += (if i == j { si * (1. - si) } else { -si * sj }) * o_blob[j];
                    }
                    data[i] = sum;
                }
                data
            })
            .flatten()
            .collect::<Vec<_>>();
        Ok(vec![Tensor::raw(out_grad.shape(), grad_inp0)?])
    }
    fn clone_box(&self) -> Box<dyn Function> {
        Box::new(self.clone())
    }

    #[cfg(feature = "gpu")]
    fn gpu_source_code(&self, out_id: TensorId, inps: &[Vec<usize>]) -> GpuFunction {
        let n = inps[0][inps[0].len() - 1];
        let works = inps[0][..inps[0].len() - 1].iter().fold(1, |a, b| a * b);

        let source_code = format!(
            "__kernel void calc_{out_id}(
                            __global float* out,
                            __global float* a) {{
            uint id = get_global_id(0);
            if(id < {works}) {{
                a += id * {n};
                out += id * {n};
                float mx = a[0];
                for(uint i = 1; i < {n}; i++) {{
                    if(a[i] > mx) {{
                        mx = a[i];
                    }}
                }}
                float sum = 0.;
                for(uint i = 0; i < {n}; i++) {{
                    sum += exp(a[i] - mx);
                }}
                for(uint i = 0; i < {n}; i++) {{
                    out[i] = exp(a[i] - mx) / sum;
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
