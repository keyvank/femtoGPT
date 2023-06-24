use super::*;

pub fn gpu_impl(out_id: TensorId, inps: &[Vec<usize>]) -> GpuFunction {
    let works = inps[0][..inps[0].len() - 2].iter().fold(1, |a, b| a * b);
    let m = inps[0][inps[0].len() - 2];
    let n = inps[0][inps[0].len() - 1];

    let forward_source_code = format!(
        "__kernel void calc_{out_id}(
                        __global float* out,
                        __global float* a) {{
        uint id = get_global_id(0);
        out += id * {m} * {n};
        a += id * {m} * {n};
        if(id < {works}) {{
            for(uint i = 0; i < {m}; i++) {{
                for(uint j = 0; j < {n}; j++) {{
                    out[j * {m} + i] = a[i * {n} + j];
                }}
            }}
        }}
    }}"
    );

    let source_code = format!(
        "__kernel void grad_{out_id}(
                        __global float* out,
                        __global float* out_grad,
                        __global float* a,
                        __global float* a_grad) {{
        uint id = get_global_id(0);
        out_grad += id * {m} * {n};
        a_grad += id * {m} * {n};
        if(id < {works}) {{
            for(uint i = 0; i < {m}; i++) {{
                for(uint j = 0; j < {n}; j++) {{
                    a_grad[i * {n} + j] += out_grad[j * {m} + i];
                }}
            }}
        }}
    }}"
    );

    GpuFunction {
        shared_buffers: vec![],
        forward_funcs: vec![KernelCall {
            source_code: forward_source_code,
            kernel_name: format!("calc_{}", out_id),
            local_work_size: 32,
            global_work_size: works,
        }],
        backward_funcs: vec![KernelCall {
            source_code,
            kernel_name: format!("grad_{}", out_id),
            local_work_size: 32,
            global_work_size: works,
        }],
    }
}
