use super::*;

pub fn gpu_run(out_id: TensorId, inps: &[Vec<usize>]) -> GpuFunction {
    let works = inps[0][..inps[0].len() - 2].iter().fold(1, |a, b| a * b);
    let m = inps[0][inps[0].len() - 2];
    let n = inps[0][inps[0].len() - 1];
    let source_code = format!(
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

pub fn gpu_grad(out_id: TensorId, inps: &[Vec<usize>]) -> GpuFunctionGroup {
    let works = inps[0][..inps[0].len() - 2].iter().fold(1, |a, b| a * b);
    let m = inps[0][inps[0].len() - 2];
    let n = inps[0][inps[0].len() - 1];

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

    let local_work_size = 32;
    let global_work_size =
        works + ((local_work_size - (works % local_work_size)) % local_work_size);

    GpuFunctionGroup {
        funcs: vec![GpuFunction {
            source_code,
            kernel_name: format!("grad_{}", out_id),
            local_work_size,
            global_work_size,
        }],
        shared_buffers: vec![],
    }
}
