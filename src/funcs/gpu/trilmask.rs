use super::*;

pub fn gpu_impl(out_id: TensorId, inps: &[Vec<usize>], n: usize) -> GpuFunctionGroup {
    let works = inps[0][..inps[0].len() - 2].iter().fold(1, |a, b| a * b);

    let forward_source_code = format!(
        "__kernel void calc_{out_id}(
                        __global float* out,
                        __global float* a) {{
        uint id = get_global_id(0);
        out += {n} * {n} * id;
        a += {n} * {n} * id;
        if(id < {works}) {{
            for(uint i = 0; i < {n}; i++) {{
                for(uint j = 0; j < {n}; j++) {{
                    if(j <= i) {{
                        out[i * {n} + j] = a[i * {n} + j];
                    }} else {{
                        out[i * {n} + j] = -INFINITY;
                    }}
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
        out_grad += {n} * {n} * id;
        a_grad += {n} * {n} * id;
        if(id < {works}) {{
            for(uint i = 0; i < {n}; i++) {{
                for(uint j = 0; j < {n}; j++) {{
                    if(j <= i) {{
                        a_grad[i * {n} + j] += out_grad[i * {n} + j];
                    }}
                }}
            }}
        }}
    }}"
    );

    GpuFunctionGroup {
        shared_buffers: vec![],
        forward_funcs: vec![GpuFunction {
            source_code: forward_source_code,
            kernel_name: format!("calc_{}", out_id),
            local_work_size: 32,
            global_work_size: works,
        }],
        backward_funcs: vec![GpuFunction {
            source_code,
            kernel_name: format!("grad_{}", out_id),
            local_work_size: 32,
            global_work_size: works,
        }],
    }
}
