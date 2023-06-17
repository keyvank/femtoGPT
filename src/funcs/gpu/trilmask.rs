use super::*;

pub fn gpu_run(out_id: TensorId, inps: &[Vec<usize>], n: usize, value: f32) -> GpuFunction {
    let works = inps[0][..inps[0].len() - 2].iter().fold(1, |a, b| a * b);

    let source_code = format!(
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
                        out[i * {n} + j] = {value};
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

pub fn gpu_grad(out_id: TensorId, inps: &[Vec<usize>], n: usize) -> GpuFunctionGroup {
    let works = inps[0][..inps[0].len() - 2].iter().fold(1, |a, b| a * b);
    let source_code = format!(
        "__kernel void grad_{out_id}(
                        __global float* out,
                        __global float* out_grad,
                        __global float* a,
                        __global float* a_grad) {{
        out_grad += {n} * {n} * id;
        a_grad += {n} * {n} * id;
        if(id < {works}) {{
            for(uint i = 0; i < {n}; i++) {{
                for(uint j = 0; j < {n}; j++) {{
                    if(j <= i) {{
                        a_grad[i * {n} + j] += out_grad[i * {n} + j];
                    }} else {{
                        a_grad[i * {n} + j] = 0.;
                    }}
                }}
            }}
        }}
    }}"
    );

    GpuFunctionGroup {
        shared_buffers: vec![],
        funcs: vec![GpuFunction {
            source_code,
            kernel_name: format!("grad_{}", out_id),
            local_work_size: 32,
            global_work_size: works + ((32 - (works % 32)) % 32),
        }],
    }
}
