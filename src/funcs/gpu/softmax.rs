use super::*;

pub fn gpu_run(out_id: TensorId, inps: &[Vec<usize>]) -> GpuFunction {
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

pub fn gpu_grad(out_id: TensorId, inps: &[Vec<usize>]) -> GpuFunctionGroup {
    let n = inps[0][inps[0].len() - 1];
    let works = inps[0][..inps[0].len() - 1].iter().fold(1, |a, b| a * b);

    let source_code = format!(
        "__kernel void grad_{out_id}(
                        __global float* out,
                        __global float* out_grad,
                        __global float* a,
                        __global float* a_grad) {{
        uint id = get_global_id(0);
        if(id < {works}) {{
            out += id * {n};
            out_grad += id * {n};
            for(uint i = 0; i < {n}; i++) {{
                float si = out[i];
                for(uint j = 0; j < {n}; j++) {{
                    if(i == j) {{
                        a_grad[i] += si * (1. - si) * out_grad[j];
                    }} else {{
                        float sj = out[j];
                        a_grad[i] = -si * sj * out_grad[j];
                    }}
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
