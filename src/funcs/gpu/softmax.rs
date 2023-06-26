use super::*;

pub fn gpu_impl(out_id: TensorId, inps: &[Vec<usize>]) -> GpuFunction {
    let n = inps[0][inps[0].len() - 1];
    let works = inps[0][..inps[0].len() - 1].iter().fold(1, |a, b| a * b);

    let forward_source_code = format!(
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

    let backward_source_code = format!(
        "__kernel void grad_{out_id}(
                        __global float* out,
                        __global float* out_grad,
                        __global float* a,
                        __global float* a_grad) {{
        uint wid = get_global_id(0);
        uint id = wid / {n};
        uint i = wid % {n};
        if(wid < {works} * {n}) {{
            out += id * {n};
            out_grad += id * {n};
            a_grad += id * {n};
            float si = out[i];
            float sum = 0.0;
            for(uint j = 0; j < {n}; j++) {{
                if(i == j) {{
                    sum += si * (1. - si) * out_grad[j];
                }} else {{
                    float sj = out[j];
                    sum += -si * sj * out_grad[j];
                }}
            }}
            a_grad[i] += sum;
        }}
    }}"
    );

    GpuFunction {
        forward_funcs: vec![KernelCall {
            source_code: forward_source_code,
            kernel_name: format!("calc_{}", out_id),
            local_work_size: 32,
            global_work_size: works,
        }],
        backward_funcs: vec![KernelCall {
            source_code: backward_source_code,
            kernel_name: format!("grad_{}", out_id),
            local_work_size: 32,
            global_work_size: works * n,
        }],
        shared_buffers: vec![],
    }
}
