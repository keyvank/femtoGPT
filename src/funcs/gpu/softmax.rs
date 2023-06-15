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
