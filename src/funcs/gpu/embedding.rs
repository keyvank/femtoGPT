use super::*;

pub fn gpu_run(out_id: TensorId, inps: &[Vec<usize>]) -> GpuFunction {
    let works = inps[0].iter().fold(1, |a, b| a * b);
    let degree = inps[1][1];

    let source_code = format!(
        "__kernel void calc_{out_id}(
                        __global float* out,
                        __global uint* inp,
                        __global float* emb) {{
        uint id = get_global_id(0);
        if(id < {works}) {{
            out += {degree} * id;
            emb += {degree} * inp[id];
            for(uint i = 0; i < {degree}; i++) {{
                out[i] = emb[i];
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
