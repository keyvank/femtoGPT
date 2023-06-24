use super::*;

pub fn gpu_grad(out_id: TensorId, inps: &[Vec<usize>]) -> GpuFunctionGroup {
    let works = inps[0].iter().fold(1, |a, b| a * b);
    let degree = inps[1][1];

    let forward_source_code = format!(
        "__kernel void calc_{out_id}(
                        __global float* out,
                        __global ulong* inp,
                        __global float* emb) {{
        uint id = get_global_id(0);
        out += {degree} * id;
        emb += {degree} * inp[id];
        if(id < {works}) {{
            for(uint i = 0; i < {degree}; i++) {{
                out[i] = emb[i];
            }}
        }}
    }}"
    );

    let source_code = format!(
        "__kernel void grad_{out_id}(
                        __global float* out,
                        __global float* out_grad,
                        __global ulong* inp,
                        __global float* inp_grad,
                        __global float* emb,
                        __global float* emb_grad) {{
        uint id = get_global_id(0);
        if(id < {degree}) {{
            for(uint i = 0; i < {works}; i++) {{
                emb_grad[{degree} * inp[i] + id] += out_grad[{degree} * i + id];
            }}
        }}
    }}"
    );

    let local_work_size = 32;
    let global_work_size =
        degree + ((local_work_size - (degree % local_work_size)) % local_work_size);

    GpuFunctionGroup {
        forward_funcs: vec![GpuFunction {
            source_code: forward_source_code,
            kernel_name: format!("calc_{}", out_id),
            local_work_size,
            global_work_size,
        }],
        funcs: vec![GpuFunction {
            source_code,
            kernel_name: format!("grad_{}", out_id),
            local_work_size,
            global_work_size,
        }],
        shared_buffers: vec![],
    }
}
