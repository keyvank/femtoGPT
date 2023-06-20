use super::*;

pub fn gpu_run(out_id: TensorId, inps: &[Vec<usize>]) -> GpuFunction {
    let works = inps[0].iter().fold(1, |a, b| a * b);
    let source_code = format!(
        "__kernel void calc_{out_id}(
                        __global float* out,
                        __global float* a) {{
        uint id = get_global_id(0);
        if(id < {works}) {{
            out[id] = a[id];
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
    let works = inps[0].iter().fold(1, |a, b| a * b);

    let source_code = format!(
        "__kernel void grad_{out_id}(
                        __global float* out,
                        __global float* out_grad,
                        __global float* a,
                        __global float* a_grad) {{
        uint id = get_global_id(0);
        if(id < {works}) {{
            a_grad[id] += out_grad[id];
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
