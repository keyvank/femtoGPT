use super::*;

pub fn gpu_impl(out_id: TensorId, inps: &[Vec<usize>]) -> GpuFunction {
    let works = inps[0].iter().fold(1, |a, b| a * b);

    let forward_source_code = format!(
        "__kernel void calc_{out_id}(
                        __global float* out,
                        __global float* a) {{
        uint id = get_global_id(0);
        if(id < {works}) {{
            float val = a[id];
            out[id] = val > 0. ? val : val * 0.01;
        }}
    }}"
    );

    let backward_source_code = format!(
        "__kernel void grad_{out_id}(
                        __global float* out,
                        __global float* out_grad,
                        __global float* a,
                        __global float* a_grad) {{
        uint id = get_global_id(0);
        if(id < {works}) {{
            float val = a[id];
            a_grad[id] += val > 0. ? out_grad[id] : out_grad[id] * 0.01;
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
            global_work_size: works,
        }],
        shared_buffers: vec![],
    }
}
