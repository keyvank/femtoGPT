use super::*;

pub fn gpu_impl(out_id: TensorId, inps: &[Vec<usize>], rate: f32) -> GpuFunction {
    let works = inps[0].iter().fold(1, |a, b| a * b);
    const M: usize = 2147483647;
    let threshold = (rate as f64) * (M as f64);
    let gain = 1.0 / (1.0 - rate);

    let forward_source_code = format!(
        "__kernel void calc_{out_id}(
                        __global float* out,
                        __global ulong* seeds,
                        __global float* a) {{
        uint id = get_global_id(0);
        ulong A = 16807;
        ulong M = {M};

        if(id < {works}) {{
            if(seeds[id] == 0) {{
                seeds[id] = 1;
            }}
            seeds[id] = (seeds[id] * A) % M;

            if(seeds[id] < {threshold}) {{
                out[id] = 0.0;
            }} else {{
                out[id] = a[id];
            }}
        }}
    }}"
    );

    let backward_source_code = format!(
        "__kernel void grad_{out_id}(
                        __global float* out,
                        __global float* out_grad,
                        __global ulong* seeds,
                        __global float* a,
                        __global float* a_grad) {{
        uint id = get_global_id(0);
        if(id < {works}) {{
            bool dropped = seeds[id] < {threshold};
            if(!dropped) {{
                a_grad[id] += out_grad[id] * {gain};
            }}
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
        shared_buffers: vec![SharedBuffer::Usize(works)],
    }
}
