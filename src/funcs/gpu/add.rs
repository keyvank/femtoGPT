use super::*;

pub fn gpu_impl(out_id: TensorId, inps: &[Vec<usize>]) -> GpuFunction {
    let a_size = inps[0].iter().fold(1, |a, b| a * b);
    let b_size = inps[1].iter().fold(1, |a, b| a * b);
    assert!(b_size <= a_size);
    let repeats = a_size / b_size;
    let works = std::cmp::max(a_size, b_size);

    let forward_source_code = format!(
        "__kernel void calc_{out_id}(
                        __global float* out,
                        __global float* a,
                        __global float* b) {{
        uint id = get_global_id(0);
        uint id_a = id % {a_size};
        uint id_b = id % {b_size};
        if(id < {works}) {{
            out[id] = a[id_a] + b[id_b];
        }}
    }}"
    );

    let backward_source_code_part_1 = format!(
        "__kernel void grad_{out_id}_1(
                        __global float* out,
                        __global float* out_grad,
                        __global float* a,
                        __global float* a_grad,
                        __global float* b,
                        __global float* b_grad) {{
        uint id = get_global_id(0);
        if(id < {works}) {{
            a_grad[id] += out_grad[id];
        }}
    }}"
    );

    let backward_source_code_part_2 = format!(
        "__kernel void grad_{out_id}_2(
                        __global float* out,
                        __global float* out_grad,
                        __global float* a,
                        __global float* a_grad,
                        __global float* b,
                        __global float* b_grad) {{
        uint id = get_global_id(0);
        if(id < {b_size}) {{
            for(uint i = 0; i < {repeats}; i++) {{
                b_grad[id] += out_grad[i * {b_size} + id];
            }}
        }}
    }}"
    );

    GpuFunction {
        shared_buffers: vec![],
        forward_funcs: vec![KernelCall {
            source_code: forward_source_code,
            kernel_name: format!("calc_{}", out_id),
            local_work_size: 32,
            global_work_size: works,
        }],
        backward_funcs: vec![
            KernelCall {
                source_code: backward_source_code_part_1,
                kernel_name: format!("grad_{}_1", out_id),
                local_work_size: 32,
                global_work_size: works,
            },
            KernelCall {
                source_code: backward_source_code_part_2,
                kernel_name: format!("grad_{}_2", out_id),
                local_work_size: 32,
                global_work_size: b_size,
            },
        ],
    }
}
