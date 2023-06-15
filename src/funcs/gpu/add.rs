use super::*;

pub fn gpu_run(out_id: TensorId, inps: &[Vec<usize>]) -> GpuFunction {
    let inp0_size = inps[0].iter().fold(1, |a, b| a * b);
    let inp1_size = inps[1].iter().fold(1, |a, b| a * b);
    let works = std::cmp::max(inp0_size, inp1_size);
    let source_code = format!(
        "__kernel void calc_{out_id}(
                        __global float* out,
                        __global float* a,
                        __global float* b) {{
        uint id = get_global_id(0);
        uint id_a = id % {inp0_size};
        uint id_b = id % {inp1_size};
        if(id < {works}) {{
            out[id] = a[id_a] + b[id_b];
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
