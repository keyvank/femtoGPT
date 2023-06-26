use super::*;

pub fn gpu_impl(out_id: TensorId, inps: &[Vec<usize>]) -> GpuFunction {
    let a_mats = inps[0][..inps[0].len() - 2].iter().fold(1, |a, b| a * b);
    let b_mats = inps[1][..inps[1].len() - 2].iter().fold(1, |a, b| a * b);
    assert!(b_mats <= a_mats);
    assert_eq!(inps[0][inps[0].len() - 1], inps[1][inps[1].len() - 2]);
    let m = inps[0][inps[0].len() - 2];
    let n = inps[0][inps[0].len() - 1];
    let p = inps[1][inps[1].len() - 1];
    let mp = m * p;
    let mn = m * n;
    let np = n * p;
    let mats = std::cmp::max(a_mats, b_mats);

    let works_forward = mats * m * p;
    let forward_source_code = format!(
        "__kernel void calc_{out_id}(
                        __global float* out,
                        __global float* grad_buff,
                        __global float* a,
                        __global float* b) {{
        uint wid = get_global_id(0);
        uint id = wid / {mp};
        uint id_a = id % {a_mats};
        uint id_b = id % {b_mats};
        uint ij = wid % {mp};
        uint i = ij / {p};
        uint j = ij % {p};
        if(wid < {works_forward}) {{
            out += {m} * {p} * id;
            a += {m} * {n} * id_a;
            b += {n} * {p} * id_b;
            float sum = 0.0;
            for(uint k = 0; k < {n}; k++) {{
                sum += a[i * {n} + k] * b[{p} * k + j];
            }}
            out[ij] = sum;
        }}
    }}"
    );

    let works_1 = mats * mn;
    let backward_source_code_part_1 = format!(
        "__kernel void grad_{out_id}_1(
                        __global float* out,
                        __global float* out_grad,
                        __global float* grad_buff,
                        __global float* a,
                        __global float* a_grad,
                        __global float* b,
                        __global float* b_grad) {{
        uint wid = get_global_id(0);
        uint id = wid / {mn};
        uint id_a = id % {a_mats};
        uint id_b = id % {b_mats};
        uint ik = wid % {mn};
        uint i = ik / {n};
        uint k = ik % {n};
        if(wid < {works_1}) {{
            out_grad += {mp} * id;
            a_grad += {mn} * id_a;
            b += {np} * id_b;
            float sum = 0.0;
            for(uint j = 0; j < {p}; j++) {{
                sum += out_grad[i * {p} + j] * b[k * {p} + j];
            }}
            a_grad[ik] += sum;
        }}
    }}"
    );

    let works_2 = mats * np;
    let backward_source_code_part_2 = format!(
        "__kernel void grad_{out_id}_2(
                        __global float* out,
                        __global float* out_grad,
                        __global float* grad_buff,
                        __global float* a,
                        __global float* a_grad,
                        __global float* b,
                        __global float* b_grad) {{
        uint wid = get_global_id(0);
        uint id = wid / {np};
        uint id_a = id % {a_mats};
        uint kj = wid % {np};
        uint k = kj / {p};
        uint j = kj % {p};

        if(id < {works_2}) {{
            out_grad += {mp} * id;
            a += {mn} * id_a;
            grad_buff += {np} * id;
            grad_buff[kj] = 0.0;
            float sum = 0.0;
            for(uint i = 0; i < {m}; i++) {{
                sum += a[i * {n} + k] * out_grad[i * {p} + j];
            }}
            grad_buff[kj] += sum;
        }}
    }}"
    );

    let inp1_mats = mats / b_mats;
    let inp1_total = b_mats * np;
    let backward_source_code_part_3 = format!(
        "__kernel void grad_{out_id}_3(
                        __global float* out,
                        __global float* out_grad,
                        __global float* grad_buff,
                        __global float* a,
                        __global float* a_grad,
                        __global float* b,
                        __global float* b_grad) {{
        uint id = get_global_id(0);
        if(id < {inp1_total}) {{
            float sum = 0.0;
            for(uint i = 0; i < {inp1_mats}; i++) {{
                sum += grad_buff[i * {inp1_total} + id];
            }}
            b_grad[id] += sum;
        }}
    }}"
    );

    GpuFunction {
        shared_buffers: vec![SharedBuffer::Float(np * mats)],
        forward_funcs: vec![KernelCall {
            source_code: forward_source_code,
            kernel_name: format!("calc_{}", out_id),
            local_work_size: 32,
            global_work_size: works_forward,
        }],
        backward_funcs: vec![
            KernelCall {
                source_code: backward_source_code_part_1,
                kernel_name: format!("grad_{}_1", out_id),
                local_work_size: 32,
                global_work_size: works_1,
            },
            KernelCall {
                source_code: backward_source_code_part_2,
                kernel_name: format!("grad_{}_2", out_id),
                local_work_size: 32,
                global_work_size: works_2,
            },
            KernelCall {
                source_code: backward_source_code_part_3,
                kernel_name: format!("grad_{}_3", out_id),
                local_work_size: 32,
                global_work_size: inp1_total,
            },
        ],
    }
}
