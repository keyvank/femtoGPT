use super::*;

pub fn gpu_run(out_id: TensorId, inps: &[Vec<usize>]) -> GpuFunction {
    let inp0_size = inps[0][..inps[0].len() - 2].iter().fold(1, |a, b| a * b);
    let inp1_size = inps[1][..inps[1].len() - 2].iter().fold(1, |a, b| a * b);
    assert_eq!(inps[0][inps[0].len() - 1], inps[1][inps[1].len() - 2]);
    let m = inps[0][inps[0].len() - 2];
    let n = inps[0][inps[0].len() - 1];
    let p = inps[1][inps[1].len() - 1];
    let mp = m * p;
    let works = std::cmp::max(inp0_size, inp1_size) * m * p;
    let source_code = format!(
        "__kernel void calc_{out_id}(
                        __global float* out,
                        __global float* a,
                        __global float* b) {{
        uint wid = get_global_id(0);
        uint id = wid / {mp};
        uint id_a = id % {inp0_size};
        uint id_b = id % {inp1_size};
        uint ij = wid % {mp};
        uint i = ij / {p};
        uint j = ij % {p};
        if(wid < {works}) {{
            out += {m} * {p} * id;
            a += {m} * {n} * id_a;
            b += {n} * {p} * id_b;
            out[ij] = 0.0;
            for(uint k = 0; k < {n}; k++) {{
                out[i * {p} + j] += a[i * {n} + k] * b[{p} * k + j];
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

pub fn gpu_grad(out_id: TensorId, inps: &[Vec<usize>]) -> GpuFunctionGroup {
    let inp0_size = inps[0][..inps[0].len() - 2].iter().fold(1, |a, b| a * b);
    let inp1_size = inps[1][..inps[1].len() - 2].iter().fold(1, |a, b| a * b);
    assert!(inp1_size <= inp0_size);
    assert_eq!(inps[0][inps[0].len() - 1], inps[1][inps[1].len() - 2]);
    let m = inps[0][inps[0].len() - 2];
    let n = inps[0][inps[0].len() - 1];
    let p = inps[1][inps[1].len() - 1];
    let mp = m * p;
    let mn = m * n;
    let np = n * p;
    let mats = std::cmp::max(inp0_size, inp1_size);
    let works_1 = mats * mn;
    let source_code = format!(
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
        uint id_a = id % {inp0_size};
        uint id_b = id % {inp1_size};
        uint ik = wid % {mn};
        uint i = ik / {n};
        uint k = ik % {n};
        if(wid < {works_1}) {{
            out += {mp} * id;
            out_grad += {mp} * id;
            a += {mn} * id_a;
            a_grad += {mn} * id_a;
            b += {np} * id_b;

            // a_grad = (out_grad ^ b_T) -> mp * pn
            // b_grad = (a_T ^ out_grad) -> nm * mp
            for(uint j = 0; j < {p}; j++) {{
                a_grad[i * {n} + k] += out_grad[i * {p} + j] * b[k * {p} + j];
            }}
        }}
    }}"
    );

    let works_2 = mats * np;
    let source_code_2 = format!(
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
        uint id_a = id % {inp0_size};
        uint id_b = id % {inp1_size};
        uint kj = wid % {np};
        uint k = kj / {p};
        uint j = kj % {p};
        if(id < {works_2}) {{
            out += {mp} * id;
            out_grad += {mp} * id;
            a += {mn} * id_a;
            a_grad += {mn} * id_a;
            b += {np} * id_b;
            float *gb = grad_buff + {np} * id;

            gb[k * {p} + j] = 0.0;

            // a_grad = (out_grad ^ b_T) -> mp * pn
            // b_grad = (a_T ^ out_grad) -> nm * mp
            for(uint i = 0; i < {m}; i++) {{
                gb[k * {p} + j] += a[i * {n} + k] * out_grad[i * {p} + j];
            }}
        }}
    }}"
    );

    let inp1_mats = mats / inp1_size;
    let inp1_total = inp1_size * np;
    let source_code_3 = format!(
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
            for(uint i = 0; i < {inp1_mats}; i++) {{
                b_grad[id] += grad_buff[i * {inp1_total} + id];
            }}
        }}
    }}"
    );

    GpuFunctionGroup {
        shared_buffers: vec![np * mats],
        funcs: vec![
            GpuFunction {
                source_code,
                kernel_name: format!("grad_{}_1", out_id),
                local_work_size: 32,
                global_work_size: works_1 + ((32 - (works_1 % 32)) % 32),
            },
            GpuFunction {
                source_code: source_code_2,
                kernel_name: format!("grad_{}_2", out_id),
                local_work_size: 32,
                global_work_size: works_2 + ((32 - (works_2 % 32)) % 32),
            },
            GpuFunction {
                source_code: source_code_3,
                kernel_name: format!("grad_{}_3", out_id),
                local_work_size: 32,
                global_work_size: inp1_total + ((32 - (inp1_total % 32)) % 32),
            },
        ],
    }
}
