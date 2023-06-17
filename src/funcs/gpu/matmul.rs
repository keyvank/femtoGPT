use super::*;

pub fn gpu_run(out_id: TensorId, inps: &[Vec<usize>]) -> GpuFunction {
    let inp0_size = inps[0][..inps[0].len() - 2].iter().fold(1, |a, b| a * b);
    let inp1_size = inps[1][..inps[1].len() - 2].iter().fold(1, |a, b| a * b);
    assert_eq!(inps[0][inps[0].len() - 1], inps[1][inps[1].len() - 2]);
    let m = inps[0][inps[0].len() - 2];
    let n = inps[0][inps[0].len() - 1];
    let p = inps[1][inps[1].len() - 1];
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
            out += {m} * {p} * id;
            a += {m} * {n} * id_a;
            b += {n} * {p} * id_b;
            for(uint i = 0; i < {m} * {p}; i++) out[i] = 0.;
            for(uint i = 0; i < {m}; i++) {{
                for(uint k = 0; k < {n}; k++) {{
                    for(uint j = 0; j < {p}; j++) {{
                        out[i * {p} + j] += a[i * {n} + k] * b[{p} * k + j];
                    }}
                }}
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
    assert_eq!(inps[0][inps[0].len() - 1], inps[1][inps[1].len() - 2]);
    let m = inps[0][inps[0].len() - 2];
    let n = inps[0][inps[0].len() - 1];
    let p = inps[1][inps[1].len() - 1];
    let mp = m * p;
    let mn = m * n;
    let np = n * p;
    let works = std::cmp::max(inp0_size, inp1_size);
    let source_code = format!(
        "__kernel void grad_{out_id}_1(
                        __global float* out,
                        __global float* out_grad,
                        __global float* grad_buff,
                        __global float* a,
                        __global float* a_grad,
                        __global float* b,
                        __global float* b_grad) {{
        uint id = get_global_id(0);
        uint id_a = id % {inp0_size};
        uint id_b = id % {inp1_size};
        if(id < {works}) {{
            out += {mp} * id;
            out_grad += {mp} * id;
            a += {mn} * id_a;
            a_grad += {mn} * id_a;
            b += {np} * id_b;
            float *gb = grad_buff + {np} * id;

            for(uint i = 0; i < {np}; i++) {{
                gb[i] = 0.0;
            }}

            // a_grad = (out_grad ^ b_T) -> mp * pn
            // b_grad = (a_T ^ out_grad) -> nm * mp
            for(uint i = 0; i < {m}; i++) {{
                for(uint j = 0; j < {p}; j++) {{
                    for(uint k = 0; k < {n}; k++) {{
                        a_grad[i * {n} + k] += out_grad[i * {p} + j] * b[k * {p} + j];
                        gb[k * {p} + j] += a[i * {n} + k] * out_grad[i * {p} + j];
                    }}
                }}
            }}
        }}
    }}"
    );

    let source_code_2 = format!(
        "__kernel void grad_{out_id}_2(
                        __global float* out,
                        __global float* out_grad,
                        __global float* grad_buff,
                        __global float* a,
                        __global float* a_grad,
                        __global float* b,
                        __global float* b_grad) {{
        uint id = get_global_id(0);
        if(id < {np}) {{
            for(uint i = 0; i < {works}; i++) {{
                b_grad[id] += grad_buff[i * {np} + id];
            }}
        }}
    }}"
    );

    GpuFunctionGroup {
        shared_buffers: vec![np * works],
        funcs: vec![
            GpuFunction {
                source_code,
                kernel_name: format!("grad_{}_1", out_id),
                local_work_size: 32,
                global_work_size: works + ((32 - (works % 32)) % 32),
            },
            GpuFunction {
                source_code: source_code_2,
                kernel_name: format!("grad_{}_2", out_id),
                local_work_size: 32,
                global_work_size: np + ((32 - (np % 32)) % 32),
            },
        ],
    }
}
