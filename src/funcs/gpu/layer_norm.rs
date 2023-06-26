use super::*;

pub fn gpu_impl(out_id: TensorId, inps: &[Vec<usize>]) -> GpuFunction {
    let n = inps[0][inps[0].len() - 1];
    let works = inps[0][..inps[0].len() - 1].iter().fold(1, |a, b| a * b);

    let forward_source_code = format!(
        "__kernel void calc_{out_id}(
                        __global float* out,
                        __global float* coeff_grad_temp,
                        __global float* avg_buff,
                        __global float* sigma2_buff,
                        __global float* a,
                        __global float* coeff,
                        __global float* bias) {{
        uint id = get_global_id(0);
        if(id < {works}) {{
            a += id * {n};
            out += id * {n};
            float size_inv = 1./{n};
            float avg = 0.;
            for(uint i = 0; i < {n}; i++) {{
                avg += a[i];
            }}
            avg *= size_inv;
            avg_buff[id] = avg;
            float var = 0.;
            for(uint i = 0; i < {n}; i++) {{
                var += (a[i] - avg) * (a[i] - avg);
            }}
            var *= size_inv;
            sigma2_buff[id] = var;
            float var_inv = 1. / sqrt(var + 1e-5);
            for(uint i = 0; i < {n}; i++) {{
                out[i] = (a[i] - avg) * var_inv * coeff[i] + bias[i];
            }}
        }}
    }}"
    );

    let backward_source_code_part_1 = format!(
        "__kernel void grad_{out_id}_0(
                        __global float* out,
                        __global float* out_grad,
                        __global float* coeff_grad_temp,
                        __global float* avg_buff,
                        __global float* sigma2_buff,
                        __global float* inp,
                        __global float* inp_grad,
                        __global float* coeff,
                        __global float* coeff_grad,
                        __global float* bias,
                        __global float* bias_grad) {{
        uint wid = get_global_id(0);
        uint id = wid / {n};
        uint i = wid % {n};

        out += id * {n};
        out_grad += id * {n};
        inp += id * {n};
        inp_grad += id * {n};
        coeff_grad_temp += id * {n};

        if(wid < {works} * {n}) {{
            for(uint ii = 0; ii < {n}; ii++) {{
                coeff_grad_temp[ii] = (out[ii] - bias[ii]) * out_grad[ii] / coeff[ii];
            }}

            float n_inv = 1.0 / {n};
            float avg = avg_buff[id];
            float sigma2 = sigma2_buff[id];
            sigma2 += 0.00001;
            float sigma2_inv = 1.0 / sigma2;
            float sigma = sqrt(sigma2);
            float sigma_inv = 1. / sigma;

            float a = inp[i];
            float sum = 0.0;
            for(uint j = 0; j < {n}; j++) {{
                if(i == j) {{
                    sum += ((1. - n_inv) * sigma - (a - avg) * (a - avg) * sigma_inv * n_inv) * sigma2_inv * out_grad[j] * coeff[j];
                }} else {{
                    float b = inp[j];
                    sum += (-n_inv * sigma - (b - avg) * (a - avg) * sigma_inv * n_inv) * sigma2_inv * out_grad[j] * coeff[j];
                }}
            }}
            inp_grad[i] += sum;
        }}
    }}"
    );

    let backward_source_code_part_2 = format!(
        "__kernel void grad_{out_id}_1(
                        __global float* out,
                        __global float* out_grad,
                        __global float* coeff_grad_temp,
                        __global float* avg_buff,
                        __global float* sigma2_buff,
                        __global float* inp,
                        __global float* inp_grad,
                        __global float* coeff,
                        __global float* coeff_grad,
                        __global float* bias,
                        __global float* bias_grad) {{
        uint id = get_global_id(0);
        if(id < {n}) {{
            for(uint i = 0; i < {works}; i++) {{
                coeff_grad[id] += coeff_grad_temp[i * {n} + id];
                bias_grad[id] += out_grad[i * {n} + id];
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
        backward_funcs: vec![
            KernelCall {
                source_code: backward_source_code_part_1,
                kernel_name: format!("grad_{}_0", out_id),
                local_work_size: 32,
                global_work_size: works * n,
            },
            KernelCall {
                source_code: backward_source_code_part_2,
                kernel_name: format!("grad_{}_1", out_id),
                local_work_size: 32,
                global_work_size: n,
            },
        ],
        shared_buffers: vec![
            SharedBuffer::Float(n * works),
            SharedBuffer::Float(works),
            SharedBuffer::Float(works),
        ],
    }
}
