use super::*;

pub fn gpu_impl(out_id: TensorId, inps: &[Vec<usize>]) -> GpuFunction {
    let works = inps[1].iter().fold(1, |a, b| a * b);
    let classes = inps[0].last().unwrap();

    let forward_source_code = format!(
        "__kernel void calc_{out_id}(
                        __global float* out,
                        __global float* exps_buff,
                        __global float* sum_buff,
                        __global float* inp,
                        __global ulong* expected) {{
        uint id = get_global_id(0);
        out += id;
        expected += id;
        inp += {classes} * id;
        sum_buff += id;
        exps_buff += {classes} * id;
        if(id < {works}) {{
            float sum = 0.0;
            for(uint i = 0; i < {classes}; i++) {{
                exps_buff[i] = exp(inp[i]);
                sum += exps_buff[i];
            }}
            *sum_buff = sum;
            *out = log(sum) - inp[*expected];
        }}
    }}"
    );

    let backward_source_code = format!(
        "__kernel void grad_{out_id}(
                        __global float* out,
                        __global float* out_grad,
                        __global float* exps_buff,
                        __global float* sum_buff,
                        __global float* inp,
                        __global float* inp_grad,
                        __global ulong* expected,
                        __global float* expected_grad) {{
        uint wid = get_global_id(0);
        uint id = wid / {classes};
        uint c = wid % {classes};
        sum_buff += id;
        exps_buff += {classes} * id;
        inp_grad += {classes} * id;
        out_grad += id;
        out += id;
        expected += id;
        inp += {classes} * id;
        if(wid < {works} * {classes}) {{
            float val = exps_buff[c];
            float sum_inv = 1.0 / *sum_buff;

            float grad = val * sum_inv;
            if(c == *expected) {{
                grad = grad - 1.0;
            }}
            grad *= *out_grad;
            inp_grad[c] += grad;
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
            global_work_size: works * classes,
        }],
        shared_buffers: vec![
            SharedBuffer::Float(works * classes),
            SharedBuffer::Float(works),
        ],
    }
}
