use super::*;

pub fn gpu_impl(out_id: TensorId, inps: &[Vec<usize>]) -> GpuFunction {
    let works = inps[0].iter().fold(1, |a, b| a * b);

    let forward_source_code = format!(
        "__kernel void calc_{out_id}(
                        __global float* out,
                        __global float* a) {{
        uint id = get_global_id(0);
        float SQRT_2_OVER_PI = 0.7978845608;
        float GELU_CONST = 0.044715;
        if(id < {works}) {{
            float x = a[id];
            float x3 = x * x * x;
            out[id] = 0.5 * x * (tanh(SQRT_2_OVER_PI * (x + GELU_CONST * x3)) + 1.);
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
        float SQRT_2_OVER_PI = 0.7978845608;
        float GELU_CONST = 0.044715;
        if(id < {works}) {{
            float x = a[id];
            float x2 = x * x;
            float x3 = x2 * x;
            float v = SQRT_2_OVER_PI * x + SQRT_2_OVER_PI * GELU_CONST * x3;
            float v_prime = SQRT_2_OVER_PI + 3. * SQRT_2_OVER_PI * GELU_CONST * x2;
            float cosh_v = cosh(v);
            float sech_2_v = 1. / (cosh_v * cosh_v);
            a_grad[id] += 0.5 * (1. + tanh(v) + x * sech_2_v * v_prime) * out_grad[i];
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
