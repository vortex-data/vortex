// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, DeviceRepr, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_array::{ArrayRef, Canonical, IntoArray, ToCanonical};
use vortex_buffer::BufferMut;
use vortex_dict::DictArray;
use vortex_dtype::{
    DType, NativePType, Nullability, UnsignedPType, match_each_native_ptype,
    match_each_unsigned_integer_ptype,
};
use vortex_error::{VortexExpect, VortexResult, vortex_err};

// For now we only support integer non-nullable codes and values.
pub fn cuda_take(dict: &DictArray, ctx: Arc<CudaContext>) -> VortexResult<Option<ArrayRef>> {
    if !matches!(dict.dtype(), DType::Primitive(_, Nullability::NonNullable)) {
        return Ok(None);
    };

    if dict.is_empty() {
        return Ok(Some(Canonical::empty(dict.dtype()).into_array()));
    }

    let values = dict.values().to_primitive();
    let codes = dict.codes().to_primitive();

    let result = match_each_native_ptype!(values.ptype(), |V| {
        match_each_unsigned_integer_ptype!(codes.ptype(), |C| {
            cuda_take_impl::<C, V>(codes, values, ctx)
        })
    });
    result.map(Some)
}

fn cuda_take_impl<Codes, Values>(
    codes: PrimitiveArray,
    values: PrimitiveArray,
    ctx: Arc<CudaContext>,
) -> VortexResult<ArrayRef>
where
    Codes: UnsignedPType + DeviceRepr,
    Values: NativePType + DeviceRepr,
{
    let values_sl = values.as_slice::<Values>();
    let codes_sl = codes.as_slice::<Codes>();

    assert!(values.len() <= 1024);
    assert_eq!(codes.len() % 1024, 0);

    let kernel_func = cuda_take_kernel::<Codes, Values>(ctx.clone())?;
    let num_chunks = u32::try_from(codes.len().div_ceil(1024)).vortex_expect("num chunks overflow");
    let stream = ctx.default_stream();

    let cu_values = stream
        .memcpy_stod(values_sl)
        .map_err(|e| vortex_err!("Failed to copy to device: {e}"))?;
    let cu_codes = stream
        .memcpy_stod(codes_sl)
        .map_err(|e| vortex_err!("Failed to copy to device: {e}"))?;
    let mut cu_out = unsafe {
        stream
            .alloc::<Values>(codes.len().next_multiple_of(1024))
            .map_err(|e| vortex_err!("Failed to allocate stream: {e}"))?
    };

    let mut launch = stream.launch_builder(&kernel_func);
    launch.arg(&cu_codes);
    launch.arg(&cu_values);
    launch.arg(&mut cu_out);
    unsafe {
        launch.launch(LaunchConfig {
            grid_dim: (num_chunks, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .map_err(|e| vortex_err!("Failed to launch: {e}"))?;

    let mut buffer = BufferMut::<Values>::with_capacity(codes.len());
    unsafe { buffer.set_len(codes.len()) }

    stream
        .memcpy_dtoh(&cu_out, &mut buffer)
        .map_err(|e| vortex_err!("Failed to copy to device: {e}"))?;
    stream
        .synchronize()
        .map_err(|e| vortex_err!("Failed to synchronize: {e}"))?;

    Ok(PrimitiveArray::new(buffer, Validity::NonNullable).into_array())
}

fn cuda_take_kernel<Codes, Values>(ctx: Arc<CudaContext>) -> VortexResult<CudaFunction>
where
    Codes: NativePType,
    Values: NativePType,
{
    let module = ctx
        .load_module(Ptx::from_file("kernels/dict_take.ptx"))
        .map_err(|e| vortex_err!("Failed to load kernel module: {e}"))?;

    let kernel_name = format!("dict_take_c{}_v{}_values", &Codes::PTYPE, &Values::PTYPE);

    let kernel_func = module
        .load_function(&kernel_name)
        .map_err(|e| vortex_err!("Failed to load function: {e}"))?;
    Ok(kernel_func)
}

#[cfg(all(target_os = "linux", feature = "cuda"))]
#[cfg(test)]
mod tests {
    use cudarc::driver::CudaContext;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::{IntoArray, ToCanonical};
    use vortex_dict::DictArray;
    use vortex_error::VortexExpect;

    use crate::take::cuda_take;

    #[test]
    fn test_cuda_take_u32_u32() {
        let values: PrimitiveArray = (0u32..1024).map(|x| (x + 2) % 1024).collect();
        let codes: PrimitiveArray = (0u32..1024).map(|x| (x + 1) % 1024).collect();
        let dict = DictArray::try_new(codes.into_array(), values.into_array()).unwrap();

        let expect: PrimitiveArray = (0u32..1024).map(|x| (x + 3) % 1024).collect();
        let dict_cpu = dict.to_primitive();
        assert_eq!(dict_cpu.as_slice::<u32>(), expect.as_slice::<u32>());

        let ctx = CudaContext::new(0).unwrap();
        ctx.set_blocking_synchronize().unwrap();
        let result = cuda_take(&dict, ctx).unwrap().unwrap().to_primitive();

        assert_eq!(result.as_slice::<u32>(), expect.as_slice::<u32>());
    }

    #[test]
    fn test_cuda_take_u64_i64() {
        let values: PrimitiveArray = (0i64..1024).map(|x| (x + 2) % 1024).collect();
        let codes: PrimitiveArray = (0u64..1024).map(|x| (x + 1) % 1024).collect();
        let dict = DictArray::try_new(codes.into_array(), values.into_array()).unwrap();

        let expect: PrimitiveArray = (0i64..1024).map(|x| (x + 3) % 1024).collect();
        let dict_cpu = dict.to_primitive();
        assert_eq!(dict_cpu.as_slice::<i64>(), expect.as_slice::<i64>());

        let ctx = CudaContext::new(0).unwrap();
        ctx.set_blocking_synchronize().unwrap();
        let result = cuda_take(&dict, ctx).unwrap().unwrap().to_primitive();

        assert_eq!(result.as_slice::<i64>(), expect.as_slice::<i64>());
    }

    #[test]
    fn test_cuda_take_long_u8_i64() {
        const LEN: usize = 1024 * 8;
        let values: PrimitiveArray = (0i64..1024).map(|x| (x + 2) % 1024).collect();
        let codes: PrimitiveArray = (0..LEN)
            .map(|x| u8::try_from((x + 1) % 255).vortex_expect(""))
            .collect();
        let dict = DictArray::try_new(codes.into_array(), values.into_array()).unwrap();

        let expect = dict.to_primitive();

        let ctx = CudaContext::new(0).unwrap();
        ctx.set_blocking_synchronize().unwrap();
        let result = cuda_take(&dict, ctx).unwrap().unwrap().to_primitive();

        assert_eq!(result.as_slice::<i64>(), expect.as_slice::<i64>());
    }
}
