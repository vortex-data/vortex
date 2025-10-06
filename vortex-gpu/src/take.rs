// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem::transmute;
use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, DeviceRepr, LaunchConfig, PushKernelArg, ValidAsZeroBits,
};
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
use vortex_mask::Mask;

pub fn cuda_take(dict: &DictArray, ctx: Arc<CudaContext>) -> VortexResult<Option<ArrayRef>> {
    cuda_take_masked(dict, None, ctx)
}

// For now we only support integer non-nullable codes and values.
pub fn cuda_take_masked(
    dict: &DictArray,
    mask: Option<Mask>,
    ctx: Arc<CudaContext>,
) -> VortexResult<Option<ArrayRef>> {
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
            cuda_take_impl::<C, V>(codes, values, mask, ctx)
        })
    });
    result.map(Some)
}

fn cuda_take_impl<Codes, Values>(
    codes: PrimitiveArray,
    values: PrimitiveArray,
    mask: Option<Mask>,
    ctx: Arc<CudaContext>,
) -> VortexResult<ArrayRef>
where
    Codes: UnsignedPType + DeviceRepr,
    Values: NativePType + DeviceRepr + ValidAsZeroBits,
{
    let values_sl = values.as_slice::<Values>();
    let codes_sl = codes.as_slice::<Codes>();

    assert!(values.len() <= 1024);
    assert_eq!(codes.len() % 1024, 0);

    let kernel_func = cuda_take_kernel::<Codes, Values>(mask.is_some(), ctx.clone())?;
    let num_chunks = u32::try_from(codes.len().div_ceil(1024)).vortex_expect("num chunks overflow");
    let stream = ctx.default_stream();

    let cu_values = stream
        .memcpy_stod(values_sl)
        .map_err(|e| vortex_err!("Failed to copy to device: {e}"))?;
    let cu_codes = stream
        .memcpy_stod(codes_sl)
        .map_err(|e| vortex_err!("Failed to copy to device: {e}"))?;
    let mut cu_out = {
        // TODO(joe): use uninit memory
        stream
            .alloc_zeros::<Values>(codes.len().next_multiple_of(1024))
            .map_err(|e| vortex_err!("Failed to allocate stream: {e}"))?
    };

    let cu_mask = mask
        .map(|mask| {
            let buffer = mask.to_boolean_buffer();
            assert_eq!(buffer.offset(), 0);
            assert_eq!(buffer.len() % 1024, 0);
            assert!((buffer.values().as_ptr() as *const u32).is_aligned());
            // SAFETY: we've checked alignment and the layout is the same.
            let slice: &[u32] = unsafe { transmute(buffer.values()) };
            stream
                .memcpy_stod(slice)
                .map_err(|e| vortex_err!("Failed to copy to device: {e}"))
        })
        .transpose()?;

    let mut launch = stream.launch_builder(&kernel_func);
    launch.arg(&cu_codes);
    launch.arg(&cu_values);
    if let Some(cu_mask) = cu_mask.as_ref() {
        launch.arg(cu_mask);
    }
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

fn cuda_take_kernel<Codes, Values>(mask: bool, ctx: Arc<CudaContext>) -> VortexResult<CudaFunction>
where
    Codes: NativePType,
    Values: NativePType,
{
    let module = ctx
        .load_module(Ptx::from_file("kernels/dict_take.ptx"))
        .map_err(|e| vortex_err!("Failed to load kernel module: {e}"))?;

    let kernel_name = format!(
        "dict_take{}_c{}_v{}",
        if mask { "_masked" } else { "" },
        &Codes::PTYPE,
        &Values::PTYPE
    );

    let kernel_func = module
        .load_function(&kernel_name)
        .map_err(|e| vortex_err!("Failed to load function: {e}"))?;
    Ok(kernel_func)
}

#[cfg(all(target_os = "linux", feature = "cuda"))]
#[cfg(test)]
mod tests {
    use cudarc::driver::CudaContext;
    use rstest::rstest;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::{IntoArray, ToCanonical};
    use vortex_buffer::BufferMut;
    use vortex_dict::DictArray;
    use vortex_dtype::NativePType;
    use vortex_error::VortexExpect;
    use vortex_mask::Mask;

    use crate::take::{cuda_take, cuda_take_masked};

    #[rstest]
    #[case::u8_u8(0u8, 0u8)]
    #[case::u8_u16(0u8, 0u16)]
    #[case::u8_u32(0u8, 0u32)]
    #[case::u8_u64(0u8, 0u64)]
    #[case::u8_i8(0u8, 0i8)]
    #[case::u8_i16(0u8, 0i16)]
    #[case::u8_i32(0u8, 0i32)]
    #[case::u8_i64(0u8, 0i64)]
    #[case::u16_u8(0u16, 0u8)]
    #[case::u16_u16(0u16, 0u16)]
    #[case::u16_u32(0u16, 0u32)]
    #[case::u16_u64(0u16, 0u64)]
    #[case::u16_i8(0u16, 0i8)]
    #[case::u16_i16(0u16, 0i16)]
    #[case::u16_i32(0u16, 0i32)]
    #[case::u16_i64(0u16, 0i64)]
    #[case::u32_u8(0u32, 0u8)]
    #[case::u32_u16(0u32, 0u16)]
    #[case::u32_u32(0u32, 0u32)]
    #[case::u32_u64(0u32, 0u64)]
    #[case::u32_i8(0u32, 0i8)]
    #[case::u32_i16(0u32, 0i16)]
    #[case::u32_i32(0u32, 0i32)]
    #[case::u32_i64(0u32, 0i64)]
    #[case::u64_u8(0u64, 0u8)]
    #[case::u64_u16(0u64, 0u16)]
    #[case::u64_u32(0u64, 0u32)]
    #[case::u64_u64(0u64, 0u64)]
    #[case::u64_i8(0u64, 0i8)]
    #[case::u64_i16(0u64, 0i16)]
    #[case::u64_i32(0u64, 0i32)]
    #[case::u64_i64(0u64, 0i64)]
    fn test_cuda_take_all_types<C: NativePType, V: NativePType>(
        #[case] _codes_type: C,
        #[case] _values_type: V,
    ) {
        let values: PrimitiveArray = (0..1024)
            .map(|x| V::from((x + 2) % 1024).unwrap())
            .collect::<BufferMut<V>>()
            .into_array()
            .to_primitive();
        let codes: PrimitiveArray = (0..1024)
            .map(|x| C::from((x + 1) % 1024).unwrap())
            .collect::<BufferMut<C>>()
            .into_array()
            .to_primitive();
        let dict = DictArray::try_new(codes.into_array(), values.into_array()).unwrap();

        let expect = dict.to_primitive();

        let ctx = CudaContext::new(0).unwrap();
        ctx.set_blocking_synchronize().unwrap();
        let result = cuda_take(&dict, ctx).unwrap().unwrap().to_primitive();

        assert_eq!(result.as_slice::<V>(), expect.as_slice::<V>());
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

    #[test]
    fn test_cuda_take_masked() {
        const LEN: u64 = 1024 * 8;
        let values: PrimitiveArray = (0u64..1024).map(|x| (x + 2) % 1024).collect();
        let codes: PrimitiveArray = (0u64..LEN).map(|x| (x + 1) % 1024).collect();
        let dict = DictArray::try_new(codes.into_array(), values.into_array()).unwrap();

        let expect = dict.to_primitive();

        let mask = Mask::from_iter((0..LEN).map(|i| (i % 4) == 0));

        let ctx = CudaContext::new(0).unwrap();
        ctx.set_blocking_synchronize().unwrap();
        let result = cuda_take_masked(&dict, Some(mask.clone()), ctx)
            .unwrap()
            .unwrap()
            .to_primitive();

        let result_sl = result.as_slice::<u64>();
        let expect_sl = expect.as_slice::<u64>();

        mask.to_boolean_buffer().set_indices().for_each(|i| {
            assert_eq!(result_sl[i], expect_sl[i]);
        })
    }
}
