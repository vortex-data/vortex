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

    match_each_native_ptype!(values.ptype(), |V| {
        match_each_unsigned_integer_ptype!(codes.ptype(), |C| {
            cuda_take_impl::<C, V>(codes, values, mask, ctx)
        })
    })
    .map(Some)
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
    let mut cu_out = unsafe {
        stream
            .alloc::<Values>(codes.len().next_multiple_of(1024))
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

    module
        .load_function(&kernel_name)
        .map_err(|e| vortex_err!("Failed to load function: {e}"))
}

#[cfg(all(target_os = "linux", feature = "cuda"))]
#[cfg(test)]
mod tests {
    use cudarc::driver::CudaContext;
    use rstest::rstest;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::{IntoArray, ToCanonical};
    use vortex_dict::DictArray;
    use vortex_dtype::match_each_native_ptype;

    use crate::take::cuda_take;

    #[rstest]
    #[case::u8_u8("u8", "u8")]
    #[case::u8_u16("u8", "u16")]
    #[case::u8_u32("u8", "u32")]
    #[case::u8_u64("u8", "u64")]
    #[case::u8_i8("u8", "i8")]
    #[case::u8_i16("u8", "i16")]
    #[case::u8_i32("u8", "i32")]
    #[case::u8_i64("u8", "i64")]
    #[case::u8_f32("u8", "f32")]
    #[case::u8_f64("u8", "f64")]
    #[case::u16_u8("u16", "u8")]
    #[case::u16_u16("u16", "u16")]
    #[case::u16_u32("u16", "u32")]
    #[case::u16_u64("u16", "u64")]
    #[case::u16_i8("u16", "i8")]
    #[case::u16_i16("u16", "i16")]
    #[case::u16_i32("u16", "i32")]
    #[case::u16_i64("u16", "i64")]
    #[case::u16_f32("u16", "f32")]
    #[case::u16_f64("u16", "f64")]
    #[case::u32_u8("u32", "u8")]
    #[case::u32_u16("u32", "u16")]
    #[case::u32_u32("u32", "u32")]
    #[case::u32_u64("u32", "u64")]
    #[case::u32_i8("u32", "i8")]
    #[case::u32_i16("u32", "i16")]
    #[case::u32_i32("u32", "i32")]
    #[case::u32_i64("u32", "i64")]
    #[case::u32_f32("u32", "f32")]
    #[case::u32_f64("u32", "f64")]
    #[case::u64_u8("u64", "u8")]
    #[case::u64_u16("u64", "u16")]
    #[case::u64_u32("u64", "u32")]
    #[case::u64_u64("u64", "u64")]
    #[case::u64_i8("u64", "i8")]
    #[case::u64_i16("u64", "i16")]
    #[case::u64_i32("u64", "i32")]
    #[case::u64_i64("u64", "i64")]
    #[case::u64_f32("u64", "f32")]
    #[case::u64_f64("u64", "f64")]
    fn test_cuda_take_all_combinations(#[case] code_type: &str, #[case] value_type: &str) {
        const LEN: usize = 1024;

        let ctx = CudaContext::new(0).unwrap();
        ctx.set_blocking_synchronize().unwrap();

        // Generate distinct values for each type using type-specific patterns
        let values: PrimitiveArray = match value_type {
            "u8" => (0u8..255)
                .map(|x| x.wrapping_mul(3).wrapping_add(100))
                .collect(),
            "u16" => (0u16..1000).map(|x| x * 5 + 1000).collect(),
            "u32" => (0u32..1000).map(|x| x * 11 + 100000).collect(),
            "u64" => (0u64..1000).map(|x| x * 17 + 1000000).collect(),
            "i8" => (-50i8..50).map(|x| x.wrapping_mul(2)).collect(),
            "i16" => (0i16..1000).map(|x| x * 7 - 5000).collect(),
            "i32" => (0i32..1000).map(|x| x * 13 - 500000).collect(),
            "i64" => (0i64..1000).map(|x| x * 23 - 10000000).collect(),
            "f32" => (0..1000)
                .map(|x| (x as f32) * std::f32::consts::PI + 1000.5)
                .collect(),
            "f64" => (0..1000)
                .map(|x| (x as f64) * std::f64::consts::E + 100000.123)
                .collect(),
            _ => panic!("Unknown value type"),
        };

        // Generate distinct codes for each code type
        // codes_mod must be compatible with both the value array size and code type max
        let codes_mod = match value_type {
            "u8" => 200,
            "i8" => 100,
            _ => 1000,
        };

        let codes: PrimitiveArray = match code_type {
            "u8" => {
                let max_code = codes_mod.min(255);
                (0..LEN)
                    .map(|x| u8::try_from((x * 13 + 17) % max_code).unwrap())
                    .collect()
            }
            "u16" => (0..LEN)
                .map(|x| u16::try_from((x * 13 + 17) % codes_mod).unwrap())
                .collect(),
            "u32" => (0..LEN)
                .map(|x| u32::try_from((x * 13 + 17) % codes_mod).unwrap())
                .collect(),
            "u64" => (0..LEN)
                .map(|x| u64::try_from((x * 13 + 17) % codes_mod).unwrap())
                .collect(),
            _ => panic!("Unknown code type"),
        };

        let dict = DictArray::try_new(codes.into_array(), values.into_array()).unwrap();
        let expect = dict.to_primitive();
        let result = cuda_take(&dict, ctx).unwrap().unwrap().to_primitive();

        match_each_native_ptype!(expect.ptype(), |P| {
            assert_eq!(
                result.as_slice::<P>(),
                expect.as_slice::<P>(),
                "Failed for {} codes with {} values",
                code_type,
                value_type
            )
        });
    }
}
