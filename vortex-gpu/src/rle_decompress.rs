// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(dead_code)]

use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, DeviceRepr, LaunchConfig, PushKernelArg, ValidAsZeroBits,
};
use cudarc::nvrtc::Ptx;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_array::{ArrayRef, IntoArray, ToCanonical};
use vortex_buffer::BufferMut;
use vortex_dtype::{
    NativePType, UnsignedPType, match_each_native_ptype, match_each_unsigned_integer_ptype,
};
use vortex_error::{VortexExpect, VortexResult, vortex_err, vortex_panic};
use vortex_fastlanes::RLEArray;

#[allow(clippy::cognitive_complexity)]
pub fn cuda_rle_decompress(rle: &RLEArray, ctx: Arc<CudaContext>) -> VortexResult<ArrayRef> {
    assert_eq!(rle.offset(), 0);
    assert_eq!(
        rle.values_idx_offsets()
            .scalar_at(0)
            .as_primitive()
            .as_::<u64>()
            .vortex_expect("non null offset"),
        0u64
    );

    match_each_native_ptype!(rle.values().dtype().as_ptype(), |V| {
        match_each_unsigned_integer_ptype!(rle.values_idx_offsets().dtype().as_ptype(), |O| {
            // RLE indices are always u16 (or u8 if downcasted).
            match rle.indices().dtype().as_ptype() {
                PType::U8 => cuda_rle_decompress_typed(
                    rle.indices().to_primitive().as_slice::<u8>(),
                    rle.values().to_primitive().as_slice::<V>(),
                    rle.values_idx_offsets().to_primitive().as_slice::<O>(),
                    rle.len(),
                    ctx,
                ),
                PType::U16 => cuda_rle_decompress_typed(
                    rle.indices().to_primitive().as_slice::<u16>(),
                    rle.values().to_primitive().as_slice::<V>(),
                    rle.values_idx_offsets().to_primitive().as_slice::<O>(),
                    rle.len(),
                    ctx,
                ),
                _ => vortex_panic!(
                    "Unsupported index type for RLE decoding: {}",
                    rle.indices().dtype().as_ptype()
                ),
            }
        })
    })
}

fn cuda_rle_decompress_typed<Values, Indices, Offsets>(
    indices: &[Indices],
    values: &[Values],
    offsets: &[Offsets],
    len: usize,
    ctx: Arc<CudaContext>,
) -> VortexResult<ArrayRef>
where
    Values: NativePType + DeviceRepr + ValidAsZeroBits,
    Indices: UnsignedPType + DeviceRepr,
    Offsets: UnsignedPType + DeviceRepr,
{
    let kernel_func = cuda_rle_kernel::<Indices, Values, Offsets>(ctx.clone())?;
    let num_chunks =
        u32::try_from(indices.len().div_ceil(1024)).vortex_expect("num chunks overflow");
    let stream = ctx.default_stream();

    let cu_indices = stream
        .memcpy_stod(indices)
        .map_err(|e| vortex_err!("Failed to copy to device: {e}"))?;
    let cu_values = stream
        .memcpy_stod(values)
        .map_err(|e| vortex_err!("Failed to copy to device: {e}"))?;
    let cu_offsets = stream
        .memcpy_stod(offsets)
        .map_err(|e| vortex_err!("Failed to copy to device: {e}"))?;

    let output_len = len.next_multiple_of(1024);
    let mut cu_out = unsafe {
        stream
            .alloc::<Values>(output_len)
            .map_err(|e| vortex_err!("Failed to allocate stream: {e}"))?
    };

    let mut launch = stream.launch_builder(&kernel_func);
    launch.arg(&cu_indices);
    launch.arg(&cu_values);
    launch.arg(&cu_offsets);
    launch.arg(&mut cu_out);
    unsafe {
        launch.launch(LaunchConfig {
            grid_dim: (num_chunks, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .map_err(|e| vortex_err!("Failed to launch: {e}"))?;

    let mut buffer = BufferMut::<Values>::with_capacity(output_len);
    unsafe { buffer.set_len(output_len) }

    stream
        .memcpy_dtoh(&cu_out, &mut buffer)
        .map_err(|e| vortex_err!("Failed to copy to device: {e}"))?;
    stream
        .synchronize()
        .map_err(|e| vortex_err!("Failed to synchronize: {e}"))?;

    Ok(PrimitiveArray::new(buffer.freeze().slice(0..len), Validity::NonNullable).into_array())
}

fn cuda_rle_kernel<Indices, Values, Offsets>(ctx: Arc<CudaContext>) -> VortexResult<CudaFunction>
where
    Indices: UnsignedPType,
    Values: NativePType,
    Offsets: UnsignedPType,
{
    let module = ctx
        .load_module(Ptx::from_file("kernels/rle_decompress.ptx"))
        .map_err(|e| vortex_err!("Failed to load kernel module: {e}"))?;

    let kernel_name = format!(
        "rle_decompress_i{}_v{}_o{}",
        &Indices::PTYPE,
        &Values::PTYPE,
        &Offsets::PTYPE,
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
    use vortex_array::ToCanonical;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_dtype::NativePType;
    use vortex_error::VortexUnwrap;
    use vortex_fastlanes::RLEArray;

    use crate::rle_decompress::cuda_rle_decompress;

    #[rstest]
    #[case::u8((0u8..100).collect::<Buffer<u8>>())]
    #[case::u16((0u16..2000).collect::<Buffer<u16>>())]
    #[case::u32((0u32..2000).collect::<Buffer<u32>>())]
    #[case::u64((0u64..2000).collect::<Buffer<u64>>())]
    #[case::i8((-100i8..100).collect::<Buffer<i8>>())]
    #[case::i16((-2000i16..2000).collect::<Buffer<i16>>())]
    #[case::i32((-2000i32..2000).collect::<Buffer<i32>>())]
    #[case::i64((-2000i64..2000).collect::<Buffer<i64>>())]
    #[case::f32((-2000..2000).map(|i| i as f32).collect::<Buffer<f32>>())]
    #[case::f64((-2000..2000).map(|i| i as f64).collect::<Buffer<f64>>())]
    fn test_cuda_rle_decompress<T: NativePType>(#[case] values: Buffer<T>) {
        let primitive_array = PrimitiveArray::new(values, Validity::NonNullable);
        let array = RLEArray::encode(&primitive_array).vortex_unwrap();
        let ctx = CudaContext::new(0).unwrap();
        ctx.set_blocking_synchronize().unwrap();
        let unpacked = cuda_rle_decompress(&array, ctx).unwrap().to_primitive();
        assert_eq!(
            primitive_array.as_slice::<u32>(),
            unpacked.as_slice::<u32>()
        );
    }
}
