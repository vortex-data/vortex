// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(dead_code)]

use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, CudaViewMut, DeviceRepr, LaunchConfig,
    PushKernelArg, ValidAsZeroBits,
};
use cudarc::nvrtc::Ptx;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_array::{Canonical, ToCanonical};
use vortex_buffer::BufferMut;
use vortex_dtype::{
    NativePType, UnsignedPType, match_each_native_ptype, match_each_unsigned_integer_ptype,
};
use vortex_error::{VortexExpect, VortexResult, vortex_err, vortex_panic};
use vortex_fastlanes::RLEArray;

use crate::task::GPUTask;

struct RLETask<V: DeviceRepr + NativePType, I, O> {
    values: CudaSlice<V>,
    indices: CudaSlice<I>,
    offsets: CudaSlice<O>,
    output: CudaSlice<V>,
    func: CudaFunction,
    launch_config: LaunchConfig,
    stream: Arc<CudaStream>,
    len: usize,
}

impl<V: DeviceRepr + NativePType, I, O> GPUTask for RLETask<V, I, O> {
    fn launch_task(&mut self) -> VortexResult<()> {
        let mut launch = self.stream.launch_builder(&self.func);
        launch.arg(&self.indices);
        launch.arg(&self.values);
        launch.arg(&self.offsets);
        launch.arg(&mut self.output);
        unsafe { launch.launch(self.launch_config) }
            .map_err(|e| vortex_err!("Failed to launch: {e}"))
            .map(|_| ())
    }

    fn export_result(&mut self) -> VortexResult<Canonical> {
        let rounded_len = self.len.next_multiple_of(1024);
        let mut buffer = BufferMut::<V>::with_capacity(rounded_len);
        unsafe { buffer.set_len(rounded_len) }

        self.stream
            .memcpy_dtoh(&self.output, &mut buffer)
            .map_err(|e| vortex_err!("Failed to copy to device: {e}"))?;
        self.stream
            .synchronize()
            .map_err(|e| vortex_err!("Failed to synchronize: {e}"))?;

        Ok(Canonical::Primitive(PrimitiveArray::new(
            buffer.freeze().slice(0..self.len),
            Validity::NonNullable,
        )))
    }

    fn output(&mut self) -> CudaViewMut<'_, u8> {
        unsafe {
            self.output
                .transmute_mut(self.len() * size_of::<V>())
                .vortex_expect("Failed to transmute")
        }
    }

    fn len(&self) -> usize {
        self.len
    }
}

#[allow(clippy::cognitive_complexity)]
pub fn new_task(
    rle: &RLEArray,
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
) -> VortexResult<Box<dyn GPUTask>> {
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
                PType::U8 => cuda_rle_task(
                    rle.indices().to_primitive().as_slice::<u8>(),
                    rle.values().to_primitive().as_slice::<V>(),
                    rle.values_idx_offsets().to_primitive().as_slice::<O>(),
                    rle.len(),
                    ctx,
                    stream,
                )
                .map(|t| Box::new(t) as Box<dyn GPUTask>),
                PType::U16 => cuda_rle_task(
                    rle.indices().to_primitive().as_slice::<u16>(),
                    rle.values().to_primitive().as_slice::<V>(),
                    rle.values_idx_offsets().to_primitive().as_slice::<O>(),
                    rle.len(),
                    ctx,
                    stream,
                )
                .map(|t| Box::new(t) as Box<dyn GPUTask>),
                _ => vortex_panic!(
                    "Unsupported index type for RLE decoding: {}",
                    rle.indices().dtype().as_ptype()
                ),
            }
        })
    })
}

fn cuda_rle_task<Values, Indices, Offsets>(
    indices: &[Indices],
    values: &[Values],
    offsets: &[Offsets],
    len: usize,
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
) -> VortexResult<RLETask<Values, Indices, Offsets>>
where
    Values: NativePType + DeviceRepr + ValidAsZeroBits,
    Indices: UnsignedPType + DeviceRepr,
    Offsets: UnsignedPType + DeviceRepr,
{
    let kernel_func = cuda_rle_kernel::<Indices, Values, Offsets>(ctx)?;
    let num_chunks =
        u32::try_from(indices.len().div_ceil(1024)).vortex_expect("num chunks overflow");

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
    let cu_out = unsafe {
        stream
            .alloc::<Values>(output_len)
            .map_err(|e| vortex_err!("Failed to allocate stream: {e}"))?
    };

    Ok(RLETask {
        values: cu_values,
        indices: cu_indices,
        offsets: cu_offsets,
        output: cu_out,
        func: kernel_func,
        launch_config: LaunchConfig {
            grid_dim: (num_chunks, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        },
        stream,
        len,
    })
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

pub fn cuda_rle_decompress(
    array: &RLEArray,
    ctx: Arc<CudaContext>,
) -> VortexResult<PrimitiveArray> {
    let stream = ctx.default_stream();
    let mut task = new_task(array, ctx, stream)?;
    task.launch_task()?;
    task.export_result().map(|c| c.into_primitive())
}

#[cfg(all(target_os = "linux", feature = "cuda"))]
#[cfg(test)]
mod tests {
    use cudarc::driver::CudaContext;
    use rstest::rstest;
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
        let unpacked = cuda_rle_decompress(&array, ctx).unwrap();
        assert_eq!(primitive_array.as_slice::<T>(), unpacked.as_slice::<T>());
    }
}
