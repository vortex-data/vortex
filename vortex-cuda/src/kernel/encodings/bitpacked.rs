// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;

use async_trait::async_trait;
use cudarc::driver::CudaFunction;
use cudarc::driver::DeviceRepr;
use cudarc::driver::LaunchConfig;
use cudarc::driver::PushKernelArg;
use tracing::instrument;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::buffer::BufferHandle;
use vortex::array::buffer::DeviceBufferExt;
use vortex::array::match_each_integer_ptype;
use vortex::dtype::NativePType;
use vortex::encodings::fastlanes::BitPacked;
use vortex::encodings::fastlanes::BitPackedArray;
use vortex::encodings::fastlanes::BitPackedDataParts;
use vortex::encodings::fastlanes::unpack_iter::BitPacked as BitPackedUnpack;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_ensure;
use vortex::error::vortex_err;

use crate::CudaBufferExt;
use crate::CudaDeviceBuffer;
use crate::executor::CudaExecute;
use crate::executor::CudaExecutionCtx;
use crate::kernel::patches::gpu::ChunkOffsetType;
use crate::kernel::patches::gpu::ChunkOffsetType_CO_U8;
use crate::kernel::patches::gpu::ChunkOffsetType_CO_U16;
use crate::kernel::patches::gpu::ChunkOffsetType_CO_U32;
use crate::kernel::patches::gpu::ChunkOffsetType_CO_U64;
use crate::kernel::patches::gpu::GPUPatches;
use crate::kernel::patches::types::load_patches;

/// CUDA decoder for bit-packed arrays.
#[derive(Debug)]
pub(crate) struct BitPackedExecutor;

impl BitPackedExecutor {
    fn try_specialize(array: ArrayRef) -> Option<BitPackedArray> {
        array.try_downcast::<BitPacked>().ok()
    }
}

#[async_trait]
impl CudaExecute for BitPackedExecutor {
    #[instrument(level = "trace", skip_all, fields(executor = ?self))]
    async fn execute(
        &self,
        array: ArrayRef,
        ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<Canonical> {
        let array =
            Self::try_specialize(array).ok_or_else(|| vortex_err!("Expected BitPackedArray"))?;

        match_each_integer_ptype!(array.ptype(array.dtype()), |A| {
            decode_bitpacked::<A>(array, A::default(), ctx).await
        })
    }
}

const fn bitpacked_thread_count(output_width: usize) -> u32 {
    if output_width == 64 { 16 } else { 32 }
}

pub fn bitpacked_cuda_kernel(
    bit_width: u8,
    output_width: usize,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<CudaFunction> {
    // Load kernel function
    // bit_unpack_{bits}_{bit_width}bw_{thread_count}t
    let thread_count = bitpacked_thread_count(output_width);
    let suffixes: [&str; _] = [&format!("{bit_width}bw"), &format!("{thread_count}t")];
    ctx.load_function_with_suffixes(&format!("bit_unpack_{}", output_width), &suffixes)
}

pub fn bitpacked_cuda_launch_config(output_width: usize, len: usize) -> VortexResult<LaunchConfig> {
    let thread_count = bitpacked_thread_count(output_width);
    let num_blocks = u32::try_from(len.div_ceil(1024))?;
    Ok(LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (thread_count, 1, 1),
        shared_mem_bytes: 0,
    })
}

unsafe impl DeviceRepr for GPUPatches {}

/// Convert a PType to the corresponding ChunkOffsetType for GPU patches.
fn ptype_to_chunk_offset_type(ptype: vortex::dtype::PType) -> VortexResult<ChunkOffsetType> {
    match ptype {
        vortex::dtype::PType::U8 => Ok(ChunkOffsetType_CO_U8),
        vortex::dtype::PType::U16 => Ok(ChunkOffsetType_CO_U16),
        vortex::dtype::PType::U32 => Ok(ChunkOffsetType_CO_U32),
        vortex::dtype::PType::U64 => Ok(ChunkOffsetType_CO_U64),
        _ => vortex_bail!("Invalid PType for chunk_offsets: {:?}", ptype),
    }
}

#[instrument(skip_all)]
pub(crate) async fn decode_bitpacked<A>(
    array: BitPackedArray,
    reference: A,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<Canonical>
where
    A: BitPackedUnpack + NativePType + DeviceRepr + Send + Sync + 'static,
    A::Physical: DeviceRepr + Send + Sync + 'static,
{
    let BitPackedDataParts {
        offset,
        bit_width,
        len,
        packed,
        patches,
        validity,
    } = BitPacked::into_parts(array);

    vortex_ensure!(len > 0, "Non empty array");
    let offset = offset as usize;

    let device_input = ctx.ensure_on_device(packed).await?;

    // Get CUDA view of input
    let input_view = device_input.cuda_view::<A::Physical>()?;

    // Allocate output buffer
    let output_slice = ctx.device_alloc::<A>(len.next_multiple_of(1024))?;
    let output_buf = CudaDeviceBuffer::new(output_slice);
    let output_view = output_buf.as_view::<A>();

    let output_width = size_of::<A>() * 8;
    let cuda_function = bitpacked_cuda_kernel(bit_width, output_width, ctx)?;
    let config = bitpacked_cuda_launch_config(output_width, len)?;

    // We hold this here to keep the device buffers alive.
    let device_patches = if let Some(patches) = patches {
        Some(load_patches(&patches, ctx).await?)
    } else {
        None
    };

    #[expect(clippy::cast_possible_truncation)]
    let patches_arg = if let Some(p) = &device_patches {
        GPUPatches {
            chunk_offsets: p.chunk_offsets.cuda_device_ptr()? as _,
            chunk_offset_type: ptype_to_chunk_offset_type(p.chunk_offset_ptype)?,
            indices: p.indices.cuda_device_ptr()? as _,
            values: p.values.cuda_device_ptr()? as _,
            offset: p.offset as u32,
            offset_within_chunk: p.offset_within_chunk as u32,
            num_patches: p.num_patches as u32,
            n_chunks: p.n_chunks as u32,
        }
    } else {
        // NULL chunk_offsets signals no patches to the kernel
        GPUPatches {
            chunk_offsets: std::ptr::null_mut(),
            chunk_offset_type: ChunkOffsetType_CO_U32,
            indices: std::ptr::null_mut(),
            values: std::ptr::null_mut(),
            offset: 0,
            offset_within_chunk: 0,
            num_patches: 0,
            n_chunks: 0,
        }
    };

    ctx.launch_kernel_config(&cuda_function, config, len, |args| {
        args.arg(&input_view)
            .arg(&output_view)
            .arg(&reference)
            .arg(&patches_arg);
    })?;

    // NOTE: we must synchronize here, as the device patches are only alive for this call.
    ctx.synchronize_stream()?;

    let output_handle =
        BufferHandle::new_device(output_buf.slice_typed::<A>(offset..(offset + len)));

    // Build result with newly allocated buffer
    Ok(Canonical::Primitive(PrimitiveArray::from_buffer_handle(
        output_handle,
        A::PTYPE,
        validity,
    )))
}

#[cfg(test)]
mod tests {
    use futures::executor::block_on;
    use rstest::rstest;
    use vortex::array::IntoArray;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::assert_arrays_eq;
    use vortex::array::dtype::NativePType;
    use vortex::array::validity::Validity::NonNullable;
    use vortex::buffer::Buffer;
    use vortex::buffer::buffer;
    use vortex::encodings::fastlanes::BitPackedArrayExt;
    use vortex::error::VortexExpect;
    use vortex::session::VortexSession;

    use super::*;
    use crate::CanonicalCudaExt;
    use crate::session::CudaSession;

    #[rstest]
    #[case::u8((0u8..128u8).cycle().take(2048), 6)]
    #[case::u32((0u16..128u16).cycle().take(2048), 6)]
    #[case::u16((0u32..128u32).cycle().take(2048), 6)]
    #[case::u16((0u64..128u64).cycle().take(2048), 6)]
    #[crate::test]
    fn test_patched<T: NativePType>(
        #[case] iter: impl Iterator<Item = T>,
        #[case] bw: u8,
    ) -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let array = PrimitiveArray::new(iter.collect::<Buffer<_>>(), NonNullable);

        // Last two items should be patched
        let bp_with_patches = BitPacked::encode(&array.into_array(), bw)?;
        assert!(bp_with_patches.patches().is_some());

        let cpu_result = crate::canonicalize_cpu(bp_with_patches.clone())?.into_array();

        let gpu_result = block_on(async {
            BitPackedExecutor
                .execute(bp_with_patches.into_array(), &mut cuda_ctx)
                .await
                .vortex_expect("GPU decompression failed")
                .into_host()
                .await
                .map(|a| a.into_array())
        })?;

        assert_arrays_eq!(cpu_result, gpu_result);

        Ok(())
    }

    #[crate::test]
    fn test_patches() -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let array = PrimitiveArray::new(
            (0u16..=513).cycle().take(3072).collect::<Buffer<_>>(),
            NonNullable,
        );

        // Last two items should be patched
        let bp_with_patches = BitPacked::encode(&array.into_array(), 9)?;
        assert!(bp_with_patches.patches().is_some());

        let cpu_result = crate::canonicalize_cpu(bp_with_patches.clone())?.into_array();

        let gpu_result = block_on(async {
            BitPackedExecutor
                .execute(bp_with_patches.into_array(), &mut cuda_ctx)
                .await
                .vortex_expect("GPU decompression failed")
                .into_host()
                .await
                .map(|a| a.into_array())
        })?;

        assert_arrays_eq!(cpu_result, gpu_result);

        Ok(())
    }

    #[rstest]
    #[case::bw_1(1)]
    #[case::bw_2(2)]
    #[case::bw_3(3)]
    #[case::bw_4(4)]
    #[case::bw_5(5)]
    #[case::bw_6(6)]
    #[case::bw_7(7)]
    #[crate::test]
    fn test_cuda_bitunpack_u8(#[case] bit_width: u8) -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let max_val = (1u8 << bit_width).saturating_sub(1);

        let primitive_array = PrimitiveArray::new(
            (0u16..1024)
                .map(|i| u8::try_from(i % (max_val as u16 + 1)).vortex_expect(""))
                .collect::<Buffer<_>>(),
            NonNullable,
        );

        let bitpacked_array = BitPacked::encode(&primitive_array.into_array(), bit_width)
            .vortex_expect("operation should succeed in test");
        let cpu_result = crate::canonicalize_cpu(bitpacked_array.clone())?;

        let gpu_result = block_on(async {
            BitPackedExecutor
                .execute(bitpacked_array.into_array(), &mut cuda_ctx)
                .await
                .vortex_expect("GPU decompression failed")
                .into_host()
                .await
                .map(|a| a.into_array())
        })?;

        assert_arrays_eq!(cpu_result.into_array(), gpu_result);

        Ok(())
    }

    #[rstest]
    #[case::bw_1(1)]
    #[case::bw_2(2)]
    #[case::bw_3(3)]
    #[case::bw_4(4)]
    #[case::bw_5(5)]
    #[case::bw_6(6)]
    #[case::bw_7(7)]
    #[case::bw_8(8)]
    #[case::bw_9(9)]
    #[case::bw_10(10)]
    #[case::bw_11(11)]
    #[case::bw_12(12)]
    #[case::bw_13(13)]
    #[case::bw_14(14)]
    #[case::bw_15(15)]
    #[crate::test]
    fn test_cuda_bitunpack_u16(#[case] bit_width: u8) -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let max_val = (1u16 << bit_width).saturating_sub(1);

        let primitive_array = PrimitiveArray::new(
            (0u16..1024)
                .map(|i| i % (max_val + 1))
                .collect::<Buffer<_>>(),
            NonNullable,
        );

        let bitpacked_array = BitPacked::encode(&primitive_array.into_array(), bit_width)
            .vortex_expect("operation should succeed in test");
        let cpu_result = crate::canonicalize_cpu(bitpacked_array.clone())?;

        let gpu_result = block_on(async {
            BitPackedExecutor
                .execute(bitpacked_array.into_array(), &mut cuda_ctx)
                .await
                .vortex_expect("GPU decompression failed")
                .into_host()
                .await
                .map(|a| a.into_array())
        })?;

        assert_arrays_eq!(cpu_result.into_array(), gpu_result);

        Ok(())
    }

    #[rstest]
    #[case::bw_1(1)]
    #[case::bw_2(2)]
    #[case::bw_3(3)]
    #[case::bw_4(4)]
    #[case::bw_5(5)]
    #[case::bw_6(6)]
    #[case::bw_7(7)]
    #[case::bw_8(8)]
    #[case::bw_9(9)]
    #[case::bw_10(10)]
    #[case::bw_11(11)]
    #[case::bw_12(12)]
    #[case::bw_13(13)]
    #[case::bw_14(14)]
    #[case::bw_15(15)]
    #[case::bw_16(16)]
    #[case::bw_17(17)]
    #[case::bw_18(18)]
    #[case::bw_19(19)]
    #[case::bw_20(20)]
    #[case::bw_21(21)]
    #[case::bw_22(22)]
    #[case::bw_23(23)]
    #[case::bw_24(24)]
    #[case::bw_25(25)]
    #[case::bw_26(26)]
    #[case::bw_27(27)]
    #[case::bw_28(28)]
    #[case::bw_29(29)]
    #[case::bw_30(30)]
    #[case::bw_31(31)]
    #[crate::test]
    fn test_cuda_bitunpack_u32(#[case] bit_width: u8) -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let max_val = (1u32 << bit_width).saturating_sub(1);

        let primitive_array = PrimitiveArray::new(
            (0u32..4096)
                .map(|i| i % (max_val + 1))
                .collect::<Buffer<_>>(),
            NonNullable,
        );

        let bitpacked_array = BitPacked::encode(&primitive_array.into_array(), bit_width)
            .vortex_expect("operation should succeed in test");
        let cpu_result = crate::canonicalize_cpu(bitpacked_array.clone())?;

        let gpu_result = block_on(async {
            BitPackedExecutor
                .execute(bitpacked_array.into_array(), &mut cuda_ctx)
                .await
                .vortex_expect("GPU decompression failed")
                .into_host()
                .await
                .map(|a| a.into_array())
        })?;

        assert_arrays_eq!(cpu_result.into_array(), gpu_result);

        Ok(())
    }

    #[rstest]
    #[case::bw_1(1)]
    #[case::bw_2(2)]
    #[case::bw_3(3)]
    #[case::bw_4(4)]
    #[case::bw_5(5)]
    #[case::bw_6(6)]
    #[case::bw_7(7)]
    #[case::bw_8(8)]
    #[case::bw_9(9)]
    #[case::bw_10(10)]
    #[case::bw_11(11)]
    #[case::bw_12(12)]
    #[case::bw_13(13)]
    #[case::bw_14(14)]
    #[case::bw_15(15)]
    #[case::bw_16(16)]
    #[case::bw_17(17)]
    #[case::bw_18(18)]
    #[case::bw_19(19)]
    #[case::bw_20(20)]
    #[case::bw_21(21)]
    #[case::bw_22(22)]
    #[case::bw_23(23)]
    #[case::bw_24(24)]
    #[case::bw_25(25)]
    #[case::bw_26(26)]
    #[case::bw_27(27)]
    #[case::bw_28(28)]
    #[case::bw_29(29)]
    #[case::bw_30(30)]
    #[case::bw_31(31)]
    #[case::bw_32(32)]
    #[case::bw_33(33)]
    #[case::bw_34(34)]
    #[case::bw_35(35)]
    #[case::bw_36(36)]
    #[case::bw_37(37)]
    #[case::bw_38(38)]
    #[case::bw_39(39)]
    #[case::bw_40(40)]
    #[case::bw_41(41)]
    #[case::bw_42(42)]
    #[case::bw_43(43)]
    #[case::bw_44(44)]
    #[case::bw_45(45)]
    #[case::bw_46(46)]
    #[case::bw_47(47)]
    #[case::bw_48(48)]
    #[case::bw_49(49)]
    #[case::bw_50(50)]
    #[case::bw_51(51)]
    #[case::bw_52(52)]
    #[case::bw_53(53)]
    #[case::bw_54(54)]
    #[case::bw_55(55)]
    #[case::bw_56(56)]
    #[case::bw_57(57)]
    #[case::bw_58(58)]
    #[case::bw_59(59)]
    #[case::bw_60(60)]
    #[case::bw_61(61)]
    #[case::bw_62(62)]
    #[case::bw_63(63)]
    #[crate::test]
    fn test_cuda_bitunpack_u64(#[case] bit_width: u8) -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let max_val = (1u64 << bit_width).saturating_sub(1);

        let primitive_array = PrimitiveArray::new(
            (0u64..1024)
                .map(|i| i % (max_val + 1))
                .collect::<Buffer<_>>(),
            NonNullable,
        );

        let bitpacked_array = BitPacked::encode(&primitive_array.into_array(), bit_width)
            .vortex_expect("operation should succeed in test");
        let cpu_result = crate::canonicalize_cpu(bitpacked_array.clone())?;
        let gpu_result = block_on(async {
            BitPackedExecutor
                .execute(bitpacked_array.into_array(), &mut cuda_ctx)
                .await
                .vortex_expect("GPU decompression failed")
                .into_host()
                .await
                .map(|a| a.into_array())
        })?;

        assert_arrays_eq!(cpu_result.into_array(), gpu_result);

        Ok(())
    }

    #[crate::test]
    fn test_cuda_bitunpack_sliced() -> VortexResult<()> {
        let bit_width = 32;
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let max_val = (1u64 << bit_width).saturating_sub(1);

        let primitive_array = PrimitiveArray::new(
            (0u64..4096)
                .map(|i| i % (max_val + 1))
                .collect::<Buffer<_>>(),
            NonNullable,
        );

        let bitpacked_array = BitPacked::encode(&primitive_array.into_array(), bit_width)
            .vortex_expect("operation should succeed in test");
        let sliced_array = bitpacked_array.into_array().slice(67..3969)?;
        assert!(sliced_array.is::<BitPacked>());
        let cpu_result = crate::canonicalize_cpu(sliced_array.clone())?;
        let gpu_result = block_on(async {
            BitPackedExecutor
                .execute(sliced_array, &mut cuda_ctx)
                .await
                .vortex_expect("GPU decompression failed")
                .into_host()
                .await
                .map(|a| a.into_array())
        })?;

        assert_arrays_eq!(cpu_result.into_array(), gpu_result);

        Ok(())
    }

    /// Test slicing a bitpacked array with patches where the slice boundary
    /// falls in the middle of a chunk's patch range, creating a non-zero
    /// offset_within_chunk.
    #[crate::test]
    fn test_cuda_bitunpack_sliced_patches_offset_within_chunk() -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        // Create an array with values that will generate patches.
        // We use values 0-511 (fits in 9 bits) but include some larger values
        // that will become patches.
        let primitive_array = PrimitiveArray::new(buffer![100u8, 101, 102, 3, 4, 5], NonNullable);

        // Encode with bit width 4. First 3 elements patched, remainder will pack.
        let bitpacked_array = BitPacked::encode(&primitive_array.into_array(), 4)?;
        assert!(
            bitpacked_array.patches().is_some(),
            "Expected patches to be present"
        );

        let sliced_array = bitpacked_array.into_array().slice(2..6)?;
        assert!(sliced_array.is::<BitPacked>());

        let cpu_result = sliced_array
            .clone()
            .execute::<Canonical>(cuda_ctx.execution_ctx())?;
        let gpu_result = block_on(async {
            BitPackedExecutor
                .execute(sliced_array, &mut cuda_ctx)
                .await
                .vortex_expect("GPU decompression failed")
                .into_host()
                .await
                .map(|a| a.into_array())
        })?;

        assert_arrays_eq!(cpu_result.into_array(), gpu_result);

        Ok(())
    }

    /// Test slicing a bitpacked array multiple times, accumulating offset_within_chunk.
    #[crate::test]
    fn test_cuda_bitunpack_double_sliced_patches() -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        // Create an array with values that will generate patches.
        let mut values: Vec<u16> = Vec::with_capacity(3072);
        for i in 0u16..3072 {
            if i == 50 || i == 100 || i == 200 || i == 300 || i == 400 || i == 1100 || i == 2100 {
                values.push(600);
            } else {
                values.push(i % 512);
            }
        }

        let primitive_array =
            PrimitiveArray::new(Buffer::from_iter(values.iter().copied()), NonNullable);

        let bitpacked_array = BitPacked::encode(&primitive_array.into_array(), 9)?;
        assert!(
            bitpacked_array.patches().is_some(),
            "Expected patches to be present"
        );

        // First slice: drop the patch at index 50 from the front of chunk 0.
        let first_slice = bitpacked_array.into_array().slice(75..3000)?;
        // Second slice (relative to first): drop patch at original index 100.
        // The second slice's range is kept wide enough that num_blocks still
        // covers every chunk in the packed buffer.
        let second_slice = first_slice.slice(50..2900)?;
        assert!(second_slice.is::<BitPacked>());

        let cpu_result = second_slice
            .clone()
            .execute::<Canonical>(cuda_ctx.execution_ctx())?;
        let gpu_result = block_on(async {
            BitPackedExecutor
                .execute(second_slice, &mut cuda_ctx)
                .await
                .vortex_expect("GPU decompression failed")
                .into_host()
                .await
                .map(|a| a.into_array())
        })?;

        assert_arrays_eq!(cpu_result.into_array(), gpu_result);

        Ok(())
    }

    /// Test slicing to skip an entire chunk's worth of patches.
    #[crate::test]
    fn test_cuda_bitunpack_sliced_skip_first_chunk_patches() -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        // Create patches in first chunk only, then slice past them all.
        let mut values: Vec<u16> = Vec::with_capacity(3072);
        for i in 0u16..3072 {
            if i == 100 || i == 200 || i == 300 {
                values.push(600);
            } else if i == 1500 || i == 2500 {
                values.push(700);
            } else {
                values.push(i % 512);
            }
        }

        let primitive_array =
            PrimitiveArray::new(Buffer::from_iter(values.iter().copied()), NonNullable);

        let bitpacked_array = BitPacked::encode(&primitive_array.into_array(), 9)?;
        assert!(
            bitpacked_array.patches().is_some(),
            "Expected patches to be present"
        );

        // Slice to skip past all first chunk patches
        let sliced_array = bitpacked_array.into_array().slice(1024..3072)?;
        assert!(sliced_array.is::<BitPacked>());

        let cpu_result = sliced_array
            .clone()
            .execute::<Canonical>(cuda_ctx.execution_ctx())?;
        let gpu_result = block_on(async {
            BitPackedExecutor
                .execute(sliced_array, &mut cuda_ctx)
                .await
                .vortex_expect("GPU decompression failed")
                .into_host()
                .await
                .map(|a| a.into_array())
        })?;

        assert_arrays_eq!(cpu_result.into_array(), gpu_result);

        Ok(())
    }
}
