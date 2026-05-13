// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::ops::Range;

use async_trait::async_trait;
use cudarc::driver::CudaFunction;
use cudarc::driver::DeviceRepr;
use cudarc::driver::LaunchConfig;
use cudarc::driver::PushKernelArg;
use tracing::instrument;
use vortex::array::ArrayRef;
use vortex::array::ArrayVTable;
use vortex::array::ArrayView;
use vortex::array::Canonical;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::Slice;
use vortex::array::arrays::slice::SliceArrayExt;
use vortex::array::buffer::BufferHandle;
use vortex::array::buffer::DeviceBufferExt;
use vortex::array::match_each_integer_ptype;
use vortex::array::patches::PATCH_CHUNK_SIZE;
use vortex::dtype::NativePType;
use vortex::encodings::fastlanes::BitPacked;
use vortex::encodings::fastlanes::BitPackedArray;
use vortex::encodings::fastlanes::BitPackedArrayExt;
use vortex::encodings::fastlanes::BitPackedDataParts;
use vortex::encodings::fastlanes::unpack_iter::BitPacked as BitPackedUnpack;
use vortex::error::VortexResult;
use vortex::error::vortex_ensure;
use vortex::error::vortex_err;

use crate::CudaBufferExt;
use crate::CudaDeviceBuffer;
use crate::executor::CudaExecute;
use crate::executor::CudaExecutionCtx;
use crate::kernel::patches::build_gpu_patches;
use crate::kernel::patches::types::load_device_patches;
use crate::kernel::patches::types::slice_device_patches;

/// CUDA decoder for bit-packed arrays.
#[derive(Debug)]
pub(crate) struct BitPackedExecutor;

/// Build the packed buffer view for decoding `Slice(BitPacked)`.
///
/// Bit-unpack kernels decode full FastLanes chunks, so the packed buffer is
/// widened to chunk boundaries and `offset` is converted into the in-chunk
/// starting position. The returned logical range is passed to patch
/// materialization so exception metadata is sliced consistently.
pub(crate) fn bitpacked_slice_view(
    bp: ArrayView<'_, BitPacked>,
    offset: usize,
    len: usize,
) -> VortexResult<(BufferHandle, u16, Range<usize>)> {
    let patch_range = offset..offset + len;
    let offset_start = patch_range.start + bp.offset() as usize;
    let offset_stop = offset_start + len;
    let bitpacked_offset = offset_start % PATCH_CHUNK_SIZE;
    let block_start = offset_start - bitpacked_offset;
    let block_stop = offset_stop.div_ceil(PATCH_CHUNK_SIZE) * PATCH_CHUNK_SIZE;

    let encoded_start = (block_start / 8) * bp.bit_width() as usize;
    let encoded_stop = (block_stop / 8) * bp.bit_width() as usize;

    Ok((
        bp.packed().slice(encoded_start..encoded_stop),
        u16::try_from(bitpacked_offset)?,
        patch_range,
    ))
}

impl BitPackedExecutor {
    fn try_specialize(
        array: ArrayRef,
    ) -> VortexResult<Option<(BitPackedArray, Option<Range<usize>>)>> {
        if let Ok(array) = array.clone().try_downcast::<BitPacked>() {
            return Ok(Some((array, None)));
        }

        let Some(slice) = array.as_opt::<Slice>() else {
            return Ok(None);
        };
        let child = slice.child();
        if child.encoding_id() != BitPacked.id() {
            return Ok(None);
        }

        let bp = child.as_::<BitPacked>();
        let offset = slice.data().slice_range().start;
        let len = array.len();
        let (packed, bitpacked_offset, patch_range) = bitpacked_slice_view(bp, offset, len)?;
        let sliced = BitPacked::try_new(
            packed,
            bp.ptype(bp.dtype()),
            child.validity()?.slice(patch_range.clone())?,
            bp.patches(),
            bp.bit_width(),
            len,
            bitpacked_offset,
        )?;

        Ok(Some((sliced, Some(patch_range))))
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
        let (array, patch_range) =
            Self::try_specialize(array)?.ok_or_else(|| vortex_err!("Expected BitPackedArray"))?;
        let ptype = array.ptype(array.dtype());

        match_each_integer_ptype!(ptype, |A| {
            decode_bitpacked::<A>(array, A::default(), patch_range, ctx).await
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

#[instrument(skip_all)]
pub(crate) async fn decode_bitpacked<A>(
    array: BitPackedArray,
    reference: A,
    patch_range: Option<Range<usize>>,
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
        let mut device_patches = load_device_patches(&patches, ctx).await?;
        if let Some(range) = patch_range {
            slice_device_patches(&patches, range, &mut device_patches);
        }
        Some(device_patches)
    } else {
        None
    };

    let patches_arg = build_gpu_patches(device_patches.as_ref())?;

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
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;

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
        let bp_with_patches = BitPacked::encode(
            &array.into_array(),
            bw,
            &mut LEGACY_SESSION.create_execution_ctx(),
        )?;
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
        let bp_with_patches = BitPacked::encode(
            &array.into_array(),
            9,
            &mut LEGACY_SESSION.create_execution_ctx(),
        )?;
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

        let bitpacked_array = BitPacked::encode(
            &primitive_array.into_array(),
            bit_width,
            &mut LEGACY_SESSION.create_execution_ctx(),
        )
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

        let bitpacked_array = BitPacked::encode(
            &primitive_array.into_array(),
            bit_width,
            &mut LEGACY_SESSION.create_execution_ctx(),
        )
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

        let bitpacked_array = BitPacked::encode(
            &primitive_array.into_array(),
            bit_width,
            &mut LEGACY_SESSION.create_execution_ctx(),
        )
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

        let bitpacked_array = BitPacked::encode(
            &primitive_array.into_array(),
            bit_width,
            &mut LEGACY_SESSION.create_execution_ctx(),
        )
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

        let bitpacked_array = BitPacked::encode(
            &primitive_array.into_array(),
            bit_width,
            &mut LEGACY_SESSION.create_execution_ctx(),
        )
        .vortex_expect("operation should succeed in test");
        let sliced_array = bitpacked_array.into_array().slice(67..3969)?;
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

    #[rstest]
    #[case::direct(None, 4096, None, 0, 4096 * 9 / 8)]
    #[case::mid_chunk(Some(67..3969), 3902, Some(67..3969), 67, 4096 * 9 / 8)]
    #[case::chunk_aligned(Some(1024..3072), 2048, Some(1024..3072), 0, 2048 * 9 / 8)]
    #[case::tail_chunk(Some(3000..4096), 1096, Some(3000..4096), 952, 2048 * 9 / 8)]
    #[crate::test]
    fn test_bitunpack_try_specialize_slices(
        #[case] range: Option<Range<usize>>,
        #[case] expected_len: usize,
        #[case] expected_patch_range: Option<Range<usize>>,
        #[case] expected_offset: u16,
        #[case] expected_packed_len: usize,
    ) -> VortexResult<()> {
        let values = PrimitiveArray::new(
            (0u16..4096)
                .map(|i| if i % 1000 == 0 { 600 } else { i % 512 })
                .collect::<Buffer<_>>(),
            NonNullable,
        );
        let bitpacked = BitPacked::encode(
            &values.into_array(),
            9,
            &mut LEGACY_SESSION.create_execution_ctx(),
        )?;
        assert!(bitpacked.patches().is_some());
        let array = if let Some(range) = range {
            bitpacked.into_array().slice(range)?
        } else {
            bitpacked.into_array()
        };

        let (specialized, patch_range) =
            BitPackedExecutor::try_specialize(array)?.vortex_expect("expected BitPacked input");

        assert_eq!(specialized.len(), expected_len);
        assert_eq!(specialized.offset(), expected_offset);
        assert_eq!(specialized.packed().len(), expected_packed_len);
        assert_eq!(patch_range, expected_patch_range);

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
        let bitpacked_array = BitPacked::encode(
            &primitive_array.into_array(),
            4,
            &mut LEGACY_SESSION.create_execution_ctx(),
        )?;
        assert!(
            bitpacked_array.patches().is_some(),
            "Expected patches to be present"
        );

        let sliced_array = bitpacked_array.into_array().slice(2..6)?;

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

        let bitpacked_array = BitPacked::encode(
            &primitive_array.into_array(),
            9,
            &mut LEGACY_SESSION.create_execution_ctx(),
        )?;
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

        let bitpacked_array = BitPacked::encode(
            &primitive_array.into_array(),
            9,
            &mut LEGACY_SESSION.create_execution_ctx(),
        )?;
        assert!(
            bitpacked_array.patches().is_some(),
            "Expected patches to be present"
        );

        // Slice to skip past all first chunk patches
        let sliced_array = bitpacked_array.into_array().slice(1024..3072)?;

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
