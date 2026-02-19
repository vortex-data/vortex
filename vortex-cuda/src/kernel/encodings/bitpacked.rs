// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::sync::Arc;

use async_trait::async_trait;
use cudarc::driver::CudaFunction;
use cudarc::driver::DeviceRepr;
use cudarc::driver::LaunchConfig;
use cudarc::driver::PushKernelArg;
use tracing::instrument;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::buffer::DeviceBufferExt;
use vortex_array::dtype::NativePType;
use vortex_array::match_each_integer_ptype;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_cuda_macros::cuda_tests;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_fastlanes::BitPackedArray;
use vortex_fastlanes::BitPackedArrayParts;
use vortex_fastlanes::BitPackedVTable;
use vortex_fastlanes::unpack_iter::BitPacked;

use crate::CudaBufferExt;
use crate::CudaDeviceBuffer;
use crate::executor::CudaExecute;
use crate::executor::CudaExecutionCtx;
use crate::kernel::patches::execute_patches;

/// CUDA decoder for bit-packed arrays.
#[derive(Debug)]
pub(crate) struct BitPackedExecutor;

impl BitPackedExecutor {
    fn try_specialize(array: ArrayRef) -> Option<BitPackedArray> {
        array.try_into::<BitPackedVTable>().ok()
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

        match_each_integer_ptype!(array.ptype(), |A| {
            decode_bitpacked::<A>(array, ctx).await
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
    ctx.load_function(&format!("bit_unpack_{}", output_width), &suffixes)
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

pub(crate) async fn decode_bitpacked<A>(
    array: BitPackedArray,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<Canonical>
where
    A: BitPacked + NativePType + DeviceRepr + Send + Sync + 'static,
    A::Physical: DeviceRepr + Send + Sync + 'static,
{
    let BitPackedArrayParts {
        offset,
        bit_width,
        len,
        packed,
        patches,
        validity,
    } = array.into_parts();

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

    ctx.launch_kernel_config(&cuda_function, config, len, |args| {
        args.arg(&input_view).arg(&output_view);
    })?;

    let output_handle = match patches {
        None => BufferHandle::new_device(output_buf.slice_typed::<A>(offset..(offset + len))),
        Some(p) => {
            let output_buf = output_buf.slice_typed::<A>(offset..(offset + len));
            let buf = output_buf
                .as_any()
                .downcast_ref::<CudaDeviceBuffer>()
                .vortex_expect("we created this as CudaDeviceBuffer")
                .clone();

            let patched_buf = match_each_unsigned_integer_ptype!(p.indices_ptype()?, |I| {
                execute_patches::<A, I>(p, buf, ctx).await?
            });

            BufferHandle::new_device(Arc::new(patched_buf))
        }
    };

    // Build result with newly allocated buffer
    Ok(Canonical::Primitive(PrimitiveArray::from_buffer_handle(
        output_handle,
        A::PTYPE,
        validity,
    )))
}

#[cuda_tests]
mod tests {
    use futures::executor::block_on;
    use rstest::rstest;
    use vortex_array::ExecutionCtx;
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::session::ArraySession;
    use vortex_array::validity::Validity::NonNullable;
    use vortex_array::vtable::VTable;
    use vortex_buffer::Buffer;
    use vortex_error::VortexExpect;
    use vortex_session::VortexSession;

    use super::*;
    use crate::CanonicalCudaExt;
    use crate::session::CudaSession;

    #[test]
    fn test_patches() -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let array = PrimitiveArray::new((0u16..=513).collect::<Buffer<_>>(), NonNullable);

        // Last two items should be patched
        let bp_with_patches = BitPackedArray::encode(array.as_ref(), 9)?;
        assert!(bp_with_patches.patches().is_some());

        let cpu_result = bp_with_patches.to_canonical()?.into_array();

        let gpu_result = block_on(async {
            BitPackedExecutor
                .execute(bp_with_patches.to_array(), &mut cuda_ctx)
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

        let bitpacked_array = BitPackedArray::encode(primitive_array.as_ref(), bit_width)
            .vortex_expect("operation should succeed in test");
        let cpu_result = bitpacked_array.to_canonical()?;

        let gpu_result = block_on(async {
            BitPackedExecutor
                .execute(bitpacked_array.to_array(), &mut cuda_ctx)
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

        let bitpacked_array = BitPackedArray::encode(primitive_array.as_ref(), bit_width)
            .vortex_expect("operation should succeed in test");
        let cpu_result = bitpacked_array.to_canonical()?;

        let gpu_result = block_on(async {
            BitPackedExecutor
                .execute(bitpacked_array.to_array(), &mut cuda_ctx)
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

        let bitpacked_array = BitPackedArray::encode(primitive_array.as_ref(), bit_width)
            .vortex_expect("operation should succeed in test");
        let cpu_result = bitpacked_array.to_canonical()?;

        let gpu_result = block_on(async {
            BitPackedExecutor
                .execute(bitpacked_array.to_array(), &mut cuda_ctx)
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

        let bitpacked_array = BitPackedArray::encode(primitive_array.as_ref(), bit_width)
            .vortex_expect("operation should succeed in test");
        let cpu_result = bitpacked_array.to_canonical()?;
        let gpu_result = block_on(async {
            BitPackedExecutor
                .execute(bitpacked_array.to_array(), &mut cuda_ctx)
                .await
                .vortex_expect("GPU decompression failed")
                .into_host()
                .await
                .map(|a| a.into_array())
        })?;

        assert_arrays_eq!(cpu_result.into_array(), gpu_result);

        Ok(())
    }

    #[test]
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

        let bitpacked_array = BitPackedArray::encode(primitive_array.as_ref(), bit_width)
            .vortex_expect("operation should succeed in test");
        let slice_ref = bitpacked_array.clone().into_array().slice(67..3969)?;
        let mut exec_ctx = ExecutionCtx::new(VortexSession::empty().with::<ArraySession>());
        let sliced_array = <BitPackedVTable as VTable>::execute_parent(
            &bitpacked_array,
            &slice_ref,
            0,
            &mut exec_ctx,
        )?
        .expect("expected slice kernel to execute");
        let cpu_result = sliced_array.to_canonical()?;
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
