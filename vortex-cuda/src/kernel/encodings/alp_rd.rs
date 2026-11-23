// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;

use async_trait::async_trait;
use tracing::instrument;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::arrays::Bool;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::bool::BoolDataParts;
use vortex::array::arrays::primitive::PrimitiveDataParts;
use vortex::array::buffer::BufferHandle;
use vortex::array::buffer::DeviceBufferExt;
use vortex::array::validity::Validity;
use vortex::dtype::NativePType;
use vortex::dtype::PType;
use vortex::encodings::alp::ALPRD;
use vortex::encodings::alp::ALPRDArray;
use vortex::encodings::alp::ALPRDArrayExt;
use vortex::encodings::alp::ALPRDArraySlotsExt;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_ensure;
use vortex::error::vortex_err;
use vortex_cuda_driver::DeviceRepr;
use vortex_cuda_driver::LaunchConfig;
use vortex_cuda_driver::PushKernelArg;

use crate::CudaBufferExt;
use crate::CudaDeviceBuffer;
use crate::executor::CudaArrayExt;
use crate::executor::CudaExecute;
use crate::executor::CudaExecutionCtx;
use crate::executor::execute_validity_cuda;
use crate::kernel::patches::build_gpu_patches;
use crate::kernel::patches::types::load_device_patches;

/// CUDA decoder for ALP-RD (ALP for "real doubles").
#[derive(Debug)]
pub(crate) struct ALPRDExecutor;

#[async_trait]
impl CudaExecute for ALPRDExecutor {
    #[instrument(level = "trace", skip_all, fields(executor = ?self))]
    async fn execute(
        &self,
        array: ArrayRef,
        ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<Canonical> {
        let array = array
            .try_downcast::<ALPRD>()
            .map_err(|_| vortex_err!("Expected ALPRDArray"))?;

        match array.dtype().as_ptype() {
            PType::F32 => decode_alp_rd::<u32>(array, PType::F32, ctx).await,
            PType::F64 => decode_alp_rd::<u64>(array, PType::F64, ctx).await,
            ptype => vortex_bail!("ALP-RD cannot decode {ptype}"),
        }
    }
}

/// Threads per block. 32 threads × 32 elements cover one 1024-element chunk.
const ALP_RD_THREADS_PER_BLOCK: u32 = 32;

/// Entries in the kernel's `AlpRdDictionary`, the `alp` crate's `MAX_DICT_SIZE`.
const ALP_RD_DICTIONARY_CAPACITY: usize = 8;

/// Decodes `array` into floats of `float_ptype`, whose bits have the unsigned type `U`.
#[instrument(skip_all)]
async fn decode_alp_rd<U>(
    array: ALPRDArray,
    float_ptype: PType,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<Canonical>
where
    U: NativePType + DeviceRepr + Send + Sync + 'static,
{
    let array_len = array.len();
    vortex_ensure!(array_len > 0, "ALP-RD array must not be empty");

    // ALP-RD files written before #11644 have patches without chunk offsets, which the patch
    // cursor needs. Reject them before any GPU work, so the CPU decodes them if they're on the
    // host.
    let left_parts_patches = array.left_parts_patches();
    if left_parts_patches
        .as_ref()
        .is_some_and(|patches| patches.chunk_offsets().is_none())
    {
        vortex_bail!("ALP-RD patches without chunk offsets cannot be decoded on the GPU");
    }

    let left_parts = array.left_parts().clone().execute_cuda(ctx).await?;
    let PrimitiveDataParts {
        buffer: left_buffer,
        validity,
        ..
    } = left_parts.into_primitive().into_data_parts();
    let right_parts = array.right_parts().clone().execute_cuda(ctx).await?;
    let PrimitiveDataParts {
        buffer: right_buffer,
        ..
    } = right_parts.into_primitive().into_data_parts();

    // Null rows may hold codes outside the dictionary, so the kernel checks codes against the
    // null mask. A null pointer stands for all-valid.
    let validity = execute_validity_cuda(validity, array_len, ctx).await?;
    let device_validity = match &validity {
        Validity::Array(validity_array) => {
            let bool_array = validity_array
                .clone()
                .try_downcast::<Bool>()
                .map_err(|array| {
                    vortex_err!("expected boolean validity, got {}", array.encoding_id())
                })?;
            let BoolDataParts { bits, meta } = bool_array.into_data().into_parts(array_len);
            Some((bits, meta.offset()))
        }
        _ => None,
    };
    let validity_ptr = match &device_validity {
        Some((bits, _)) => bits.cuda_device_ptr(ctx.stream())?,
        None => 0,
    };
    let validity_offset = device_validity
        .as_ref()
        .map_or(0, |(_, offset)| *offset as u64);

    let device_left = ctx.ensure_on_device(left_buffer)?;
    let left_view = device_left.cuda_view::<u16>()?;
    let device_right = ctx.ensure_on_device(right_buffer)?;
    let right_view = device_right.cuda_view::<U>()?;

    let dictionary = array.left_parts_dictionary();
    vortex_ensure!(
        dictionary.len() <= ALP_RD_DICTIONARY_CAPACITY,
        "ALP-RD dictionary holds {} entries, more than {ALP_RD_DICTIONARY_CAPACITY}",
        dictionary.len()
    );
    // Entries past `dictionary_len` are never read, because the kernel checks codes against it.
    let mut dictionary_entries = [0u16; ALP_RD_DICTIONARY_CAPACITY];
    dictionary_entries[..dictionary.len()].copy_from_slice(dictionary.as_slice());
    let dictionary_len = u32::try_from(dictionary.len())?;

    let device_patches = match left_parts_patches {
        Some(patches) => Some(load_device_patches(&patches, ctx).await?),
        None => None,
    };
    let patches_arg = build_gpu_patches(device_patches.as_ref(), ctx.stream())?;

    // Patch positions count from the 1024-row chunk the slice starts in, so the kernel decodes
    // chunk-aligned rows and the first `slice_shift` rows are trimmed off below.
    let slice_shift = device_patches
        .as_ref()
        .map_or(0, |patches| patches.offset % 1024);
    let aligned_len = array_len + slice_shift;
    let mut output_slice = ctx.device_alloc::<U>(aligned_len.next_multiple_of(1024))?;
    let mut invalid_code = ctx.device_alloc::<u32>(1)?;
    ctx.stream()
        .memset_zeros(&mut invalid_code)
        .map_err(|e| vortex_err!("Failed to zero the ALP-RD invalid code flag: {e}"))?;

    let cuda_function =
        ctx.load_function_with_suffixes("alp_rd", &[U::PTYPE.to_string().as_str(), "32t"])?;

    let num_blocks = u32::try_from(aligned_len.div_ceil(1024))?;
    let config = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (ALP_RD_THREADS_PER_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };

    let right_bit_width = u32::from(array.right_bit_width());
    let array_len_u64 = array_len as u64;
    let slice_shift_u32 = u32::try_from(slice_shift)?;
    ctx.launch_kernel_config(&cuda_function, config, array_len, |args| {
        args.arg(&left_view)
            .arg(&right_view)
            .arg(&mut output_slice)
            .arg(&dictionary_entries)
            .arg(&dictionary_len)
            .arg(&right_bit_width)
            .arg(&array_len_u64)
            .arg(&slice_shift_u32)
            .arg(&validity_ptr)
            .arg(&validity_offset)
            .arg(&mut invalid_code)
            .arg(&patches_arg);
    })?;

    // Synchronize so the device patches and validity stay alive for the kernel.
    ctx.synchronize_stream()?;
    drop(device_patches);
    drop(device_validity);

    let invalid_code = ctx
        .stream()
        .clone_dtoh(&invalid_code)
        .map_err(|e| vortex_err!("Failed to copy the ALP-RD invalid code flag to the host: {e}"))?;
    // With every row null, no code can be invalid.
    vortex_ensure!(
        invalid_code[0] == 0 || matches!(validity, Validity::AllInvalid),
        "ALP-RD left-parts code in a valid row is not in the {dictionary_len}-entry dictionary"
    );

    // The kernel writes the float bits, so the buffer is read back as floats of `float_ptype`.
    let output_buf = CudaDeviceBuffer::new(output_slice)?;
    let output_handle =
        BufferHandle::new_device(output_buf.slice_typed::<U>(slice_shift..aligned_len));
    Ok(Canonical::Primitive(PrimitiveArray::from_buffer_handle(
        output_handle,
        float_ptype,
        validity,
    )))
}

#[cfg(test)]
mod tests;
