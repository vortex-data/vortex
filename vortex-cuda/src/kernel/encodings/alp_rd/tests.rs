// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::LazyLock;

use rstest::rstest;
use vortex::VortexSessionDefault;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::slice::SliceKernel;
use vortex::array::assert_arrays_eq;
use vortex::array::patches::Patches;
use vortex::buffer::Buffer;
use vortex::compressor::BtrBlocksCompressorBuilder;
use vortex::encodings::alp::ALPRD;
use vortex::encodings::alp::ALPRDArrayExt;
use vortex::encodings::alp::RDEncoder;
use vortex::encodings::alp::RDEncoderExt;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::session::VortexSession;
use vortex_array::VortexSessionExecute;

use super::*;
use crate::CanonicalCudaExt;
use crate::CudaDispatchMode;
use crate::executor::CudaArrayExt;
use crate::session::CudaSession;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

/// A session with the default encodings and editions, which the compressor needs to pick schemes.
static DEFAULT_SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::default);

/// A value whose left part isn't in a dictionary built from `0.1`, so it becomes a patch.
const OUTLIER: f64 = 3e100;

/// `len` values: `0.1`, `0.2` or `0.3` by row, with `OUTLIER` at each of `outliers` and nulls at
/// each of `nulls`.
fn values(len: usize, outliers: &[usize], nulls: &[usize]) -> PrimitiveArray {
    PrimitiveArray::from_option_iter((0..len).map(|row| {
        if nulls.contains(&row) {
            None
        } else if outliers.contains(&row) {
            Some(OUTLIER)
        } else {
            Some([0.1, 0.2, 0.3][row % 3])
        }
    }))
}

/// Encodes `array` with ALP-RD, sampling the dictionary from its first few values.
fn encode(array: &PrimitiveArray) -> ALPRDArray {
    RDEncoder::new(&[0.1f64, 0.2, 0.3]).encode(array.as_view())
}

/// Decodes `array` with the ALP-RD kernel directly, so nothing can fall back to the CPU.
async fn decode_on_gpu(array: ArrayRef, cuda_ctx: &mut CudaExecutionCtx) -> VortexResult<ArrayRef> {
    Ok(ALPRDExecutor
        .execute(array, cuda_ctx)
        .await?
        .into_host(cuda_ctx.execution_ctx())
        .await?
        .into_array())
}

#[rstest]
#[case::one_value(1, &[], &[])]
#[case::part_of_a_chunk(1000, &[], &[])]
#[case::exactly_one_chunk(1024, &[0, 1023], &[])]
#[case::one_past_a_chunk(1025, &[1024], &[])]
#[case::patches_skip_a_chunk(3000, &[5, 2100, 2999], &[])]
#[case::patch_at_every_chunk_boundary(4096, &[0, 1023, 1024, 2047, 2048, 3071, 3072, 4095], &[])]
#[case::nulls(3000, &[7], &[1, 1500, 2999])]
#[case::null_where_an_outlier_was(2000, &[10, 1500], &[10])]
#[crate::test]
async fn test_cuda_alp_rd_matches_cpu_f64(
    #[case] len: usize,
    #[case] outliers: &[usize],
    #[case] nulls: &[usize],
) -> VortexResult<()> {
    let mut cuda_ctx = CudaSession::create_execution_ctx(&crate::cuda_session())
        .vortex_expect("failed to create execution context");
    let array = values(len, outliers, nulls);
    let encoded = encode(&array);
    assert_eq!(
        encoded.left_parts_patches().is_some(),
        outliers.iter().any(|row| !nulls.contains(row)),
        "the cases with outliers must exercise the patch path"
    );

    let gpu_result = decode_on_gpu(encoded.into_array(), &mut cuda_ctx).await?;
    assert_arrays_eq!(gpu_result, array, &mut SESSION.create_execution_ctx());
    Ok(())
}

#[crate::test]
async fn test_cuda_alp_rd_matches_cpu_f32() -> VortexResult<()> {
    let mut cuda_ctx = CudaSession::create_execution_ctx(&crate::cuda_session())
        .vortex_expect("failed to create execution context");
    let array = PrimitiveArray::from_option_iter((0..3000).map(|row| match row {
        5 | 2100 => Some(3e30f32),
        1500 => None,
        _ => Some([0.1f32, 0.2, 0.3][row % 3]),
    }));
    let encoded = RDEncoder::new(&[0.1f32, 0.2, 0.3]).encode(array.as_view());
    assert!(encoded.left_parts_patches().is_some());

    let gpu_result = decode_on_gpu(encoded.into_array(), &mut cuda_ctx).await?;
    assert_arrays_eq!(gpu_result, array, &mut SESSION.create_execution_ctx());
    Ok(())
}

/// Ranges that start mid-chunk, across a chunk boundary, on a chunk start, and at the tail, so
/// the sliced patches carry every kind of offset into their first chunk.
#[rstest]
#[case::inside_a_chunk(5..1000)]
#[case::across_a_boundary(1000..1100)]
#[case::from_a_chunk_start(1024..2048)]
#[case::tail(2000..3000)]
#[crate::test]
async fn test_cuda_alp_rd_sliced(#[case] range: Range<usize>) -> VortexResult<()> {
    let mut cuda_ctx = CudaSession::create_execution_ctx(&crate::cuda_session())
        .vortex_expect("failed to create execution context");
    let array = values(3000, &[5, 1023, 1024, 1050, 2100, 2999], &[1500]);
    let encoded = encode(&array);
    let expected = array.into_array().slice(range.clone())?;

    // Slicing is lazy, so ask ALP-RD's own slice kernel for the sliced ALP-RD the GPU decodes.
    let sliced = slice_alp_rd(&encoded, range.clone())?;
    let gpu_result = decode_on_gpu(sliced.into_array(), &mut cuda_ctx).await?;
    assert_arrays_eq!(gpu_result, expected, &mut SESSION.create_execution_ctx());

    // A scan hands the GPU the lazy slice instead, which must still decode there.
    let lazy = encoded.into_array().slice(range)?;
    let decoded = lazy.execute_cuda(&mut cuda_ctx).await?;
    assert!(
        !decoded
            .clone()
            .into_primitive()
            .buffer_handle()
            .is_on_host(),
        "a sliced ALP-RD array should decode on the GPU"
    );
    let gpu_result = decoded
        .into_host(cuda_ctx.execution_ctx())
        .await?
        .into_array();
    assert_arrays_eq!(gpu_result, expected, &mut SESSION.create_execution_ctx());
    Ok(())
}

#[rstest]
#[case::auto(CudaDispatchMode::Auto)]
#[case::standalone(CudaDispatchMode::StandaloneOnly)]
#[crate::test]
async fn test_cuda_alp_rd_is_dispatched_to_the_gpu(
    #[case] mode: CudaDispatchMode,
) -> VortexResult<()> {
    let mut cuda_ctx = CudaSession::create_execution_ctx(&crate::cuda_session())
        .vortex_expect("failed to create execution context")
        .with_dispatch_mode(mode);
    let array = values(3000, &[5, 2100], &[1500]);

    let decoded = encode(&array)
        .into_array()
        .execute_cuda(&mut cuda_ctx)
        .await?;
    // A CPU fallback would return host buffers, so a device result proves the kernel ran.
    assert!(
        !decoded
            .clone()
            .into_primitive()
            .buffer_handle()
            .is_on_host(),
        "ALP-RD should decode on the GPU"
    );
    let gpu_result = decoded
        .into_host(cuda_ctx.execution_ctx())
        .await?
        .into_array();
    assert_arrays_eq!(gpu_result, array, &mut SESSION.create_execution_ctx());
    Ok(())
}

#[crate::test]
async fn test_cuda_preset_writes_alp_rd_for_full_precision_floats() -> VortexResult<()> {
    let mut cuda_ctx = CudaSession::create_execution_ctx(&crate::cuda_session())
        .vortex_expect("failed to create execution context");
    // Full-precision values, which plain ALP can't encode, so the CUDA preset picks ALP-RD.
    let array =
        PrimitiveArray::from_iter((0..10_000u32).map(|row| f64::from(row).sqrt() * 1.0e-3 + 0.1))
            .into_array();
    let compressed = BtrBlocksCompressorBuilder::from_session(&DEFAULT_SESSION)
        .only_cuda_compatible()
        .build()
        .compress(&array, &mut DEFAULT_SESSION.create_execution_ctx())?;
    assert!(
        compressed.is::<ALPRD>(),
        "expected ALP-RD, got {}",
        compressed.encoding_id()
    );

    let gpu_result = compressed
        .execute_cuda(&mut cuda_ctx)
        .await?
        .into_host(cuda_ctx.execution_ctx())
        .await?
        .into_array();
    assert_arrays_eq!(gpu_result, array, &mut SESSION.create_execution_ctx());
    Ok(())
}

/// Slices `encoded` with ALP-RD's own slice kernel, which the GPU decodes without a slice wrapper.
fn slice_alp_rd(encoded: &ALPRDArray, range: Range<usize>) -> VortexResult<ALPRDArray> {
    <ALPRD as SliceKernel>::slice(
        encoded.as_view(),
        range,
        &mut SESSION.create_execution_ctx(),
    )?
    .vortex_expect("ALP-RD always slices itself")
    .try_downcast::<ALPRD>()
    .map_err(|array| {
        vortex_err!(
            "expected a sliced ALP-RD array, got {}",
            array.encoding_id()
        )
    })
}

#[crate::test]
async fn test_cuda_alp_rd_slice_of_a_slice() -> VortexResult<()> {
    let mut cuda_ctx = CudaSession::create_execution_ctx(&crate::cuda_session())
        .vortex_expect("failed to create execution context");
    let array = values(3000, &[5, 1023, 1024, 1050, 2100, 2999], &[1500]);
    // Rows 400..1500 of rows 700..2900 are rows 1100..2200 of the original.
    let sliced = slice_alp_rd(&slice_alp_rd(&encode(&array), 700..2900)?, 400..1500)?;

    let gpu_result = decode_on_gpu(sliced.into_array(), &mut cuda_ctx).await?;
    assert_arrays_eq!(
        gpu_result,
        array.into_array().slice(1100..2200)?,
        &mut SESSION.create_execution_ctx()
    );
    Ok(())
}

#[crate::test]
async fn test_cuda_alp_rd_sliced_f32() -> VortexResult<()> {
    let mut cuda_ctx = CudaSession::create_execution_ctx(&crate::cuda_session())
        .vortex_expect("failed to create execution context");
    let array = PrimitiveArray::from_option_iter((0..3000).map(|row| match row {
        5 | 1050 | 2100 => Some(3e30f32),
        1500 => None,
        _ => Some([0.1f32, 0.2, 0.3][row % 3]),
    }));
    let encoded = RDEncoder::new(&[0.1f32, 0.2, 0.3]).encode(array.as_view());
    let sliced = slice_alp_rd(&encoded, 1000..2200)?;

    let gpu_result = decode_on_gpu(sliced.into_array(), &mut cuda_ctx).await?;
    assert_arrays_eq!(
        gpu_result,
        array.into_array().slice(1000..2200)?,
        &mut SESSION.create_execution_ctx()
    );
    Ok(())
}

/// Rebuilds `encoded` with patches that have no chunk offsets, as files written before #11644
/// store them.
fn without_chunk_offsets(encoded: &ALPRDArray) -> VortexResult<ALPRDArray> {
    let patches = encoded
        .left_parts_patches()
        .vortex_expect("the fixture has outliers");
    let patches = Patches::new(
        patches.array_len(),
        patches.offset(),
        patches.indices().clone(),
        patches.values().clone(),
        None,
        &mut SESSION.create_execution_ctx(),
    )?;
    ALPRD::try_new(
        encoded.dtype().clone(),
        encoded.left_parts().clone(),
        encoded.left_parts_dictionary().clone(),
        encoded.right_parts().clone(),
        encoded.right_bit_width(),
        Some(patches),
    )
}

#[crate::test]
async fn test_cuda_alp_rd_without_chunk_offsets_decodes_on_the_cpu() -> VortexResult<()> {
    let mut cuda_ctx = CudaSession::create_execution_ctx(&crate::cuda_session())
        .vortex_expect("failed to create execution context");
    let array = values(3000, &[5, 2100], &[1500]);
    let old_file_array = without_chunk_offsets(&encode(&array))?;

    assert!(
        decode_on_gpu(old_file_array.clone().into_array(), &mut cuda_ctx)
            .await
            .is_err(),
        "the kernel needs chunk offsets"
    );
    let result = old_file_array
        .into_array()
        .execute_cuda(&mut cuda_ctx)
        .await?
        .into_host(cuda_ctx.execution_ctx())
        .await?
        .into_array();
    assert_arrays_eq!(result, array, &mut SESSION.create_execution_ctx());
    Ok(())
}

/// Encodes `array`, then replaces the left-parts code at `row` with one outside the dictionary.
fn with_code_outside_the_dictionary(
    array: &PrimitiveArray,
    row: usize,
) -> VortexResult<ALPRDArray> {
    let encoded = encode(array);
    let left_parts = encoded
        .left_parts()
        .clone()
        .execute::<PrimitiveArray>(&mut SESSION.create_execution_ctx())?;
    let mut codes = left_parts.as_slice::<u16>().to_vec();
    codes[row] = u16::try_from(encoded.left_parts_dictionary().len())?;
    ALPRD::try_new(
        encoded.dtype().clone(),
        PrimitiveArray::new(Buffer::from(codes), left_parts.validity()?).into_array(),
        encoded.left_parts_dictionary().clone(),
        encoded.right_parts().clone(),
        encoded.right_bit_width(),
        encoded.left_parts_patches(),
    )
}

/// Like the CPU decoder, the kernel accepts a code outside the dictionary only in a null row. The
/// slice moves the row's validity bit off a byte boundary.
#[rstest]
#[crate::test]
async fn test_cuda_alp_rd_code_outside_the_dictionary(
    #[values(false, true)] in_null_row: bool,
    #[values(None, Some(5..90))] slice: Option<Range<usize>>,
) -> VortexResult<()> {
    let mut cuda_ctx = CudaSession::create_execution_ctx(&crate::cuda_session())
        .vortex_expect("failed to create execution context");
    let nulls: &[usize] = if in_null_row { &[7] } else { &[] };
    let array = values(100, &[], nulls);
    let mut corrupt = with_code_outside_the_dictionary(&array, 7)?;
    let mut expected = array.into_array();
    if let Some(range) = slice {
        corrupt = slice_alp_rd(&corrupt, range.clone())?;
        expected = expected.slice(range)?;
    }

    let gpu_result = decode_on_gpu(corrupt.clone().into_array(), &mut cuda_ctx).await;
    let cpu_result = corrupt
        .into_array()
        .execute::<PrimitiveArray>(&mut SESSION.create_execution_ctx());
    if in_null_row {
        assert_arrays_eq!(gpu_result?, expected, &mut SESSION.create_execution_ctx());
        assert_arrays_eq!(cpu_result?, expected, &mut SESSION.create_execution_ctx());
    } else {
        assert!(
            gpu_result.is_err(),
            "the kernel must reject a valid row's bad code"
        );
        assert!(cpu_result.is_err(), "the CPU decoder rejects it too");
    }
    Ok(())
}
