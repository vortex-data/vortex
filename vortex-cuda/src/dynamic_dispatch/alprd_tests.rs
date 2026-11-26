// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::LazyLock;

use rstest::rstest;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::arrays::DictArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::slice::SliceKernel;
use vortex::array::validity::Validity::NonNullable;
use vortex::buffer::Buffer;
use vortex::encodings::alp::ALPRD;
use vortex::encodings::alp::ALPRDArray;
use vortex::encodings::alp::ALPRDArrayExt;
use vortex::encodings::alp::ALPRDArraySlotsExt;
use vortex::encodings::alp::RDEncoder;
use vortex::encodings::alp::RDEncoderExt;
use vortex::encodings::fastlanes::BitPacked;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::session::VortexSession;
use vortex_array::VortexSessionExecute;
use vortex_array::assert_arrays_eq;

use super::DispatchPlan;
use crate::CanonicalCudaExt;
use crate::CudaExecutionCtx;
use crate::executor::CudaDispatchMode;
use crate::hybrid_dispatch::try_gpu_dispatch;
use crate::session::CudaSession;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

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

fn encode(array: &PrimitiveArray) -> ALPRDArray {
    let encoded = RDEncoder::new(&[0.1f64, 0.2, 0.3]).encode(array.as_view());
    assert!(encoded.left_parts().is::<BitPacked>() && encoded.right_parts().is::<BitPacked>());
    encoded
}

fn fused_ctx() -> VortexResult<CudaExecutionCtx> {
    Ok(CudaSession::create_execution_ctx(&crate::cuda_session())?
        .with_dispatch_mode(CudaDispatchMode::DynDispatchOnly))
}

/// Decodes `array` with a single fused plan, failing if it can't be fused.
async fn decode_fused(array: ArrayRef, ctx: &mut CudaExecutionCtx) -> VortexResult<ArrayRef> {
    assert!(
        matches!(
            DispatchPlan::new(&array, CudaDispatchMode::DynDispatchOnly)?,
            DispatchPlan::Fused(_)
        ),
        "expected a fully fused plan"
    );
    Ok(try_gpu_dispatch(&array, ctx)
        .await?
        .into_host(ctx.execution_ctx())
        .await?
        .into_array())
}

#[rstest]
#[case::one_value(1, &[], &[])]
#[case::part_of_a_chunk(1000, &[], &[])]
#[case::exactly_one_chunk(1024, &[0, 1023], &[])]
#[case::one_past_a_chunk(1025, &[1024], &[])]
#[case::several_blocks(5000, &[5, 2047, 2048, 4999], &[])]
#[case::patch_at_every_chunk_boundary(4096, &[0, 1023, 1024, 2047, 2048, 3071, 3072, 4095], &[])]
#[case::nulls(3000, &[7], &[1, 1500, 2999])]
#[case::null_where_an_outlier_was(2000, &[10, 1500], &[10])]
#[crate::test]
async fn test_fused_alp_rd_matches_cpu_f64(
    #[case] len: usize,
    #[case] outliers: &[usize],
    #[case] nulls: &[usize],
) -> VortexResult<()> {
    let array = values(len, outliers, nulls);
    let encoded = encode(&array);
    assert_eq!(
        encoded.left_parts_patches().is_some(),
        outliers.iter().any(|row| !nulls.contains(row)),
        "the cases with outliers must exercise the patch path"
    );

    let gpu_result = decode_fused(encoded.into_array(), &mut fused_ctx()?).await?;
    assert_arrays_eq!(gpu_result, array, &mut SESSION.create_execution_ctx());
    Ok(())
}

#[crate::test]
async fn test_fused_alp_rd_matches_cpu_f32() -> VortexResult<()> {
    let array = PrimitiveArray::from_option_iter((0..3000).map(|row| match row {
        5 | 2100 => Some(3e30f32),
        1500 => None,
        _ => Some([0.1f32, 0.2, 0.3][row % 3]),
    }));
    let encoded = RDEncoder::new(&[0.1f32, 0.2, 0.3]).encode(array.as_view());
    assert!(encoded.left_parts_patches().is_some());

    let gpu_result = decode_fused(encoded.into_array(), &mut fused_ctx()?).await?;
    assert_arrays_eq!(gpu_result, array, &mut SESSION.create_execution_ctx());
    Ok(())
}

/// Ranges that start mid-chunk, across a chunk boundary, on a chunk start, and at the tail, so
/// the codes and patches carry every kind of offset into their first chunk.
#[rstest]
#[case::inside_a_chunk(5..1000)]
#[case::across_a_boundary(1000..1100)]
#[case::from_a_chunk_start(1024..2048)]
#[case::tail(2000..3000)]
#[crate::test]
async fn test_fused_alp_rd_sliced(#[case] range: Range<usize>) -> VortexResult<()> {
    let array = values(3000, &[5, 1023, 1024, 1050, 2100, 2999], &[1500]);
    let encoded = encode(&array);
    let expected = array.into_array().slice(range.clone())?;
    let mut ctx = fused_ctx()?;

    // A scan hands the GPU a lazy slice of the ALP-RD array.
    let lazy = encoded.clone().into_array().slice(range.clone())?;
    let gpu_result = decode_fused(lazy, &mut ctx).await?;
    assert_arrays_eq!(gpu_result, expected, &mut SESSION.create_execution_ctx());

    // ALP-RD's slice kernel instead gives an ALP-RD array with sliced children and patches.
    let sliced = <ALPRD as SliceKernel>::slice(
        encoded.as_view(),
        range,
        &mut SESSION.create_execution_ctx(),
    )?
    .ok_or_else(|| vortex_err!("ALP-RD always slices"))?;
    let gpu_result = decode_fused(sliced, &mut ctx).await?;
    assert_arrays_eq!(gpu_result, expected, &mut SESSION.create_execution_ctx());
    Ok(())
}

/// Dictionary values decode in an input stage, which locates rows differently from the output
/// stage.
#[crate::test]
async fn test_fused_alp_rd_dictionary_values() -> VortexResult<()> {
    let dictionary_values = values(1500, &[3, 1100], &[]);
    let codes: Buffer<u16> = (0..4000u16).map(|row| (row * 7) % 1500).collect();
    let codes = PrimitiveArray::new(codes, NonNullable);
    let expected = dictionary_values
        .clone()
        .into_array()
        .take(codes.clone().into_array())?;
    let dict = DictArray::try_new(codes.into_array(), encode(&dictionary_values).into_array())?;

    let gpu_result = decode_fused(dict.into_array(), &mut fused_ctx()?).await?;
    assert_arrays_eq!(gpu_result, expected, &mut SESSION.create_execution_ctx());
    Ok(())
}

/// The op unpacks the codes itself, so uncompressed codes fall back to the standalone kernel,
/// which must still decode them on the GPU.
#[crate::test]
async fn test_alp_rd_with_uncompressed_codes_is_not_fused() -> VortexResult<()> {
    let array = values(3000, &[5, 2100], &[1500]);
    let encoded = encode(&array);
    let left_parts = encoded
        .left_parts()
        .clone()
        .execute::<PrimitiveArray>(&mut SESSION.create_execution_ctx())?;
    let uncompressed_codes = ALPRD::try_new(
        encoded.dtype().clone(),
        left_parts.into_array(),
        encoded.left_parts_dictionary().clone(),
        encoded.right_parts().clone(),
        encoded.right_bit_width(),
        encoded.left_parts_patches(),
    )?
    .into_array();
    assert!(matches!(
        DispatchPlan::new(&uncompressed_codes, CudaDispatchMode::Auto)?,
        DispatchPlan::Unfused
    ));

    let mut ctx = CudaSession::create_execution_ctx(&crate::cuda_session())?;
    let gpu_result = try_gpu_dispatch(&uncompressed_codes, &mut ctx)
        .await?
        .into_host(ctx.execution_ctx())
        .await?
        .into_array();
    assert_arrays_eq!(gpu_result, array, &mut SESSION.create_execution_ctx());
    Ok(())
}
