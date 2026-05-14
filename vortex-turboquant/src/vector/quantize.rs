// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Core TurboQuant quantization helpers.
//!
//! Quantization consumes the TurboQuant-local normalized `Vector` child. Valid rows are transformed
//! and mapped to scalar centroid indices. Invalid rows remain in the full-length output but are
//! skipped: their physical code bytes are placeholders guarded by the `codes` row validity.
//!
//! This matters because TurboQuant's scalar codebook is optimized for coordinates of transformed
//! unit-norm vectors. The codebook does not generally contain an exact zero centroid, and a
//! physical code byte of `0` means "centroid 0", not "zero coordinate". Null vectors therefore
//! should not be converted to zero vectors and fed through the quantizer.

use half::f16;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::fixed_size_list::FixedSizeListArrayExt;
use vortex_array::dtype::PType;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_mask::Mask;

use super::tq_padded_dim;
use crate::TurboQuantConfig;
use crate::centroids::compute_centroid_boundaries;
use crate::centroids::compute_or_get_centroids;
use crate::centroids::find_nearest_centroid;
use crate::sorf::SorfMatrix;

/// Shared intermediate results from the quantization loop.
pub(crate) struct QuantizationResult {
    pub(crate) all_indices: Buffer<u8>,
    pub(crate) padded_dim: usize,
}

pub(crate) fn empty_quantization(padded_dim: usize) -> QuantizationResult {
    QuantizationResult {
        all_indices: Buffer::empty(),
        padded_dim,
    }
}

/// Core quantization: transform and quantize already-normalized rows.
///
/// # Safety
///
/// The input `fsl` must contain unit-norm vectors (already L2-normalized) for every valid row.
/// Invalid rows are left row-aligned in the output but are not transformed or quantized. The
/// transform and centroid lookup happen in f32.
pub(crate) unsafe fn turboquant_quantize_core(
    fsl: &FixedSizeListArray,
    config: &TurboQuantConfig,
    ctx: &mut ExecutionCtx,
) -> VortexResult<QuantizationResult> {
    let dimension = fsl.list_size();
    let num_vectors = fsl.len();
    let padded_dim = tq_padded_dim(dimension)?;

    let sorf_transform =
        SorfMatrix::try_new(padded_dim, config.num_rounds() as usize, config.seed())?;
    debug_assert_eq!(sorf_transform.padded_dim(), padded_dim);
    let padded_dim_u32 = u32::try_from(padded_dim)
        .map_err(|_| vortex_err!("TurboQuant padded dimension does not fit u32"))?;

    let elements_prim: PrimitiveArray = fsl.elements().clone().execute(ctx)?;
    let f32_elements = cast_to_f32(elements_prim)?;
    let validity = fsl.validity()?;
    let mask = validity.execute_mask(num_vectors, ctx)?;

    let centroids = compute_or_get_centroids(padded_dim_u32, config.bit_width())?;
    let boundaries = compute_centroid_boundaries(&centroids);

    let codes_len = num_vectors
        .checked_mul(padded_dim)
        .ok_or_else(|| vortex_err!("TurboQuant codes length overflow"))?;
    let mut all_indices = BufferMut::<u8>::with_capacity(codes_len);

    let mut padded = vec![0.0f32; padded_dim];
    let mut transformed = vec![0.0f32; padded_dim];

    // Pad, SORF-transform, and quantize a single row, pushing `padded_dim` codes into
    // `all_indices`. Captures the read-only inputs and the scratch buffers so each call site
    // only needs to pass `all_indices` and the row index.
    //
    // NB: `all_indices` cannot be captured here: the `Values` arm interleaves the closure call
    // with direct `all_indices.push_n_unchecked` calls.
    let f32_slice = f32_elements.as_slice();
    let dimension = dimension as usize;
    let mut quantize_row = |all_indices: &mut BufferMut<u8>, row: usize| {
        // Reuse `padded` and `transformed` from the outer scope.
        padded[..dimension].copy_from_slice(&f32_slice[row * dimension..][..dimension]);
        padded[dimension..].fill(0.0);
        sorf_transform.transform(&padded, &mut transformed);

        for &value in &transformed {
            // SAFETY: total pushes across all match arms equal `codes_len`.
            unsafe { all_indices.push_unchecked(find_nearest_centroid(value, &boundaries)) };
        }
    };

    // The total number of pushes is always exactly `num_vectors * padded_dim == codes_len`
    // across every arm below, which is the invariant the per-row `unsafe` blocks rely on.
    match &mask {
        Mask::AllFalse(_) => {
            // Every row is invalid: bulk-fill placeholder zero codes.
            //
            // SAFETY: `all_indices` was allocated with capacity `codes_len`, and this push
            // writes exactly `codes_len` zero codes.
            unsafe { all_indices.push_n_unchecked(0, codes_len) };
        }
        Mask::AllTrue(_) => {
            for row in 0..num_vectors {
                quantize_row(&mut all_indices, row);
            }
        }
        Mask::Values(values_mask) => {
            let mut cursor = 0;

            for &(start, end) in values_mask.slices() {
                if start > cursor {
                    // SAFETY: total pushes across all arms equal `codes_len`.
                    unsafe { all_indices.push_n_unchecked(0, (start - cursor) * padded_dim) };
                }

                for row in start..end {
                    quantize_row(&mut all_indices, row);
                }

                cursor = end;
            }

            if cursor < num_vectors {
                // SAFETY: total pushes across all arms equal `codes_len`.
                unsafe { all_indices.push_n_unchecked(0, (num_vectors - cursor) * padded_dim) };
            }
        }
    }

    Ok(QuantizationResult {
        all_indices: all_indices.freeze(),
        padded_dim,
    })
}

/// Cast a float [`PrimitiveArray`] to a `Buffer<f32>`.
///
/// Several operations in this crate (SORF transform, TurboQuant quantization) work exclusively
/// in f32. This function handles the cast from any float ptype:
///
/// - f16: losslessly widened to f32.
/// - f32: zero-copy buffer extraction.
/// - f64: truncated to f32 precision. Values outside f32 range become +/- infinity. This is
///   acceptable because callers of this function operate in f32 and document this constraint.
fn cast_to_f32(prim: PrimitiveArray) -> VortexResult<Buffer<f32>> {
    match prim.ptype() {
        PType::F16 => Ok(prim
            .as_slice::<f16>()
            .iter()
            .map(|&v| f32::from(v))
            .collect()),
        PType::F32 => Ok(prim.into_buffer()),
        PType::F64 => Ok(prim
            .as_slice::<f64>()
            .iter()
            .map(|&v| {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "f64 values outside f32 range become infinity, matching tensor TQ"
                )]
                let v = v as f32;
                v
            })
            .collect()),
        other => vortex_bail!("expected float elements, got {other:?}"),
    }
}
