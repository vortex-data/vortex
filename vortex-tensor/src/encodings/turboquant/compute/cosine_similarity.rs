// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Approximate cosine similarity in the quantized domain.
//!
//! Since the SRHT is orthogonal, inner products are preserved in the rotated
//! domain. For two TurboQuant arrays that share the same SRHT rotation (i.e.,
//! encoded from the same column), we can compute the dot product of their
//! quantized representations without full decompression:
//!
//! ```text
//! cos_approx(a, b) = sum(centroids[code_a[j]] × centroids[code_b[j]])
//! ```
//!
//! where `code_a` and `code_b` are the quantized coordinate indices of the
//! unit-norm rotated vectors `â_rot` and `b̂_rot`.
//!
//! # Bias and error bounds
//!
//! This estimate is **biased**. The MSE quantizer minimizes reconstruction error
//! but does not guarantee unbiased inner products; the discrete centroid grid
//! introduces systematic bias in the dot product.
//!
//! The approximation error is bounded by the MSE quantization distortion. For
//! unit-norm vectors quantized at `b` bits, the per-coordinate MSE is bounded by
//! `(√3 · π / 2) / 4^b` (Theorem 1). The inner product error scales with this
//! distortion: at 4 bits the error is typically < 0.1, at 8 bits < 0.001.
//!
//! For approximate nearest neighbor (ANN) search, biased-but-accurate ranking is
//! usually sufficient -- the relative ordering of cosine similarities is preserved
//! even if the absolute values have bounded error.

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::match_each_float_ptype;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;

use crate::encodings::turboquant::TurboQuant;
use crate::encodings::turboquant::array::float_from_f32;
use crate::utils::extension_element_ptype;

/// Compute the per-row unit-norm dot products in f32 (centroids are always f32).
///
/// Returns a `Vec<f32>` of length `num_rows`.
fn compute_unit_dots(
    lhs: &ArrayView<TurboQuant>,
    rhs: &ArrayView<TurboQuant>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Vec<f32>> {
    let pd = lhs.padded_dim() as usize;
    let num_rows = lhs.norms().len();

    let lhs_codes_fsl: FixedSizeListArray = lhs.codes().clone().execute(ctx)?;
    let rhs_codes_fsl: FixedSizeListArray = rhs.codes().clone().execute(ctx)?;
    let lhs_codes: PrimitiveArray = lhs_codes_fsl.elements().clone().execute(ctx)?;
    let rhs_codes: PrimitiveArray = rhs_codes_fsl.elements().clone().execute(ctx)?;
    let ca = lhs_codes.as_slice::<u8>();
    let cb = rhs_codes.as_slice::<u8>();

    // Read centroids from both arrays. They may have different codebooks (e.g., different bit
    // widths).
    let lhs_centroids: PrimitiveArray = lhs.centroids().clone().execute(ctx)?;
    let rhs_centroids: PrimitiveArray = rhs.centroids().clone().execute(ctx)?;
    let cl = lhs_centroids.as_slice::<f32>();
    let cr = rhs_centroids.as_slice::<f32>();

    let mut dots = Vec::with_capacity(num_rows);
    for row in 0..num_rows {
        let row_ca = &ca[row * pd..(row + 1) * pd];
        let row_cb = &cb[row * pd..(row + 1) * pd];
        let dot: f32 = row_ca
            .iter()
            .zip(row_cb.iter())
            .map(|(&a, &b)| cl[a as usize] * cr[b as usize])
            .sum();
        dots.push(dot);
    }

    Ok(dots)
}

/// Compute approximate cosine similarity for all rows between two TurboQuant arrays without
/// full decompression.
///
/// Both arrays must share the same rotation (i.e., were encoded from the same TurboQuant
/// column). For this function, results are meaningless if the rotations differ (there are other
/// methods that can allow this, but that is future work).
///
/// Since TurboQuant stores unit-normalized rotated vectors, the dot product of the quantized
/// codes directly approximates cosine similarity without needing the stored norms.
///
/// The output dtype matches the Vector's element type (f16, f32, or f64).
pub fn cosine_similarity_quantized_column(
    lhs: ArrayView<TurboQuant>,
    rhs: ArrayView<TurboQuant>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    vortex_ensure_eq!(
        lhs.dimension(),
        rhs.dimension(),
        "TurboQuant quantized dot product requires matching dimensions",
    );

    let element_ptype = extension_element_ptype(lhs.dtype().as_extension())?;
    let validity = lhs.norms().validity()?.and(rhs.norms().validity()?)?;
    let dots = compute_unit_dots(&lhs, &rhs, ctx)?;

    // The unit-norm dot product IS the cosine similarity. Cast from f32 to the native type.
    match_each_float_ptype!(element_ptype, |T| {
        let mut result = BufferMut::<T>::with_capacity(dots.len());
        for &dot in &dots {
            // SAFETY: We allocated the correct amount.
            unsafe { result.push_unchecked(float_from_f32(dot)) };
        }

        // SAFETY: `result` has the same length as the input arrays, matching `validity`.
        Ok(unsafe { PrimitiveArray::new_unchecked(result.freeze(), validity) }.into_array())
    })
}

/// Compute approximate dot product for all rows between two TurboQuant arrays without
/// full decompression.
///
/// Both arrays must share the same SRHT rotation (i.e., were encoded from the same TurboQuant
/// column). For this function, results are meaningless if the rotations differ (there are other
/// methods that can allow this, but that is future work).
///
/// `dot_product(a, b) = ||a|| * ||b|| * sum(c[code_a[j]] * c[code_b[j]])`
///
/// The output dtype matches the Vector's element type (f16, f32, or f64).
pub fn dot_product_quantized_column(
    lhs: ArrayView<TurboQuant>,
    rhs: ArrayView<TurboQuant>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    vortex_ensure_eq!(
        lhs.dimension(),
        rhs.dimension(),
        "TurboQuant quantized dot product requires matching dimensions",
    );

    let element_ptype = extension_element_ptype(lhs.dtype().as_extension())?;
    let validity = lhs.norms().validity()?.and(rhs.norms().validity()?)?;
    let dots = compute_unit_dots(&lhs, &rhs, ctx)?;
    let num_rows = lhs.norms().len();

    let lhs_norms: PrimitiveArray = lhs.norms().clone().execute(ctx)?;
    let rhs_norms: PrimitiveArray = rhs.norms().clone().execute(ctx)?;

    // Scale the f32 unit-norm dot product by native-precision norms.
    match_each_float_ptype!(element_ptype, |T| {
        let na = lhs_norms.as_slice::<T>();
        let nb = rhs_norms.as_slice::<T>();

        let mut result = BufferMut::<T>::with_capacity(num_rows);
        for row in 0..num_rows {
            let dot_t: T = float_from_f32(dots[row]);
            // SAFETY: We allocated the correct amount.
            unsafe { result.push_unchecked(na[row] * nb[row] * dot_t) };
        }

        // SAFETY: `result` has the same length as the input arrays, matching `validity`.
        Ok(unsafe { PrimitiveArray::new_unchecked(result.freeze(), validity) }.into_array())
    })
}
