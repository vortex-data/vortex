// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Approximate cosine similarity in the quantized domain.
//!
//! Since the SRHT is orthogonal, inner products are preserved in the rotated
//! domain. For two vectors from the same TurboQuant column (same rotation and
//! centroids), we can compute the dot product of their quantized representations
//! without full decompression:
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
//! usually sufficient — the relative ordering of cosine similarities is preserved
//! even if the absolute values have bounded error.

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::encodings::turboquant::TurboQuant;

/// Shared helper: read codes, norms, and centroids from two TurboQuant arrays,
/// then compute per-row quantized unit-norm dot products.
///
/// Both arrays must have the same dimension (vector length) and row count.
/// They may have different codebooks (e.g., different bit widths), in which
/// case each array's own centroids are used for its code lookups.
///
/// Returns `(norms_a, norms_b, unit_dots)` where `unit_dots[i]` is the dot product
/// of the unit-norm quantized vectors for row i.
fn quantized_unit_dots(
    lhs: ArrayView<TurboQuant>,
    rhs: ArrayView<TurboQuant>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    vortex_ensure!(
        lhs.dimension() == rhs.dimension(),
        "TurboQuant quantized dot product requires matching dimensions, got {} and {}",
        lhs.dimension(),
        rhs.dimension()
    );

    let pd = lhs.padded_dim() as usize;
    let num_rows = lhs.norms().len();

    let lhs_norms: PrimitiveArray = lhs.norms().clone().execute(ctx)?;
    let rhs_norms: PrimitiveArray = rhs.norms().clone().execute(ctx)?;
    let na = lhs_norms.as_slice::<f32>();
    let nb = rhs_norms.as_slice::<f32>();

    let lhs_codes_fsl: FixedSizeListArray = lhs.codes().clone().execute(ctx)?;
    let rhs_codes_fsl: FixedSizeListArray = rhs.codes().clone().execute(ctx)?;
    let lhs_codes = lhs_codes_fsl.elements().to_canonical()?.into_primitive();
    let rhs_codes = rhs_codes_fsl.elements().to_canonical()?.into_primitive();
    let ca = lhs_codes.as_slice::<u8>();
    let cb = rhs_codes.as_slice::<u8>();

    // Read centroids from both arrays — they may have different codebooks
    // (e.g., different bit widths).
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

    Ok((na.to_vec(), nb.to_vec(), dots))
}

/// Compute approximate cosine similarity for all rows between two TurboQuant
/// arrays (same rotation matrix and codebook) without full decompression.
pub fn cosine_similarity_quantized_column(
    lhs: ArrayView<TurboQuant>,
    rhs: ArrayView<TurboQuant>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let num_rows = lhs.norms().len();
    let (na, nb, dots) = quantized_unit_dots(lhs, rhs, ctx)?;

    let mut result = BufferMut::<f32>::with_capacity(num_rows);
    for row in 0..num_rows {
        if na[row] == 0.0 || nb[row] == 0.0 {
            result.push(0.0);
        } else {
            // Unit-norm dot product IS the cosine similarity.
            result.push(dots[row]);
        }
    }

    Ok(PrimitiveArray::new::<f32>(result.freeze(), Validity::NonNullable).into_array())
}

/// Compute approximate dot product for all rows between two TurboQuant
/// arrays (same rotation matrix and codebook) without full decompression.
///
/// `dot_product(a, b) ≈ ||a|| * ||b|| * sum(c[code_a[j]] * c[code_b[j]])`
pub fn dot_product_quantized_column(
    lhs: ArrayView<TurboQuant>,
    rhs: ArrayView<TurboQuant>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let num_rows = lhs.norms().len();
    let (na, nb, dots) = quantized_unit_dots(lhs, rhs, ctx)?;

    let mut result = BufferMut::<f32>::with_capacity(num_rows);
    for row in 0..num_rows {
        // Scale the unit-norm dot product by both norms to get the actual dot product.
        result.push(na[row] * nb[row] * dots[row]);
    }

    Ok(PrimitiveArray::new::<f32>(result.freeze(), Validity::NonNullable).into_array())
}
