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
//! This estimate is **biased** — it uses only the MSE-quantized codes and does
//! not incorporate the QJL residual correction. The MSE quantizer minimizes
//! reconstruction error but does not guarantee unbiased inner products; the
//! discrete centroid grid introduces systematic bias in the dot product.
//!
//! The TurboQuant paper's Theorem 2 shows that unbiased inner product estimation
//! requires the full QJL correction term, which involves decoding the per-row
//! QJL signs and computing cross-terms — nearly as expensive as full decompression.
//!
//! The approximation error is bounded by the MSE quantization distortion. For
//! unit-norm vectors quantized at `b` bits, the per-coordinate MSE is bounded by
//! `(√3 · π / 2) / 4^b` (Theorem 1). The inner product error scales with this
//! distortion: at 4 bits the error is typically < 0.1, at 8 bits < 0.001.
//!
//! For approximate nearest neighbor (ANN) search, biased-but-accurate ranking is
//! usually sufficient — the relative ordering of cosine similarities is preserved
//! even if the absolute values have bounded error.

use vortex::array::ExecutionCtx;
use vortex::array::arrays::FixedSizeListArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::error::VortexResult;

use crate::encodings::turboquant::array::TurboQuantArray;

/// Compute approximate cosine similarity between two rows of a TurboQuant array
/// without full decompression.
///
/// Both rows must come from the same array (same rotation matrix and codebook).
/// The result is a **biased estimate** using only MSE-quantized codes (no QJL
/// correction). The error is bounded by the quantization distortion — see the
/// module-level documentation for details.
///
/// TODO: Wire into `vortex-tensor` cosine_similarity scalar function dispatch
/// so that `cosine_similarity(Extension(TurboQuant), Extension(TurboQuant))`
/// short-circuits to this when both arguments share the same encoding.
#[allow(dead_code)] // TODO: wire into vortex-tensor cosine_similarity dispatch
pub fn cosine_similarity_quantized(
    array: &TurboQuantArray,
    row_a: usize,
    row_b: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<f32> {
    let pd = array.padded_dim() as usize;

    // Read norms — execute to handle cascade-compressed children.
    let norms_prim = array.norms().clone().execute::<PrimitiveArray>(ctx)?;
    let norms = norms_prim.as_slice::<f32>();
    let norm_a = norms[row_a];
    let norm_b = norms[row_b];

    if norm_a == 0.0 || norm_b == 0.0 {
        return Ok(0.0);
    }

    // Read codes from the FixedSizeListArray → flat u8.
    let codes_fsl = array.codes().clone().execute::<FixedSizeListArray>(ctx)?;
    let codes_prim = codes_fsl.elements().to_canonical()?.into_primitive();
    let all_codes = codes_prim.as_slice::<u8>();

    // Read centroids.
    let centroids_prim = array.centroids().clone().execute::<PrimitiveArray>(ctx)?;
    let c = centroids_prim.as_slice::<f32>();

    let codes_a = &all_codes[row_a * pd..(row_a + 1) * pd];
    let codes_b = &all_codes[row_b * pd..(row_b + 1) * pd];

    // Dot product of unit-norm quantized vectors in rotated domain.
    // Since SRHT preserves inner products, this equals the dot product
    // of the dequantized (but still unit-norm) vectors.
    let dot: f32 = codes_a
        .iter()
        .zip(codes_b.iter())
        .map(|(&ca, &cb)| c[ca as usize] * c[cb as usize])
        .sum();

    Ok(dot)
}
