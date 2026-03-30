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
//! cos(a, b) = dot(a, b) / (||a|| × ||b||)
//!           = ||a|| × ||b|| × dot(â_rot, b̂_rot) / (||a|| × ||b||)
//!           = sum(centroids[code_a[j]] × centroids[code_b[j]])
//! ```
//!
//! where `â_rot` and `b̂_rot` are the quantized unit-norm rotated vectors.

use vortex_array::DynArray;
use vortex_error::VortexResult;

use crate::array::TurboQuantArray;

/// Compute approximate cosine similarity between two rows of a TurboQuant array
/// without full decompression.
///
/// Both rows must come from the same array (same rotation matrix and codebook).
/// The result has bounded error proportional to the quantization distortion.
///
/// TODO: Wire into `vortex-tensor` cosine_similarity scalar function dispatch
/// so that `cosine_similarity(Extension(TurboQuant), Extension(TurboQuant))`
/// short-circuits to this when both arguments share the same encoding.
#[allow(dead_code)] // TODO: wire into vortex-tensor cosine_similarity dispatch
pub fn cosine_similarity_quantized(
    array: &TurboQuantArray,
    row_a: usize,
    row_b: usize,
) -> VortexResult<f32> {
    let pd = array.padded_dim() as usize;

    // Read norms directly — no decompression.
    let norms_prim = array.norms().to_canonical()?.into_primitive();
    let norms = norms_prim.as_slice::<f32>();
    let norm_a = norms[row_a];
    let norm_b = norms[row_b];

    if norm_a == 0.0 || norm_b == 0.0 {
        return Ok(0.0);
    }

    // Read codes from the FixedSizeListArray → flat u8.
    let codes_fsl = array.codes().to_canonical()?.into_fixed_size_list();
    let codes_prim = codes_fsl.elements().to_canonical()?.into_primitive();
    let all_codes = codes_prim.as_slice::<u8>();

    // Read centroids.
    let centroids_prim = array.centroids().to_canonical()?.into_primitive();
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
