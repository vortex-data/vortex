// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! TurboQuant decoding (dequantization) logic.

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;

use crate::array::TurboQuantArray;
use crate::rotation::RotationMatrix;

/// QJL correction scale factor: `sqrt(π/2) / padded_dim`.
///
/// Accounts for the SRHT normalization (`1/padded_dim^{3/2}` per transform)
/// combined with `E[|z|] = sqrt(2/π)` for half-normal sign expectations.
/// Verified empirically via the `qjl_inner_product_bias` test suite.
#[inline]
fn qjl_correction_scale(padded_dim: usize) -> f32 {
    (std::f32::consts::FRAC_PI_2).sqrt() / (padded_dim as f32)
}

/// Decompress a `TurboQuantArray` into a `FixedSizeListArray` of floats.
///
/// Reads stored centroids and rotation signs from the array's children,
/// avoiding any recomputation. If QJL correction is present, the MSE decode
/// and QJL correction are fused into a single pass over rows to avoid an
/// intermediate buffer allocation and extra memory traffic.
pub fn execute_decompress(
    array: TurboQuantArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let dim = array.dimension() as usize;
    let padded_dim = array.padded_dim() as usize;
    let num_rows = array.norms.len();

    if num_rows == 0 {
        let elements = PrimitiveArray::empty::<f32>(array.dtype.nullability());
        return Ok(FixedSizeListArray::try_new(
            elements.into_array(),
            array.dimension(),
            Validity::NonNullable,
            0,
        )?
        .into_array());
    }

    // Read stored centroids — no recomputation.
    let centroids_prim = array.centroids.clone().execute::<PrimitiveArray>(ctx)?;
    let centroids = centroids_prim.as_slice::<f32>();

    // FastLanes SIMD-unpacks the 1-bit bitpacked rotation signs into u8 0/1 values,
    // then we expand to u32 XOR masks once (amortized over all rows).
    let signs_prim = array
        .rotation_signs
        .clone()
        .execute::<PrimitiveArray>(ctx)?;
    let rotation = RotationMatrix::from_u8_slice(signs_prim.as_slice::<u8>(), dim)?;

    // Unpack codes.
    let codes_prim = array.codes.clone().execute::<PrimitiveArray>(ctx)?;
    let indices = codes_prim.as_slice::<u8>();

    let norms_prim = array.norms.clone().execute::<PrimitiveArray>(ctx)?;
    let norms = norms_prim.as_slice::<f32>();

    // Prepare QJL data (if present) before entering the row loop.
    // QJL reuses the MSE rotation matrix — no separate rotation to reconstruct.
    let qjl_scale = qjl_correction_scale(padded_dim);
    let qjl_data = if let Some(qjl) = &array.qjl {
        let qjl_signs_prim = qjl.signs.clone().execute::<PrimitiveArray>(ctx)?;
        let residual_norms_prim = qjl.residual_norms.clone().execute::<PrimitiveArray>(ctx)?;
        Some((qjl_signs_prim, residual_norms_prim))
    } else {
        None
    };

    // Single fused loop: MSE decode + optional QJL correction per row.
    let mut output = BufferMut::<f32>::with_capacity(num_rows * dim);
    let mut dequantized = vec![0.0f32; padded_dim];
    let mut unrotated = vec![0.0f32; padded_dim];
    // QJL scratch buffers (only used when qjl_data is Some).
    let mut qjl_signs_vec = vec![0.0f32; padded_dim];
    let mut qjl_projected = vec![0.0f32; padded_dim];

    for row in 0..num_rows {
        let row_indices = &indices[row * padded_dim..(row + 1) * padded_dim];
        let norm = norms[row];

        // MSE: dequantize → inverse rotate → scale by norm.
        for idx in 0..padded_dim {
            dequantized[idx] = centroids[row_indices[idx] as usize];
        }
        rotation.inverse_rotate(&dequantized, &mut unrotated);
        for idx in 0..dim {
            unrotated[idx] *= norm;
        }

        if let Some((ref qjl_signs_prim, ref residual_norms_prim)) = qjl_data {
            // QJL: apply residual correction inline, reusing the MSE rotation.
            let qjl_signs_u8 = qjl_signs_prim.as_slice::<u8>();
            let residual_norms = residual_norms_prim.as_slice::<f32>();
            let residual_norm = residual_norms[row];

            let row_signs = &qjl_signs_u8[row * padded_dim..(row + 1) * padded_dim];
            for idx in 0..padded_dim {
                qjl_signs_vec[idx] = if row_signs[idx] != 0 { 1.0 } else { -1.0 };
            }

            rotation.inverse_rotate(&qjl_signs_vec, &mut qjl_projected);
            let scale = qjl_scale * residual_norm;

            for idx in 0..dim {
                output.push(unrotated[idx] + scale * qjl_projected[idx]);
            }
        } else {
            output.extend_from_slice(&unrotated[..dim]);
        }
    }

    let elements = PrimitiveArray::new::<f32>(output.freeze(), Validity::NonNullable);
    Ok(FixedSizeListArray::try_new(
        elements.into_array(),
        array.dimension(),
        Validity::NonNullable,
        num_rows,
    )?
    .into_array())
}
