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

use crate::encodings::turboquant::array::TurboQuantArray;
use crate::encodings::turboquant::rotation::RotationMatrix;

/// QJL correction scale factor: `sqrt(π/2) / padded_dim`.
///
/// Accounts for the SRHT normalization (`1/padded_dim^{3/2}` per transform)
/// combined with `E[|z|] = sqrt(2/π)` for half-normal sign expectations.
#[inline]
fn qjl_correction_scale(padded_dim: usize) -> f32 {
    (std::f32::consts::FRAC_PI_2).sqrt() / (padded_dim as f32)
}

/// Decompress a `TurboQuantArray` into a `FixedSizeListArray` of floats.
///
/// Reads stored centroids and rotation signs from the array's children,
/// avoiding any recomputation. If QJL correction is present, applies
/// the residual correction after MSE decoding.
pub fn execute_decompress(
    array: TurboQuantArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let dim = array.dimension() as usize;
    let padded_dim = array.padded_dim() as usize;
    let num_rows = array.norms().len();

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
    let centroids_prim = array.centroids().clone().execute::<PrimitiveArray>(ctx)?;
    let centroids = centroids_prim.as_slice::<f32>();

    // Unpack optional permutation (for non-power-of-2 dims).
    let perm: Option<Vec<u16>> = array
        .permutation()
        .map(|arr| {
            let prim = arr.clone().execute::<PrimitiveArray>(ctx)?;
            Ok::<_, vortex_error::VortexError>(prim.as_slice::<u16>().to_vec())
        })
        .transpose()?;

    // FastLanes SIMD-unpacks the 1-bit bitpacked rotation signs into u8 0/1 values,
    // then we expand to u32 XOR masks once (amortized over all rows). This enables
    // branchless XOR-based sign application in the per-row SRHT hot loop.
    let signs_prim = array
        .rotation_signs()
        .clone()
        .execute::<PrimitiveArray>(ctx)?;
    let rotation = {
        let rot = RotationMatrix::from_u8_slice(signs_prim.as_slice::<u8>(), dim)?;
        if let Some(ref p) = perm {
            rot.with_permutation(p.clone())
        } else {
            rot
        }
    };

    // Unpack codes from FixedSizeListArray → flat u8 elements.
    let codes_fsl = array.codes().clone().execute::<FixedSizeListArray>(ctx)?;
    let codes_prim = codes_fsl.elements().to_canonical()?.into_primitive();
    let indices = codes_prim.as_slice::<u8>();

    let norms_prim = array.norms().clone().execute::<PrimitiveArray>(ctx)?;
    let norms = norms_prim.as_slice::<f32>();

    // MSE decode: dequantize → inverse rotate → scale by norm.
    let mut mse_output = BufferMut::<f32>::with_capacity(num_rows * dim);
    let mut dequantized = vec![0.0f32; padded_dim];
    let mut unrotated = vec![0.0f32; padded_dim];

    for row in 0..num_rows {
        let row_indices = &indices[row * padded_dim..(row + 1) * padded_dim];
        let norm = norms[row];

        for idx in 0..padded_dim {
            dequantized[idx] = centroids[row_indices[idx] as usize];
        }

        rotation.inverse_rotate(&dequantized, &mut unrotated);

        for idx in 0..dim {
            unrotated[idx] *= norm;
        }

        mse_output.extend_from_slice(&unrotated[..dim]);
    }

    // If no QJL correction, we're done.
    let Some(qjl) = array.qjl() else {
        let elements = PrimitiveArray::new::<f32>(mse_output.freeze(), Validity::NonNullable);
        return Ok(FixedSizeListArray::try_new(
            elements.into_array(),
            array.dimension(),
            Validity::NonNullable,
            num_rows,
        )?
        .into_array());
    };

    // Apply QJL residual correction.
    // Unpack QJL signs from FixedSizeListArray → flat u8 0/1 values.
    let qjl_signs_fsl = qjl.signs.clone().execute::<FixedSizeListArray>(ctx)?;
    let qjl_signs_prim = qjl_signs_fsl.elements().to_canonical()?.into_primitive();
    let qjl_signs_u8 = qjl_signs_prim.as_slice::<u8>();

    let residual_norms_prim = qjl.residual_norms.clone().execute::<PrimitiveArray>(ctx)?;
    let residual_norms = residual_norms_prim.as_slice::<f32>();

    let qjl_rot_signs_prim = qjl.rotation_signs.clone().execute::<PrimitiveArray>(ctx)?;
    let qjl_rot = {
        let rot = RotationMatrix::from_u8_slice(qjl_rot_signs_prim.as_slice::<u8>(), dim)?;
        if let Some(ref p) = perm {
            rot.with_permutation(p.clone())
        } else {
            rot
        }
    };

    let qjl_scale = qjl_correction_scale(padded_dim);
    let mse_elements = mse_output.as_ref();

    let mut output = BufferMut::<f32>::with_capacity(num_rows * dim);
    let mut qjl_signs_vec = vec![0.0f32; padded_dim];
    let mut qjl_projected = vec![0.0f32; padded_dim];

    for row in 0..num_rows {
        let mse_row = &mse_elements[row * dim..(row + 1) * dim];
        let residual_norm = residual_norms[row];

        // Branchless u8 0/1 → f32 ±1.0 via XOR on the IEEE 754 sign bit.
        // 1.0f32 = 0x3F800000; flipping the sign bit gives -1.0 = 0xBF800000.
        // For sign=0 (negative): mask = 0x80000000, 1.0 XOR mask = -1.0.
        // For sign=1 (positive): mask = 0x00000000, 1.0 XOR mask = 1.0.
        let row_signs = &qjl_signs_u8[row * padded_dim..(row + 1) * padded_dim];
        for (dst, &sign) in qjl_signs_vec.iter_mut().zip(row_signs.iter()) {
            let mask = ((sign as u32) ^ 1) << 31;
            *dst = f32::from_bits(0x3F80_0000 ^ mask);
        }

        qjl_rot.inverse_rotate(&qjl_signs_vec, &mut qjl_projected);
        let scale = qjl_scale * residual_norm;

        for idx in 0..dim {
            output.push(mse_row[idx] + scale * qjl_projected[idx]);
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
