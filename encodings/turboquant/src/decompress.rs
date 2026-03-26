// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! TurboQuant decoding (dequantization) logic.

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;

use crate::mse::array::TurboQuantMSEArray;
use crate::qjl::array::TurboQuantQJLArray;
use crate::rotation::RotationMatrix;

/// Decompress a `TurboQuantMSEArray` into a `FixedSizeListArray` of floats.
///
/// Reads stored centroids and rotation signs from the array's children,
/// avoiding any recomputation.
pub fn execute_decompress_mse(
    array: TurboQuantMSEArray,
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

    // Expand stored rotation signs into f32 ±1.0 vectors once (amortized over all rows).
    // This costs 3 × padded_dim × 4 bytes of temporary memory (e.g. 12KB for dim=1024)
    // but enables autovectorized f32 multiply in the per-row SRHT hot loop.
    let signs_bool = array.rotation_signs.clone().execute::<BoolArray>(ctx)?;
    let rotation = RotationMatrix::from_bool_array(&signs_bool, dim)?;

    // Unpack codes.
    let codes_prim = array.codes.clone().execute::<PrimitiveArray>(ctx)?;
    let indices = codes_prim.as_slice::<u8>();

    let norms_prim = array.norms.clone().execute::<PrimitiveArray>(ctx)?;
    let norms = norms_prim.as_slice::<f32>();

    let mut output = BufferMut::<f32>::with_capacity(num_rows * dim);
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

        output.extend_from_slice(&unrotated[..dim]);
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

/// Decompress a `TurboQuantQJLArray` into a `FixedSizeListArray` of floats.
///
/// First decodes the inner MSE array, then applies QJL residual correction.
pub fn execute_decompress_qjl(
    array: TurboQuantQJLArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let padded_dim = array.padded_dim() as usize;
    let num_rows = array.residual_norms.len();

    if num_rows == 0 {
        return Ok(array
            .mse_inner
            .execute::<FixedSizeListArray>(ctx)?
            .into_array());
    }

    // Decode MSE inner → FixedSizeListArray.
    let mse_decoded = array.mse_inner.clone().execute::<FixedSizeListArray>(ctx)?;
    let mse_elements_prim = mse_decoded.elements().to_canonical()?.into_primitive();
    let mse_elements = mse_elements_prim.as_slice::<f32>();
    let dim = mse_decoded.list_size() as usize;

    // Read QJL signs.
    let qjl_signs_bool = array.qjl_signs.clone().execute::<BoolArray>(ctx)?;
    let qjl_bit_buf = qjl_signs_bool.to_bit_buffer();

    // Read residual norms.
    let residual_norms_prim = array
        .residual_norms
        .clone()
        .execute::<PrimitiveArray>(ctx)?;
    let residual_norms = residual_norms_prim.as_slice::<f32>();

    // Read QJL rotation signs and reconstruct the rotation matrix.
    let qjl_rot_signs_bool = array.rotation_signs.clone().execute::<BoolArray>(ctx)?;
    let qjl_rot = RotationMatrix::from_bool_array(&qjl_rot_signs_bool, dim)?;

    // QJL correction scale: sqrt(π/2) / padded_dim.
    // This accounts for the SRHT normalization (1/padded_dim^{3/2} per transform)
    // combined with the E[|z|] = sqrt(2/π) expectation of half-normal signs.
    // Verified empirically via the `qjl_inner_product_bias` test suite.
    let qjl_scale = (std::f32::consts::FRAC_PI_2).sqrt() / (padded_dim as f32);

    let mut output = BufferMut::<f32>::with_capacity(num_rows * dim);
    let mut qjl_signs_vec = vec![0.0f32; padded_dim];
    let mut qjl_projected = vec![0.0f32; padded_dim];

    for row in 0..num_rows {
        let mse_row = &mse_elements[row * dim..(row + 1) * dim];
        let residual_norm = residual_norms[row];

        let bit_offset = row * padded_dim;
        for idx in 0..padded_dim {
            qjl_signs_vec[idx] = if qjl_bit_buf.value(bit_offset + idx) {
                1.0
            } else {
                -1.0
            };
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
        mse_decoded.list_size(),
        Validity::NonNullable,
        num_rows,
    )?
    .into_array())
}
