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
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::array::TurboQuantArray;
use crate::array::TurboQuantVariant;
use crate::centroids::get_centroids;
use crate::rotation::RotationMatrix;

/// Decompress a TurboQuantArray back into a FixedSizeListArray of floats.
pub fn execute_decompress(
    array: TurboQuantArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    match array.variant() {
        TurboQuantVariant::Mse => decode_mse(array, ctx),
        TurboQuantVariant::Prod => decode_prod(array, ctx),
    }
}

fn decode_mse(array: TurboQuantArray, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
    let dimension = array.dimension();
    let dim = dimension as usize;
    let bit_width = array.bit_width();
    let seed = array.rotation_seed();
    let num_rows = array.norms.len();

    if num_rows == 0 {
        let elements = PrimitiveArray::empty::<f32>(array.dtype.nullability());
        return Ok(FixedSizeListArray::try_new(
            elements.into_array(),
            dimension,
            Validity::NonNullable,
            0,
        )?
        .into_array());
    }

    let rotation = RotationMatrix::try_new(seed, dim)?;
    let padded_dim = rotation.padded_dim();

    // Unpack codes — these are padded_dim indices per row.
    let codes_prim = array.codes.clone().execute::<PrimitiveArray>(ctx)?;
    let indices = codes_prim.as_slice::<u8>();

    let norms_prim = array.norms.clone().execute::<PrimitiveArray>(ctx)?;
    let norms = norms_prim.as_slice::<f32>();

    #[allow(clippy::cast_possible_truncation)]
    let centroids = get_centroids(padded_dim as u32, bit_width)?;

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

        // Scale by norm and take only the first dim elements.
        for idx in 0..dim {
            unrotated[idx] *= norm;
        }

        output.extend_from_slice(&unrotated[..dim]);
    }

    let elements = PrimitiveArray::new::<f32>(output.freeze(), Validity::NonNullable);
    Ok(FixedSizeListArray::try_new(
        elements.into_array(),
        dimension,
        Validity::NonNullable,
        num_rows,
    )?
    .into_array())
}

fn decode_prod(array: TurboQuantArray, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
    let dimension = array.dimension();
    let dim = dimension as usize;
    let mse_bit_width = array.bit_width() - 1;
    let seed = array.rotation_seed();
    let num_rows = array.norms.len();

    if num_rows == 0 {
        let elements = PrimitiveArray::empty::<f32>(array.dtype.nullability());
        return Ok(FixedSizeListArray::try_new(
            elements.into_array(),
            dimension,
            Validity::NonNullable,
            0,
        )?
        .into_array());
    }

    let rotation = RotationMatrix::try_new(seed, dim)?;
    let padded_dim = rotation.padded_dim();

    let codes_prim = array.codes.clone().execute::<PrimitiveArray>(ctx)?;
    let indices = codes_prim.as_slice::<u8>();

    let norms_prim = array.norms.clone().execute::<PrimitiveArray>(ctx)?;
    let norms = norms_prim.as_slice::<f32>();

    let residual_norms_prim = array
        .residual_norms
        .as_ref()
        .vortex_expect("Prod variant must have residual_norms")
        .clone()
        .execute::<PrimitiveArray>(ctx)?;
    let residual_norms = residual_norms_prim.as_slice::<f32>();

    let qjl_prim = array
        .qjl_signs
        .as_ref()
        .vortex_expect("Prod variant must have qjl_signs")
        .clone()
        .execute::<PrimitiveArray>(ctx)?;
    let sign_bytes = qjl_prim.as_slice::<u8>();

    #[allow(clippy::cast_possible_truncation)]
    let centroids = get_centroids(padded_dim as u32, mse_bit_width)?;
    let qjl_rotation = RotationMatrix::try_new(seed.wrapping_add(1), dim)?;

    let qjl_scale = (std::f32::consts::FRAC_PI_2).sqrt() / (padded_dim as f32);

    let mut output = BufferMut::<f32>::with_capacity(num_rows * dim);
    let mut dequantized = vec![0.0f32; padded_dim];
    let mut unrotated = vec![0.0f32; padded_dim];
    let mut qjl_signs_vec = vec![0.0f32; padded_dim];
    let mut qjl_projected = vec![0.0f32; padded_dim];

    for row in 0..num_rows {
        let row_indices = &indices[row * padded_dim..(row + 1) * padded_dim];
        let norm = norms[row];
        let residual_norm = residual_norms[row];

        for idx in 0..padded_dim {
            dequantized[idx] = centroids[row_indices[idx] as usize];
        }
        rotation.inverse_rotate(&dequantized, &mut unrotated);

        for val in unrotated[..dim].iter_mut() {
            *val *= norm;
        }

        // QJL decode.
        let bit_offset = row * padded_dim;
        for idx in 0..padded_dim {
            let bit_idx = bit_offset + idx;
            let sign_bit = (sign_bytes[bit_idx / 8] >> (bit_idx % 8)) & 1;
            qjl_signs_vec[idx] = if sign_bit == 1 { 1.0 } else { -1.0 };
        }

        qjl_rotation.inverse_rotate(&qjl_signs_vec, &mut qjl_projected);
        let scale = qjl_scale * residual_norm;

        for idx in 0..dim {
            unrotated[idx] += scale * qjl_projected[idx];
        }

        output.extend_from_slice(&unrotated[..dim]);
    }

    let elements = PrimitiveArray::new::<f32>(output.freeze(), Validity::NonNullable);
    Ok(FixedSizeListArray::try_new(
        elements.into_array(),
        dimension,
        Validity::NonNullable,
        num_rows,
    )?
    .into_array())
}
