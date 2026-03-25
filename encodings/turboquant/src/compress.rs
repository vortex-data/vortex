// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! TurboQuant encoding (quantization) logic.

use vortex_array::IntoArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::PType;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_fastlanes::bitpack_compress::bitpack_encode;

use crate::array::TurboQuantArray;
use crate::array::TurboQuantVariant;
use crate::centroids::find_nearest_centroid;
use crate::centroids::get_centroids;
use crate::rotation::RotationMatrix;

/// Configuration for TurboQuant encoding.
#[derive(Clone, Debug)]
pub struct TurboQuantConfig {
    /// Bits per coordinate (1-4).
    pub bit_width: u8,
    /// Which variant to use.
    pub variant: TurboQuantVariant,
    /// Optional seed for the rotation matrix. If None, a random seed is generated.
    pub seed: Option<u64>,
}

/// Encode a FixedSizeListArray of floats into a TurboQuantArray.
///
/// The input should be the storage array of a Vector or FixedShapeTensor extension type.
/// Each row (fixed-size-list element) is treated as a d-dimensional vector to quantize.
pub fn turboquant_encode(
    fsl: &FixedSizeListArray,
    config: &TurboQuantConfig,
) -> VortexResult<TurboQuantArray> {
    vortex_ensure!(
        config.bit_width >= 1 && config.bit_width <= 4,
        "bit_width must be 1-4, got {}",
        config.bit_width
    );
    if config.variant == TurboQuantVariant::Prod {
        vortex_ensure!(
            config.bit_width >= 2,
            "Prod variant requires bit_width >= 2, got {}",
            config.bit_width
        );
    }

    let dimension = fsl.list_size();
    let num_rows = fsl.len();

    if num_rows == 0 {
        return encode_empty(fsl, config, dimension);
    }

    let seed = config.seed.unwrap_or_else(rand::random);

    // Extract flat f32 elements from the FixedSizeListArray.
    let f32_elements = extract_f32_elements(fsl)?;

    match config.variant {
        TurboQuantVariant::Mse => encode_mse(
            &f32_elements,
            num_rows,
            dimension,
            config.bit_width,
            seed,
            fsl,
        ),
        TurboQuantVariant::Prod => encode_prod(
            &f32_elements,
            num_rows,
            dimension,
            config.bit_width,
            seed,
            fsl,
        ),
    }
}

/// Extract elements from a FixedSizeListArray as a flat f32 vec.
#[allow(clippy::cast_possible_truncation)]
fn extract_f32_elements(fsl: &FixedSizeListArray) -> VortexResult<Vec<f32>> {
    let elements = fsl.elements();
    let ptype = elements.dtype().as_ptype();
    let primitive = elements.to_canonical()?.into_primitive();

    match ptype {
        PType::F32 => Ok(primitive.as_slice::<f32>().to_vec()),
        PType::F64 => Ok(primitive
            .as_slice::<f64>()
            .iter()
            .map(|&v| v as f32)
            .collect()),
        _ => vortex_bail!("TurboQuant requires f32 or f64 elements, got {ptype:?}"),
    }
}

fn encode_empty(
    fsl: &FixedSizeListArray,
    config: &TurboQuantConfig,
    dimension: u32,
) -> VortexResult<TurboQuantArray> {
    let seed = config.seed.unwrap_or(0);
    let codes = PrimitiveArray::empty::<u8>(fsl.dtype().nullability());
    let norms = PrimitiveArray::empty::<f32>(fsl.dtype().nullability());

    match config.variant {
        TurboQuantVariant::Mse => TurboQuantArray::try_new_mse(
            fsl.dtype().clone(),
            codes.into_array(),
            norms.into_array(),
            dimension,
            config.bit_width,
            seed,
        ),
        TurboQuantVariant::Prod => {
            let qjl_signs = PrimitiveArray::empty::<u8>(fsl.dtype().nullability());
            let residual_norms = PrimitiveArray::empty::<f32>(fsl.dtype().nullability());
            TurboQuantArray::try_new_prod(
                fsl.dtype().clone(),
                codes.into_array(),
                norms.into_array(),
                qjl_signs.into_array(),
                residual_norms.into_array(),
                dimension,
                config.bit_width,
                seed,
            )
        }
    }
}

fn encode_mse(
    elements: &[f32],
    num_rows: usize,
    dimension: u32,
    bit_width: u8,
    seed: u64,
    fsl: &FixedSizeListArray,
) -> VortexResult<TurboQuantArray> {
    let d = dimension as usize;
    let rotation = RotationMatrix::try_new(seed, d)?;
    let centroids = get_centroids(dimension, bit_width)?;

    let mut all_indices = BufferMut::<u8>::with_capacity(num_rows * d);
    let mut norms_buf = BufferMut::<f32>::with_capacity(num_rows);

    let mut rotated = vec![0.0f32; d];

    for row in 0..num_rows {
        let x = &elements[row * d..(row + 1) * d];

        // Compute L2 norm.
        let norm = l2_norm(x);
        norms_buf.push(norm);

        // Normalize and rotate.
        if norm > 0.0 {
            let inv_norm = 1.0 / norm;
            let normalized: Vec<f32> = x.iter().map(|&v| v * inv_norm).collect();
            rotation.rotate(&normalized, &mut rotated);
        } else {
            rotated.fill(0.0);
        }

        // Quantize each coordinate to nearest centroid.
        for j in 0..d {
            all_indices.push(find_nearest_centroid(rotated[j], &centroids));
        }
    }

    // Bitpack indices via FastLanes.
    let indices_array = PrimitiveArray::new::<u8>(all_indices.freeze(), Validity::NonNullable);
    let bitpacked = bitpack_encode(&indices_array, bit_width, None)?;

    let norms_array = PrimitiveArray::new::<f32>(norms_buf.freeze(), Validity::NonNullable);

    TurboQuantArray::try_new_mse(
        fsl.dtype().clone(),
        bitpacked.into_array(),
        norms_array.into_array(),
        dimension,
        bit_width,
        seed,
    )
}

fn encode_prod(
    elements: &[f32],
    num_rows: usize,
    dimension: u32,
    bit_width: u8,
    seed: u64,
    fsl: &FixedSizeListArray,
) -> VortexResult<TurboQuantArray> {
    let d = dimension as usize;
    let mse_bit_width = bit_width - 1;

    let rotation = RotationMatrix::try_new(seed, d)?;
    let centroids = get_centroids(dimension, mse_bit_width)?;

    let mut all_indices = BufferMut::<u8>::with_capacity(num_rows * d);
    let mut norms_buf = BufferMut::<f32>::with_capacity(num_rows);
    let mut residual_norms_buf = BufferMut::<f32>::with_capacity(num_rows);

    // QJL sign bits: num_rows * d bits, packed into bytes.
    let total_sign_bits = num_rows * d;
    let sign_bytes = total_sign_bits.div_ceil(8);
    let mut sign_buf = vec![0u8; sign_bytes];

    let mut rotated = vec![0.0f32; d];
    let mut dequantized_rotated = vec![0.0f32; d];
    let mut dequantized = vec![0.0f32; d];

    // QJL random sign matrix generator (using seed + 1).
    let qjl_rotation = RotationMatrix::try_new(seed.wrapping_add(1), d)?;

    for row in 0..num_rows {
        let x = &elements[row * d..(row + 1) * d];

        // Compute L2 norm.
        let norm = l2_norm(x);
        norms_buf.push(norm);

        // Normalize and rotate.
        if norm > 0.0 {
            let inv_norm = 1.0 / norm;
            let normalized: Vec<f32> = x.iter().map(|&v| v * inv_norm).collect();
            rotation.rotate(&normalized, &mut rotated);
        } else {
            rotated.fill(0.0);
        }

        // MSE quantize at (bit_width - 1) bits.
        for j in 0..d {
            let idx = find_nearest_centroid(rotated[j], &centroids);
            all_indices.push(idx);
            dequantized_rotated[j] = centroids[idx as usize];
        }

        // Dequantize MSE result.
        rotation.inverse_rotate(&dequantized_rotated, &mut dequantized);
        if norm > 0.0 {
            for j in 0..d {
                dequantized[j] *= norm;
            }
        }

        // Compute residual r = x - x_hat_mse.
        let residual: Vec<f32> = x
            .iter()
            .zip(dequantized.iter())
            .map(|(&a, &b)| a - b)
            .collect();
        let residual_norm = l2_norm(&residual);
        residual_norms_buf.push(residual_norm);

        // QJL: sign(S * r) where S is another orthogonal matrix.
        // We use the QJL rotation to project the residual, then take signs.
        let mut projected = vec![0.0f32; d];
        if residual_norm > 0.0 {
            qjl_rotation.rotate(&residual, &mut projected);
        }

        // Store sign bits.
        let bit_offset = row * d;
        for j in 0..d {
            if projected[j] >= 0.0 {
                let bit_idx = bit_offset + j;
                sign_buf[bit_idx / 8] |= 1 << (bit_idx % 8);
            }
        }
    }

    // Bitpack MSE indices via FastLanes.
    let indices_array = PrimitiveArray::new::<u8>(all_indices.freeze(), Validity::NonNullable);
    let bitpacked = if mse_bit_width > 0 {
        bitpack_encode(&indices_array, mse_bit_width, None)?
    } else {
        // 0-bit MSE encoding (bit_width=1 for Prod means 0-bit MSE).
        // This shouldn't happen since we validate bit_width >= 2 for Prod.
        unreachable!("Prod variant requires bit_width >= 2")
    };

    let norms_array = PrimitiveArray::new::<f32>(norms_buf.freeze(), Validity::NonNullable);
    let residual_norms_array =
        PrimitiveArray::new::<f32>(residual_norms_buf.freeze(), Validity::NonNullable);

    // Store QJL signs as a u8 PrimitiveArray (packed bits).
    let mut sign_buf_mut = BufferMut::<u8>::with_capacity(sign_buf.len());
    sign_buf_mut.extend_from_slice(&sign_buf);
    let qjl_signs = PrimitiveArray::new::<u8>(sign_buf_mut.freeze(), Validity::NonNullable);

    TurboQuantArray::try_new_prod(
        fsl.dtype().clone(),
        bitpacked.into_array(),
        norms_array.into_array(),
        qjl_signs.into_array(),
        residual_norms_array.into_array(),
        dimension,
        bit_width,
        seed,
    )
}

/// Compute the L2 norm of a vector.
#[inline]
fn l2_norm(x: &[f32]) -> f32 {
    x.iter().map(|&v| v * v).sum::<f32>().sqrt()
}
