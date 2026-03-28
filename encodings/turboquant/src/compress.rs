// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! TurboQuant encoding (quantization) logic.

use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::validity::Validity;
use vortex_buffer::BitBufferMut;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_fastlanes::bitpack_compress::bitpack_encode;

use crate::centroids::compute_boundaries;
use crate::centroids::find_nearest_centroid;
use crate::centroids::get_centroids;
use crate::mse::array::TurboQuantMSEArray;
use crate::qjl::array::TurboQuantQJLArray;
use crate::rotation::RotationMatrix;

/// Configuration for TurboQuant encoding.
#[derive(Clone, Debug)]
pub struct TurboQuantConfig {
    /// Bits per coordinate.
    ///
    /// For MSE encoding: 1-8.
    /// For QJL encoding: 2-9 (the MSE inner uses `bit_width - 1`).
    pub bit_width: u8,
    /// Optional seed for the rotation matrix. If None, a random seed is generated.
    pub seed: Option<u64>,
}

/// Extract elements from a FixedSizeListArray as a flat f32 vec.
#[allow(clippy::cast_possible_truncation)]
fn extract_f32_elements(fsl: &FixedSizeListArray) -> VortexResult<Vec<f32>> {
    let elements = fsl.elements();
    let primitive = elements.to_canonical()?.into_primitive();
    let ptype = primitive.ptype();

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

/// Compute the L2 norm of a vector.
#[inline]
fn l2_norm(x: &[f32]) -> f32 {
    x.iter().map(|&v| v * v).sum::<f32>().sqrt()
}

/// Encode a FixedSizeListArray into a `TurboQuantMSEArray`.
///
/// The input must be non-nullable. TurboQuant is a lossy encoding that does not
/// preserve null positions; callers must handle validity externally.
pub fn turboquant_encode_mse(
    fsl: &FixedSizeListArray,
    config: &TurboQuantConfig,
) -> VortexResult<TurboQuantMSEArray> {
    vortex_ensure!(
        fsl.dtype().nullability() == Nullability::NonNullable,
        "TurboQuant requires non-nullable input, got nullable FixedSizeListArray"
    );
    vortex_ensure!(
        config.bit_width >= 1 && config.bit_width <= 8,
        "MSE bit_width must be 1-8, got {}",
        config.bit_width
    );
    let dimension = fsl.list_size();
    vortex_ensure!(
        dimension >= 2,
        "TurboQuant requires dimension >= 2, got {dimension}"
    );

    let seed = config.seed.unwrap_or_else(rand::random);
    let dim = dimension as usize;
    let num_rows = fsl.len();

    let rotation = RotationMatrix::try_new(seed, dim)?;
    let padded_dim = rotation.padded_dim();

    if num_rows == 0 {
        return build_empty_mse_array(fsl, config.bit_width, padded_dim, seed);
    }

    let f32_elements = extract_f32_elements(fsl)?;
    #[allow(clippy::cast_possible_truncation)]
    let centroids = get_centroids(padded_dim as u32, config.bit_width)?;
    let boundaries = compute_boundaries(&centroids);

    let mut all_indices = BufferMut::<u8>::with_capacity(num_rows * padded_dim);
    let mut norms_buf = BufferMut::<f32>::with_capacity(num_rows);
    let mut padded = vec![0.0f32; padded_dim];
    let mut rotated = vec![0.0f32; padded_dim];

    for row in 0..num_rows {
        let x = &f32_elements[row * dim..(row + 1) * dim];
        let norm = l2_norm(x);
        norms_buf.push(norm);

        // Normalize and write into [..dim]; tail [dim..padded_dim] stays zero
        // from initialization and is never overwritten.
        if norm > 0.0 {
            let inv_norm = 1.0 / norm;
            for (dst, &src) in padded[..dim].iter_mut().zip(x.iter()) {
                *dst = src * inv_norm;
            }
        } else {
            padded[..dim].fill(0.0);
        }
        rotation.rotate(&padded, &mut rotated);

        for j in 0..padded_dim {
            all_indices.push(find_nearest_centroid(rotated[j], &boundaries));
        }
    }

    // Pack indices: bitpack for 1-7 bits, store raw u8 for 8 bits.
    let indices_array = PrimitiveArray::new::<u8>(all_indices.freeze(), Validity::NonNullable);
    let codes = if config.bit_width < 8 {
        bitpack_encode(&indices_array, config.bit_width, None)?.into_array()
    } else {
        indices_array.into_array()
    };

    let norms_array = PrimitiveArray::new::<f32>(norms_buf.freeze(), Validity::NonNullable);

    // Store centroids as a child array.
    // TODO(perf): `get_centroids` returns Vec<f32>; could avoid the copy by
    // supporting Buffer::from(Vec<T>) or caching as Buffer directly.
    let mut centroids_buf = BufferMut::<f32>::with_capacity(centroids.len());
    centroids_buf.extend_from_slice(&centroids);
    let centroids_array = PrimitiveArray::new::<f32>(centroids_buf.freeze(), Validity::NonNullable);

    // Store rotation signs as a BoolArray child.
    let rotation_signs = rotation.export_inverse_signs_bool_array();

    #[allow(clippy::cast_possible_truncation)]
    TurboQuantMSEArray::try_new(
        fsl.dtype().clone(),
        codes,
        norms_array.into_array(),
        centroids_array.into_array(),
        rotation_signs.into_array(),
        dimension,
        config.bit_width,
        padded_dim as u32,
        seed,
    )
}

/// Encode a FixedSizeListArray into a `TurboQuantQJLArray`.
///
/// Produces a cascaded structure: QJLArray wrapping an MSEArray at `bit_width - 1`.
/// The input must be non-nullable. TurboQuant is a lossy encoding that does not
/// preserve null positions; callers must handle validity externally.
pub fn turboquant_encode_qjl(
    fsl: &FixedSizeListArray,
    config: &TurboQuantConfig,
) -> VortexResult<TurboQuantQJLArray> {
    vortex_ensure!(
        fsl.dtype().nullability() == Nullability::NonNullable,
        "TurboQuant requires non-nullable input, got nullable FixedSizeListArray"
    );
    vortex_ensure!(
        config.bit_width >= 2 && config.bit_width <= 9,
        "QJL bit_width must be 2-9, got {}",
        config.bit_width
    );
    let dimension = fsl.list_size();
    vortex_ensure!(
        dimension >= 2,
        "TurboQuant requires dimension >= 2, got {dimension}"
    );

    let seed = config.seed.unwrap_or_else(rand::random);
    let dim = dimension as usize;
    let num_rows = fsl.len();
    let mse_bit_width = config.bit_width - 1;

    // First, encode the MSE inner at (bit_width - 1).
    let mse_config = TurboQuantConfig {
        bit_width: mse_bit_width,
        seed: Some(seed),
    };
    let mse_inner = turboquant_encode_mse(fsl, &mse_config)?;

    // TODO(perf): `turboquant_encode_mse` above already constructs the same
    // RotationMatrix from the same seed. Refactor to share it.
    let rotation = RotationMatrix::try_new(seed, dim)?;
    let padded_dim = rotation.padded_dim();

    if num_rows == 0 {
        return build_empty_qjl_array(fsl, config.bit_width, padded_dim, seed);
    }

    // TODO(perf): `turboquant_encode_mse` above already extracts f32 elements
    // internally. Refactor to share the buffer to avoid double materialization.
    let f32_elements = extract_f32_elements(fsl)?;
    #[allow(clippy::cast_possible_truncation)]
    let centroids = get_centroids(padded_dim as u32, mse_bit_width)?;
    let boundaries = compute_boundaries(&centroids);

    // QJL uses a different rotation than the MSE stage to ensure statistical
    // independence between the quantization noise and the sign projection.
    let qjl_rotation = RotationMatrix::try_new(seed.wrapping_add(1), dim)?;

    let mut residual_norms_buf = BufferMut::<f32>::with_capacity(num_rows);
    let total_sign_bits = num_rows * padded_dim;
    let mut qjl_sign_bits = BitBufferMut::new_unset(total_sign_bits);

    let mut padded = vec![0.0f32; padded_dim];
    let mut rotated = vec![0.0f32; padded_dim];
    let mut dequantized_rotated = vec![0.0f32; padded_dim];
    let mut dequantized = vec![0.0f32; padded_dim];
    let mut residual = vec![0.0f32; padded_dim];
    let mut projected = vec![0.0f32; padded_dim];

    for row in 0..num_rows {
        let x = &f32_elements[row * dim..(row + 1) * dim];
        let norm = l2_norm(x);

        // Reproduce the same quantization as MSE encoding.
        if norm > 0.0 {
            let inv_norm = 1.0 / norm;
            for (dst, &src) in padded[..dim].iter_mut().zip(x.iter()) {
                *dst = src * inv_norm;
            }
        } else {
            padded[..dim].fill(0.0);
        }
        rotation.rotate(&padded, &mut rotated);

        for j in 0..padded_dim {
            let idx = find_nearest_centroid(rotated[j], &boundaries);
            dequantized_rotated[j] = centroids[idx as usize];
        }

        rotation.inverse_rotate(&dequantized_rotated, &mut dequantized);
        if norm > 0.0 {
            for val in dequantized.iter_mut() {
                *val *= norm;
            }
        }

        // Compute residual: r = x - x̂. Only [..dim] is written; tail stays zero
        // from initialization and is never modified.
        for j in 0..dim {
            residual[j] = x[j] - dequantized[j];
        }
        let residual_norm = l2_norm(&residual[..dim]);
        residual_norms_buf.push(residual_norm);

        // QJL: sign(S · r). rotate() writes all of `projected` when called;
        // when residual_norm == 0 we must zero it since it has stale data.
        if residual_norm > 0.0 {
            qjl_rotation.rotate(&residual, &mut projected);
        } else {
            projected.fill(0.0);
        }

        let bit_offset = row * padded_dim;
        for j in 0..padded_dim {
            if projected[j] >= 0.0 {
                qjl_sign_bits.set(bit_offset + j);
            }
        }
    }

    let residual_norms_array =
        PrimitiveArray::new::<f32>(residual_norms_buf.freeze(), Validity::NonNullable);
    let qjl_signs = BoolArray::new(qjl_sign_bits.freeze(), Validity::NonNullable);
    let qjl_rotation_signs = qjl_rotation.export_inverse_signs_bool_array();

    #[allow(clippy::cast_possible_truncation)]
    TurboQuantQJLArray::try_new(
        fsl.dtype().clone(),
        mse_inner.into_array(),
        qjl_signs.into_array(),
        residual_norms_array.into_array(),
        qjl_rotation_signs.into_array(),
        config.bit_width,
        padded_dim as u32,
        seed.wrapping_add(1),
    )
}

fn build_empty_mse_array(
    fsl: &FixedSizeListArray,
    bit_width: u8,
    padded_dim: usize,
    seed: u64,
) -> VortexResult<TurboQuantMSEArray> {
    let rotation = RotationMatrix::try_new(seed, fsl.list_size() as usize)?;
    let codes = PrimitiveArray::empty::<u8>(fsl.dtype().nullability());
    let norms = PrimitiveArray::empty::<f32>(fsl.dtype().nullability());
    #[allow(clippy::cast_possible_truncation)]
    let centroids_vec = get_centroids(padded_dim as u32, bit_width)?;
    let mut centroids_buf = BufferMut::<f32>::with_capacity(centroids_vec.len());
    centroids_buf.extend_from_slice(&centroids_vec);
    let centroids = PrimitiveArray::new::<f32>(centroids_buf.freeze(), Validity::NonNullable);
    let rotation_signs = rotation.export_inverse_signs_bool_array();

    #[allow(clippy::cast_possible_truncation)]
    TurboQuantMSEArray::try_new(
        fsl.dtype().clone(),
        codes.into_array(),
        norms.into_array(),
        centroids.into_array(),
        rotation_signs.into_array(),
        fsl.list_size(),
        bit_width,
        padded_dim as u32,
        seed,
    )
}

fn build_empty_qjl_array(
    fsl: &FixedSizeListArray,
    bit_width: u8,
    padded_dim: usize,
    seed: u64,
) -> VortexResult<TurboQuantQJLArray> {
    let mse_config = TurboQuantConfig {
        bit_width: bit_width - 1,
        seed: Some(seed),
    };
    let mse_inner = turboquant_encode_mse(fsl, &mse_config)?;
    let qjl_rotation = RotationMatrix::try_new(seed.wrapping_add(1), fsl.list_size() as usize)?;
    let residual_norms = PrimitiveArray::empty::<f32>(fsl.dtype().nullability());
    let qjl_signs = BoolArray::new(BitBufferMut::new_unset(0).freeze(), Validity::NonNullable);
    let qjl_rotation_signs = qjl_rotation.export_inverse_signs_bool_array();

    #[allow(clippy::cast_possible_truncation)]
    TurboQuantQJLArray::try_new(
        fsl.dtype().clone(),
        mse_inner.into_array(),
        qjl_signs.into_array(),
        residual_norms.into_array(),
        qjl_rotation_signs.into_array(),
        bit_width,
        padded_dim as u32,
        seed.wrapping_add(1),
    )
}
