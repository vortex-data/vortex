// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! TurboQuant encoding (quantization) logic.

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_fastlanes::bitpack_compress::bitpack_encode;

use crate::array::QjlCorrection;
use crate::array::TurboQuantArray;
use crate::centroids::compute_boundaries;
use crate::centroids::find_nearest_centroid;
use crate::centroids::get_centroids;
use crate::rotation::RotationMatrix;

/// Configuration for TurboQuant encoding.
#[derive(Clone, Debug)]
pub struct TurboQuantConfig {
    /// Bits per coordinate.
    ///
    /// For MSE encoding: 1-8.
    /// For QJL encoding: 2-9 (the MSE component uses `bit_width - 1`).
    pub bit_width: u8,
    /// Optional seed for the rotation matrix. If None, the default seed is used.
    pub seed: Option<u64>,
}

impl Default for TurboQuantConfig {
    fn default() -> Self {
        Self {
            bit_width: 5,
            seed: Some(42),
        }
    }
}

/// Extract elements from a FixedSizeListArray as a flat f32 PrimitiveArray.
#[allow(clippy::cast_possible_truncation)]
fn extract_f32_elements(fsl: &FixedSizeListArray) -> VortexResult<PrimitiveArray> {
    let elements = fsl.elements();
    let primitive = elements.to_canonical()?.into_primitive();
    let ptype = primitive.ptype();

    match ptype {
        PType::F16 => Ok(primitive
            .as_slice::<half::f16>()
            .iter()
            .map(|&v| f32::from(v))
            .collect()),
        PType::F32 => Ok(primitive),
        PType::F64 => Ok(primitive
            .as_slice::<f64>()
            .iter()
            .map(|&v| v as f32)
            .collect()),
        _ => vortex_bail!("TurboQuant requires float elements, got {ptype:?}"),
    }
}

/// Compute the L2 norm of a vector.
#[inline]
fn l2_norm(x: &[f32]) -> f32 {
    x.iter().map(|&v| v * v).sum::<f32>().sqrt()
}

/// Shared intermediate results from the MSE quantization loop.
struct MseQuantizationResult {
    rotation: RotationMatrix,
    f32_elements: PrimitiveArray,
    centroids: Vec<f32>,
    all_indices: BufferMut<u8>,
    norms: BufferMut<f32>,
    padded_dim: usize,
}

/// Core quantization: extract f32 elements, build rotation, normalize/rotate/quantize all rows.
fn turboquant_quantize_core(
    fsl: &FixedSizeListArray,
    seed: u64,
    bit_width: u8,
) -> VortexResult<MseQuantizationResult> {
    let dimension = fsl.list_size() as usize;
    let num_rows = fsl.len();

    let rotation = RotationMatrix::try_new(seed, dimension)?;
    let padded_dim = rotation.padded_dim();

    let f32_elements = extract_f32_elements(fsl)?;

    let centroids = get_centroids(padded_dim as u32, bit_width)?;
    let boundaries = compute_boundaries(&centroids);

    let mut all_indices = BufferMut::<u8>::with_capacity(num_rows * padded_dim);
    let mut norms = BufferMut::<f32>::with_capacity(num_rows);
    let mut padded = vec![0.0f32; padded_dim];
    let mut rotated = vec![0.0f32; padded_dim];

    let f32_slice = f32_elements.as_slice::<f32>();
    for row in 0..num_rows {
        let x = &f32_slice[row * dimension..(row + 1) * dimension];
        let norm = l2_norm(x);
        norms.push(norm);

        if norm > 0.0 {
            let inv_norm = 1.0 / norm;
            for (dst, &src) in padded[..dimension].iter_mut().zip(x.iter()) {
                *dst = src * inv_norm;
            }
        } else {
            padded[..dimension].fill(0.0);
        }
        rotation.rotate(&padded, &mut rotated);

        for j in 0..padded_dim {
            all_indices.push(find_nearest_centroid(rotated[j], &boundaries));
        }
    }

    Ok(MseQuantizationResult {
        rotation,
        f32_elements,
        centroids,
        all_indices,
        norms,
        padded_dim,
    })
}

/// Build a `TurboQuantArray` (MSE-only) from quantization results.
fn build_turboquant_mse(
    dtype: &FixedSizeListArray,
    core: MseQuantizationResult,
    bit_width: u8,
) -> VortexResult<TurboQuantArray> {
    let dimension = dtype.list_size();

    let codes =
        PrimitiveArray::new::<u8>(core.all_indices.freeze(), Validity::NonNullable).into_array();
    let norms_array =
        PrimitiveArray::new::<f32>(core.norms.freeze(), Validity::NonNullable).into_array();

    // TODO(perf): `get_centroids` returns Vec<f32>; could avoid the copy by
    // supporting Buffer::from(Vec<T>) or caching as Buffer directly.
    let mut centroids_buf = BufferMut::<f32>::with_capacity(core.centroids.len());
    centroids_buf.extend_from_slice(&core.centroids);
    let centroids_array =
        PrimitiveArray::new::<f32>(centroids_buf.freeze(), Validity::NonNullable).into_array();

    let rotation_signs = bitpack_rotation_signs(&core.rotation)?;

    TurboQuantArray::try_new_mse(
        dtype.dtype().clone(),
        codes,
        norms_array,
        centroids_array,
        rotation_signs,
        dimension,
        bit_width,
    )
}

/// Encode a FixedSizeListArray into a MSE-only `TurboQuantArray`.
///
/// The input must be non-nullable. TurboQuant is a lossy encoding that does not
/// preserve null positions; callers must handle validity externally.
pub fn turboquant_encode_mse(
    fsl: &FixedSizeListArray,
    config: &TurboQuantConfig,
) -> VortexResult<ArrayRef> {
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

    if fsl.is_empty() {
        return Ok(fsl.clone().into_array());
    }

    let seed = config.seed.unwrap_or(42);
    let core = turboquant_quantize_core(fsl, seed, config.bit_width)?;

    Ok(build_turboquant_mse(fsl, core, config.bit_width)?.into_array())
}

/// Encode a FixedSizeListArray into a `TurboQuantArray` with QJL correction.
///
/// The QJL variant uses `bit_width - 1` MSE bits plus 1 bit of QJL residual
/// correction, giving unbiased inner product estimation. The input must be
/// non-nullable.
pub fn turboquant_encode_qjl(
    fsl: &FixedSizeListArray,
    config: &TurboQuantConfig,
) -> VortexResult<ArrayRef> {
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

    if fsl.is_empty() {
        return Ok(fsl.clone().into_array());
    }

    let seed = config.seed.unwrap_or(42);
    let dim = dimension as usize;
    let mse_bit_width = config.bit_width - 1;

    let core = turboquant_quantize_core(fsl, seed, mse_bit_width)?;
    let padded_dim = core.padded_dim;

    // QJL uses a different rotation than the MSE stage to ensure statistical
    // independence between the quantization noise and the sign projection.
    let qjl_rotation = RotationMatrix::try_new(seed.wrapping_add(25), dim)?;

    let num_rows = fsl.len();
    let mut residual_norms_buf = BufferMut::<f32>::with_capacity(num_rows);
    let mut qjl_sign_u8 = BufferMut::<u8>::with_capacity(num_rows * padded_dim);

    let mut dequantized_rotated = vec![0.0f32; padded_dim];
    let mut dequantized = vec![0.0f32; padded_dim];
    let mut residual = vec![0.0f32; padded_dim];
    let mut projected = vec![0.0f32; padded_dim];

    // Compute QJL residuals using precomputed indices and norms from the core.
    {
        let f32_slice = core.f32_elements.as_slice::<f32>();
        let indices_slice: &[u8] = &core.all_indices;
        let norms_slice: &[f32] = &core.norms;

        for row in 0..num_rows {
            let x = &f32_slice[row * dim..(row + 1) * dim];
            let norm = norms_slice[row];

            // Dequantize from precomputed indices.
            let row_indices = &indices_slice[row * padded_dim..(row + 1) * padded_dim];
            for j in 0..padded_dim {
                dequantized_rotated[j] = core.centroids[row_indices[j] as usize];
            }

            core.rotation
                .inverse_rotate(&dequantized_rotated, &mut dequantized);
            if norm > 0.0 {
                for val in dequantized[..dim].iter_mut() {
                    *val *= norm;
                }
            }

            // Compute residual: r = x - x̂.
            for j in 0..dim {
                residual[j] = x[j] - dequantized[j];
            }
            let residual_norm = l2_norm(&residual[..dim]);
            residual_norms_buf.push(residual_norm);

            // QJL: sign(S · r).
            if residual_norm > 0.0 {
                qjl_rotation.rotate(&residual, &mut projected);
            } else {
                projected.fill(0.0);
            }

            for j in 0..padded_dim {
                qjl_sign_u8.push(if projected[j] >= 0.0 { 1u8 } else { 0u8 });
            }
        }
    }

    // Build the MSE part.
    let mut array = build_turboquant_mse(fsl, core, mse_bit_width)?;

    // Attach QJL correction.
    let residual_norms_array =
        PrimitiveArray::new::<f32>(residual_norms_buf.freeze(), Validity::NonNullable);
    let qjl_signs_prim = PrimitiveArray::new::<u8>(qjl_sign_u8.freeze(), Validity::NonNullable);
    let qjl_signs_packed = bitpack_encode(&qjl_signs_prim, 1, None)?.into_array();
    let qjl_rotation_signs = bitpack_rotation_signs(&qjl_rotation)?;

    array.qjl = Some(QjlCorrection {
        signs: qjl_signs_packed,
        residual_norms: residual_norms_array.into_array(),
        rotation_signs: qjl_rotation_signs,
    });

    Ok(array.into_array())
}

/// Export rotation signs as a 1-bit `BitPackedArray` for efficient storage.
///
/// The rotation matrix's 3 × padded_dim sign values are exported as 0/1 u8
/// values in inverse application order, then bitpacked to 1 bit per sign.
/// On decode, FastLanes SIMD-unpacks back to `&[u8]` of 0/1 values.
fn bitpack_rotation_signs(rotation: &RotationMatrix) -> VortexResult<ArrayRef> {
    let signs_u8 = rotation.export_inverse_signs_u8();
    let mut buf = BufferMut::<u8>::with_capacity(signs_u8.len());
    buf.extend_from_slice(&signs_u8);
    let prim = PrimitiveArray::new::<u8>(buf.freeze(), Validity::NonNullable);
    Ok(bitpack_encode(&prim, 1, None)?.into_array())
}
