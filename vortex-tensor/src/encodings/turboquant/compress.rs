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

use crate::encodings::turboquant::array::TurboQuantData;
use crate::encodings::turboquant::centroids::compute_boundaries;
use crate::encodings::turboquant::centroids::find_nearest_centroid;
use crate::encodings::turboquant::centroids::get_centroids;
use crate::encodings::turboquant::rotation::RotationMatrix;

/// Configuration for TurboQuant encoding.
#[derive(Clone, Debug)]
pub struct TurboQuantConfig {
    /// Bits per coordinate (1-8).
    pub bit_width: u8,
    /// Optional seed for the rotation matrix. If None, the default seed is used.
    pub seed: Option<u64>,
}

impl Default for TurboQuantConfig {
    fn default() -> Self {
        Self {
            bit_width: 4,
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
struct QuantizationResult {
    rotation: RotationMatrix,
    centroids: Vec<f32>,
    all_indices: BufferMut<u8>,
    norms: BufferMut<f32>,
    padded_dim: usize,
}

/// Core quantization: extract f32 elements, build rotation, normalize/rotate/quantize all rows.
#[allow(clippy::cast_possible_truncation)]
fn turboquant_quantize_core(
    fsl: &FixedSizeListArray,
    seed: u64,
    bit_width: u8,
) -> VortexResult<QuantizationResult> {
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

    Ok(QuantizationResult {
        rotation,
        centroids,
        all_indices,
        norms,
        padded_dim,
    })
}

/// Build a `TurboQuantArray` from quantization results.
#[allow(clippy::cast_possible_truncation)]
fn build_turboquant(
    fsl: &FixedSizeListArray,
    core: QuantizationResult,
    bit_width: u8,
) -> VortexResult<TurboQuantData> {
    let dimension = fsl.list_size();

    let num_rows = fsl.len();
    let padded_dim = core.padded_dim;
    let codes_elements =
        PrimitiveArray::new::<u8>(core.all_indices.freeze(), Validity::NonNullable);
    let codes = FixedSizeListArray::try_new(
        codes_elements.into_array(),
        padded_dim as u32,
        Validity::NonNullable,
        num_rows,
    )?
    .into_array();
    let norms_array =
        PrimitiveArray::new::<f32>(core.norms.freeze(), Validity::NonNullable).into_array();

    // TODO(perf): `get_centroids` returns Vec<f32>; could avoid the copy by
    // supporting Buffer::from(Vec<T>) or caching as Buffer directly.
    let mut centroids_buf = BufferMut::<f32>::with_capacity(core.centroids.len());
    centroids_buf.extend_from_slice(&core.centroids);
    let centroids_array =
        PrimitiveArray::new::<f32>(centroids_buf.freeze(), Validity::NonNullable).into_array();

    let rotation_signs = bitpack_rotation_signs(&core.rotation)?;

    TurboQuantData::try_new(
        fsl.dtype().clone(),
        codes,
        norms_array,
        centroids_array,
        rotation_signs,
        dimension,
        bit_width,
    )
}

/// Encode a FixedSizeListArray into a `TurboQuantArray`.
///
/// The input must be non-nullable. TurboQuant is a lossy encoding that does not
/// preserve null positions; callers must handle validity externally.
pub fn turboquant_encode(
    fsl: &FixedSizeListArray,
    config: &TurboQuantConfig,
) -> VortexResult<ArrayRef> {
    vortex_ensure!(
        fsl.dtype().nullability() == Nullability::NonNullable,
        "TurboQuant requires non-nullable input, got nullable FixedSizeListArray"
    );
    vortex_ensure!(
        config.bit_width >= 1 && config.bit_width <= 8,
        "bit_width must be 1-8, got {}",
        config.bit_width
    );
    let dimension = fsl.list_size();
    vortex_ensure!(
        dimension >= 3,
        "TurboQuant requires dimension >= 3, got {dimension}"
    );

    if fsl.is_empty() {
        return Ok(fsl.clone().into_array());
    }

    let seed = config.seed.unwrap_or(42);
    let core = turboquant_quantize_core(fsl, seed, config.bit_width)?;

    Ok(build_turboquant(fsl, core, config.bit_width)?.into_array())
}

/// Export rotation signs as a 1-bit `BitPackedArray` for efficient storage.
///
/// The rotation matrix's 3 x padded_dim sign values are exported as 0/1 u8
/// values in inverse application order, then bitpacked to 1 bit per sign.
/// On decode, FastLanes SIMD-unpacks back to `&[u8]` of 0/1 values.
fn bitpack_rotation_signs(rotation: &RotationMatrix) -> VortexResult<ArrayRef> {
    let signs_u8 = rotation.export_inverse_signs_u8();
    let mut buf = BufferMut::<u8>::with_capacity(signs_u8.len());
    buf.extend_from_slice(&signs_u8);
    let prim = PrimitiveArray::new::<u8>(buf.freeze(), Validity::NonNullable);
    Ok(bitpack_encode(&prim, 1, None)?.into_array())
}
