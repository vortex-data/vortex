// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! TurboQuant encoding (quantization) logic.

use num_traits::ToPrimitive;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::Extension;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrays::fixed_size_list::FixedSizeListArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::match_each_float_ptype;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_fastlanes::bitpack_compress::bitpack_encode;

use crate::encodings::turboquant::TurboQuant;
use crate::encodings::turboquant::array::centroids::compute_centroid_boundaries;
use crate::encodings::turboquant::array::centroids::find_nearest_centroid;
use crate::encodings::turboquant::array::centroids::get_centroids;
use crate::encodings::turboquant::array::rotation::RotationMatrix;
use crate::encodings::turboquant::vtable::TurboQuantArray;
use crate::scalar_fns::ApproxOptions;
use crate::scalar_fns::l2_norm::L2Norm;

/// Configuration for TurboQuant encoding.
#[derive(Clone, Debug)]
pub struct TurboQuantConfig {
    /// Bits per coordinate (1-8).
    pub bit_width: u8,
    /// Optional seed for the rotation matrix. If None, the default seed is used.
    pub seed: Option<u64>,
    /// Number of sign-diagonal + WHT rounds in the structured rotation (default 3).
    pub num_rounds: u8,
}

impl Default for TurboQuantConfig {
    fn default() -> Self {
        Self {
            bit_width: TurboQuant::MAX_BIT_WIDTH,
            seed: Some(42),
            num_rounds: 3,
        }
    }
}

/// Extract elements from a FixedSizeListArray as a flat f32 PrimitiveArray for quantization.
///
/// All quantization (rotation, centroid lookup) happens in f32. f16 is upcast; f64 is truncated.
fn extract_f32_elements(
    fsl: &FixedSizeListArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    let elements = fsl.elements();
    let primitive = elements.clone().execute::<PrimitiveArray>(ctx)?;
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
            .map(|&v| {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "TurboQuant quantization operates in f32, so f64 inputs are intentionally downcast"
                )]
                let v = v as f32;
                v
            })
            .collect()),
        _ => vortex_bail!("TurboQuant requires float elements, got {ptype:?}"),
    }
}

/// Shared intermediate results from the quantization loop.
struct QuantizationResult {
    rotation: RotationMatrix,
    centroids: Vec<f32>,
    all_indices: BufferMut<u8>,
    /// Native-precision norms (matching the Vector element type). Carries validity: null vectors
    /// have null norms.
    norms_array: ArrayRef,
    padded_dim: usize,
}

/// Core quantization: compute norms via [`L2Norm`], extract f32 elements, then
/// normalize/rotate/quantize all rows.
///
/// Norms are computed in the native element precision via the [`L2Norm`] scalar function.
/// The rotation and centroid lookup happen in f32. Null rows (per the input validity) produce
/// all-zero codes.
fn turboquant_quantize_core(
    ext: ArrayView<Extension>,
    fsl: &FixedSizeListArray,
    seed: u64,
    bit_width: u8,
    num_rounds: u8,
    validity: &Validity,
    ctx: &mut ExecutionCtx,
) -> VortexResult<QuantizationResult> {
    let dimension =
        usize::try_from(fsl.list_size()).vortex_expect("u32 FixedSizeList dimension fits in usize");
    let num_rows = fsl.len();

    // Compute native-precision norms via the L2Norm scalar fn. L2Norm propagates validity from
    // the input, so null vectors get null norms automatically.
    let norms_sfn = L2Norm::try_new_array(&ApproxOptions::Exact, ext.as_ref().clone(), num_rows)?;
    let norms_array: ArrayRef = norms_sfn.into_array().execute(ctx)?;
    let norms_prim: PrimitiveArray = norms_array.clone().execute(ctx)?;

    // Extract f32 norms for the internal quantization loop.
    let f32_norms: Vec<f32> = match_each_float_ptype!(norms_prim.ptype(), |T| {
        norms_prim
            .as_slice::<T>()
            .iter()
            .map(|&v| {
                // `ToPrimitive::to_f32` is infallible for all float types: f16 -> f32 is lossless,
                // f32 is identity, and f64 -> f32 saturates to +-inf.
                ToPrimitive::to_f32(&v).vortex_expect("float-to-f32 conversion is infallible")
            })
            .collect()
    });

    let rotation = RotationMatrix::try_new(seed, dimension, num_rounds as usize)?;
    let padded_dim = rotation.padded_dim();
    let padded_dim_u32 =
        u32::try_from(padded_dim).vortex_expect("padded_dim stays representable as u32");

    let f32_elements = extract_f32_elements(fsl, ctx)?;

    let centroids = get_centroids(padded_dim_u32, bit_width)?;
    let boundaries = compute_centroid_boundaries(&centroids);

    let mut all_indices = BufferMut::<u8>::with_capacity(num_rows * padded_dim);
    let mut padded = vec![0.0f32; padded_dim];
    let mut rotated = vec![0.0f32; padded_dim];

    let f32_slice = f32_elements.as_slice::<f32>();
    for row in 0..num_rows {
        // Null vectors get all-zero codes.
        if !validity.is_valid(row)? {
            all_indices.extend(std::iter::repeat_n(0u8, padded_dim));
            continue;
        }

        let x = &f32_slice[row * dimension..(row + 1) * dimension];
        let norm = f32_norms[row];

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
        norms_array,
        padded_dim,
    })
}

/// Build a `TurboQuantArray` from quantization results.
fn build_turboquant(
    fsl: &FixedSizeListArray,
    core: QuantizationResult,
    ext_dtype: DType,
) -> VortexResult<TurboQuantArray> {
    let num_rows = fsl.len();
    let padded_dim = core.padded_dim;
    let padded_dim_u32 =
        u32::try_from(padded_dim).vortex_expect("padded_dim stays representable as u32");
    let codes_elements =
        PrimitiveArray::new::<u8>(core.all_indices.freeze(), Validity::NonNullable);
    let codes = FixedSizeListArray::try_new(
        codes_elements.into_array(),
        padded_dim_u32,
        Validity::NonNullable,
        num_rows,
    )?
    .into_array();

    // TODO(perf): `get_centroids` returns Vec<f32>; could avoid the copy by
    // supporting Buffer::from(Vec<T>) or caching as Buffer directly.
    let mut centroids_buf = BufferMut::<f32>::with_capacity(core.centroids.len());
    centroids_buf.extend_from_slice(&core.centroids);
    let centroids_array =
        PrimitiveArray::new::<f32>(centroids_buf.freeze(), Validity::NonNullable).into_array();

    let rotation_signs = bitpack_rotation_signs(&core.rotation)?;

    TurboQuant::try_new_array(
        ext_dtype,
        codes,
        core.norms_array,
        centroids_array,
        rotation_signs,
    )
}

/// Encode a [`Vector`](crate::vector::Vector) extension array into a `TurboQuantArray`.
///
/// Nullable inputs are supported: null vectors get all-zero codes and null norms. The validity
/// of the resulting TurboQuant array is carried by the norms child.
pub fn turboquant_encode(
    ext: ArrayView<Extension>,
    config: &TurboQuantConfig,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let ext_dtype = ext.dtype().clone();
    let storage = ext.storage_array();
    let fsl = storage.clone().execute::<FixedSizeListArray>(ctx)?;

    vortex_ensure!(
        config.bit_width >= 1 && config.bit_width <= TurboQuant::MAX_BIT_WIDTH,
        "bit_width must be 1-{}, got {}",
        TurboQuant::MAX_BIT_WIDTH,
        config.bit_width
    );
    let dimension = fsl.list_size();
    vortex_ensure!(
        dimension >= TurboQuant::MIN_DIMENSION,
        "TurboQuant requires dimension >= {}, got {dimension}",
        TurboQuant::MIN_DIMENSION
    );

    if fsl.is_empty() {
        let padded_dim = dimension.next_power_of_two();
        let empty_codes = FixedSizeListArray::try_new(
            PrimitiveArray::empty::<u8>(Nullability::NonNullable).into_array(),
            padded_dim,
            Validity::NonNullable,
            0,
        )?;

        // Norms dtype matches the element type and carries the parent's nullability.
        let element_ptype = fsl.elements().dtype().as_ptype();
        let norms_nullability = ext_dtype.nullability();
        let empty_norms: ArrayRef = match_each_float_ptype!(element_ptype, |T| {
            PrimitiveArray::empty::<T>(norms_nullability).into_array()
        });

        let empty_centroids = PrimitiveArray::empty::<f32>(Nullability::NonNullable);
        let empty_signs = FixedSizeListArray::try_new(
            PrimitiveArray::empty::<u8>(Nullability::NonNullable).into_array(),
            padded_dim,
            Validity::NonNullable,
            0,
        )?;
        return Ok(TurboQuant::try_new_array(
            ext_dtype,
            empty_codes.into_array(),
            empty_norms,
            empty_centroids.into_array(),
            empty_signs.into_array(),
        )?
        .into_array());
    }

    let validity = ext.as_ref().validity()?;
    let seed = config.seed.unwrap_or(42);
    let core = turboquant_quantize_core(
        ext,
        &fsl,
        seed,
        config.bit_width,
        config.num_rounds,
        &validity,
        ctx,
    )?;

    Ok(build_turboquant(&fsl, core, ext_dtype)?.into_array())
}

/// Export rotation signs as a `FixedSizeListArray` wrapping a 1-bit [`BitPackedArray`].
///
/// The rotation matrix's `num_rounds * padded_dim` sign values are exported as 0/1 u8 values in
/// inverse application order, bitpacked to 1 bit per sign, then wrapped in a
/// `FixedSizeListArray` with `list_size = padded_dim` and `len = num_rounds`.
fn bitpack_rotation_signs(rotation: &RotationMatrix) -> VortexResult<ArrayRef> {
    let signs_u8 = rotation.export_inverse_signs_u8();
    let num_rounds = rotation.num_rounds();
    let padded_dim = u32::try_from(rotation.padded_dim()).vortex_expect("padded_dim fits in u32");

    let mut buf = BufferMut::<u8>::with_capacity(signs_u8.len());
    buf.extend_from_slice(&signs_u8);
    let prim = PrimitiveArray::new::<u8>(buf.freeze(), Validity::NonNullable);
    let bitpacked = bitpack_encode(&prim, 1, None)?;

    let fsl = FixedSizeListArray::try_new(
        bitpacked.into_array(),
        padded_dim,
        Validity::NonNullable,
        num_rounds,
    )?;
    Ok(fsl.into_array())
}
