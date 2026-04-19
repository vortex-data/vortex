// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! TurboQuant encoding (quantization) logic.
//!
//! The input to [`turboquant_encode`] must be a non-nullable [`Vector`](crate::vector::Vector)
//! extension array whose rows are already L2-normalized (unit norm). Normalization is handled
//! externally by [`normalize_as_l2_denorm`](crate::scalar_fns::l2_denorm::normalize_as_l2_denorm),
//! which the [`TurboQuantScheme`] calls before invoking this function.
//!
//! [`TurboQuantScheme`]: crate::encodings::turboquant::TurboQuantScheme

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::Extension;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::dict::DictArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrays::fixed_size_list::FixedSizeListArrayExt;
use vortex_array::arrays::scalar_fn::ScalarFnArrayExt;
use vortex_array::dtype::Nullability;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::encodings::turboquant::MAX_BIT_WIDTH;
use crate::encodings::turboquant::MIN_DIMENSION;
use crate::encodings::turboquant::centroids::compute_centroid_boundaries;
use crate::encodings::turboquant::centroids::find_nearest_centroid;
use crate::encodings::turboquant::centroids::get_centroids;
use crate::scalar_fns::l2_denorm::L2Denorm;
use crate::scalar_fns::l2_denorm::normalize_as_l2_denorm;
use crate::scalar_fns::sorf_transform::SorfMatrix;
use crate::scalar_fns::sorf_transform::SorfOptions;
use crate::scalar_fns::sorf_transform::SorfTransform;
use crate::utils::cast_to_f32;
use crate::vector::AnyVector;
use crate::vector::Vector;

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
            bit_width: MAX_BIT_WIDTH,
            seed: Some(42),
            num_rounds: 3,
        }
    }
}

/// Apply the full TurboQuant compression pipeline to a [`Vector`](crate::vector::Vector)
/// extension array: normalize the rows via [`normalize_as_l2_denorm`], quantize the normalized
/// child via [`turboquant_encode_unchecked`], and reattach the stored norms as the outer
/// [`L2Denorm`] wrapper.
///
/// The returned array has the canonical TurboQuant shape:
///
/// ```text
/// ScalarFnArray(L2Denorm, [
///     ScalarFnArray(SorfTransform, [FSL(Dict(codes, centroids))]),
///     norms,
/// ])
/// ```
///
/// # Errors
///
/// Returns an error if `input` is not a tensor-like extension array, if normalization fails, or
/// if [`turboquant_encode_unchecked`] rejects the input shape.
pub fn turboquant_encode(
    input: ArrayRef,
    config: &TurboQuantConfig,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    // We must normalize the array before we can encode it with TurboQuant.
    let l2_denorm = normalize_as_l2_denorm(input, ctx)?;
    let normalized = l2_denorm.child_at(0).clone();
    let norms = l2_denorm.child_at(1).clone();
    let num_rows = l2_denorm.len();

    let normalized_ext = normalized
        .as_opt::<Extension>()
        .vortex_expect("normalize_as_l2_denorm always produces an Extension array child");

    // SAFETY: `normalize_as_l2_denorm` guarantees every row is unit-norm (or zero for null rows).
    let tq = unsafe { turboquant_encode_unchecked(normalized_ext, config, ctx) }?;

    // SAFETY: TurboQuant is a lossy approximation of the normalized child, so we intentionally
    // bypass the strict normalized-row validation when reattaching the stored norms.
    Ok(unsafe { L2Denorm::new_array_unchecked(tq, norms, num_rows) }?.into_array())
}

/// Encode a non-nullable, L2-normalized [`Vector`](crate::vector::Vector) extension array into a
/// `ScalarFnArray(SorfTransform, [FSL(Dict(codes, centroids))])`, without validating the unit-norm
/// precondition.
///
/// # Safety
///
/// The caller must ensure:
///
/// - The input dtype is non-nullable.
/// - Every row is L2-normalized (unit norm) or is a zero vector.
///
/// Passing non-unit-norm vectors will not cause memory unsafety, but will produce silently
/// incorrect quantization results.
pub unsafe fn turboquant_encode_unchecked(
    ext: ArrayView<Extension>,
    config: &TurboQuantConfig,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let ext_dtype = ext.dtype().clone();
    let storage = ext.storage_array();
    let fsl = storage.clone().execute::<FixedSizeListArray>(ctx)?;

    vortex_ensure!(
        config.bit_width >= 1 && config.bit_width <= MAX_BIT_WIDTH,
        "bit_width must be 1-{MAX_BIT_WIDTH}, got {}",
        config.bit_width
    );
    let dimension = fsl.list_size();
    vortex_ensure!(
        dimension >= MIN_DIMENSION,
        "TurboQuant requires dimension >= {MIN_DIMENSION}, got {dimension}",
    );

    let vector_metadata = ext_dtype.as_extension().metadata::<AnyVector>();
    let element_ptype = vector_metadata.element_ptype();

    let seed = config.seed.unwrap_or(42);
    let num_rows = fsl.len();

    if fsl.is_empty() {
        let padded_dim = dimension.next_power_of_two();
        let empty_codes = PrimitiveArray::empty::<u8>(Nullability::NonNullable);
        let empty_centroids = PrimitiveArray::empty::<f32>(Nullability::NonNullable);
        let empty_dict =
            DictArray::try_new(empty_codes.into_array(), empty_centroids.into_array())?;
        let empty_fsl = FixedSizeListArray::try_new(
            empty_dict.into_array(),
            padded_dim,
            Validity::NonNullable,
            0,
        )?;
        let empty_padded_vector = Vector::try_new_vector_array(empty_fsl.into_array())?;

        let sorf_options = SorfOptions {
            seed,
            num_rounds: config.num_rounds,
            dimension,
            element_ptype,
        };
        return Ok(
            SorfTransform::try_new_array(&sorf_options, empty_padded_vector, 0)?.into_array(),
        );
    }

    let core = turboquant_quantize_core(&fsl, seed, config.bit_width, config.num_rounds, ctx)?;
    let quantized_fsl =
        build_quantized_fsl(num_rows, core.all_indices, core.centroids, core.padded_dim)?;
    let padded_vector = Vector::try_new_vector_array(quantized_fsl)?;

    let sorf_options = SorfOptions {
        seed,
        num_rounds: config.num_rounds,
        dimension,
        element_ptype,
    };
    Ok(SorfTransform::try_new_array(&sorf_options, padded_vector, num_rows)?.into_array())
}

/// Shared intermediate results from the quantization loop.
struct QuantizationResult {
    centroids: Buffer<f32>,
    all_indices: Buffer<u8>,
    padded_dim: usize,
}

/// Core quantization: rotate and quantize already-normalized rows.
///
/// The input `fsl` must contain non-nullable, unit-norm vectors (already L2-normalized). Null
/// vectors are not supported and must be zeroed out before reaching this function. The rotation
/// and centroid lookup happen in f32.
fn turboquant_quantize_core(
    fsl: &FixedSizeListArray,
    seed: u64,
    bit_width: u8,
    num_rounds: u8,
    ctx: &mut ExecutionCtx,
) -> VortexResult<QuantizationResult> {
    let dimension = fsl.list_size() as usize;
    let num_rows = fsl.len();

    let rotation = SorfMatrix::try_new(seed, dimension, num_rounds as usize)?;
    let padded_dim = rotation.padded_dim();
    let padded_dim_u32 =
        u32::try_from(padded_dim).vortex_expect("padded_dim stays representable as u32");

    let elements_prim: PrimitiveArray = fsl.elements().clone().execute(ctx)?;
    let f32_elements = cast_to_f32(elements_prim)?;

    let centroids = get_centroids(padded_dim_u32, bit_width)?;
    let boundaries = compute_centroid_boundaries(&centroids);

    let mut all_indices = BufferMut::<u8>::with_capacity(num_rows * padded_dim);
    let mut padded = vec![0.0f32; padded_dim];
    let mut rotated = vec![0.0f32; padded_dim];

    let f32_slice = f32_elements.as_slice();
    for row in 0..num_rows {
        let x = &f32_slice[row * dimension..(row + 1) * dimension];

        // Zero-pad to the next power of 2.
        padded[..dimension].copy_from_slice(x);
        padded[dimension..].fill(0.0);

        rotation.rotate(&padded, &mut rotated);

        for j in 0..padded_dim {
            all_indices.push(find_nearest_centroid(rotated[j], &boundaries));
        }
    }

    Ok(QuantizationResult {
        centroids,
        all_indices: all_indices.freeze(),
        padded_dim,
    })
}

/// Build a quantized representation: `FSL(DictArray(codes, centroids), padded_dim)`.
///
/// This is a Dict-encoded FixedSizeList where each row of `padded_dim` u8 codes indexes into the
/// centroid codebook. The Dict can be independently sliced, taken, or executed (dequantized)
/// without knowledge of the rotation.
fn build_quantized_fsl(
    num_rows: usize,
    all_indices: Buffer<u8>,
    centroids: Buffer<f32>,
    padded_dim: usize,
) -> VortexResult<ArrayRef> {
    let codes = PrimitiveArray::new::<u8>(all_indices, Validity::NonNullable);
    let centroids_array = PrimitiveArray::new::<f32>(centroids, Validity::NonNullable);

    let dict = DictArray::try_new(codes.into_array(), centroids_array.into_array())?;

    let padded_dim_u32 =
        u32::try_from(padded_dim).vortex_expect("padded_dim stays representable as u32");
    Ok(FixedSizeListArray::try_new(
        dict.into_array(),
        padded_dim_u32,
        Validity::NonNullable,
        num_rows,
    )?
    .into_array())
}
