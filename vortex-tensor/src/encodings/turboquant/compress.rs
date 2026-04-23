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
use vortex_array::arrays::ExtensionArray;
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
use crate::encodings::turboquant::centroids::compute_or_get_centroids;
use crate::encodings::turboquant::centroids::find_nearest_centroid;
use crate::normalized_vector::AnyNormalizedVector;
use crate::scalar_fns::l2_denorm::L2Denorm;
use crate::scalar_fns::l2_denorm::normalize_as_l2_denorm;
use crate::scalar_fns::sorf_transform::SorfMatrix;
use crate::scalar_fns::sorf_transform::SorfOptions;
use crate::scalar_fns::sorf_transform::SorfTransform;
use crate::types::normalized_vector::NormalizedVector;
use crate::utils::cast_to_f32;

/// Configuration for TurboQuant encoding.
#[derive(Clone, Debug)]
pub struct TurboQuantConfig {
    /// Bits per coordinate (1-8).
    pub bit_width: u8,
    /// Seed for the rotation matrix.
    pub seed: u64,
    /// Number of sign-diagonal + WHT rounds in the structured rotation (default 3).
    pub num_rounds: u8,
}

impl Default for TurboQuantConfig {
    fn default() -> Self {
        Self {
            bit_width: MAX_BIT_WIDTH,
            seed: 42,
            num_rounds: 3,
        }
    }
}

/// Apply the full TurboQuant compression pipeline to a [`Vector`](crate::vector::Vector)
/// extension array: normalize the rows via [`normalize_as_l2_denorm`], quantize the normalized
/// child via [`turboquant_encode_normalized`], and reattach the stored norms as the outer
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
/// Returns an error if `input` is not a tensor-like extension array, if normalization fails, or if
/// [`turboquant_encode_normalized`] rejects the input shape.
pub fn turboquant_encode(
    input: ArrayRef,
    config: &TurboQuantConfig,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    // We must normalize the array before we can encode it with TurboQuant.
    let l2_denorm = normalize_as_l2_denorm(input, ctx)?;

    // This is guaranteed to be a `NormalizedVector` extension type.
    let normalized = l2_denorm.child_at(0).clone();
    let norms = l2_denorm.child_at(1).clone();
    let num_rows = l2_denorm.len();

    let normalized_ext = normalized
        .as_opt::<Extension>()
        .vortex_expect("normalize_as_l2_denorm always produces an Extension array child");

    let tq = turboquant_encode_normalized(normalized_ext, config, ctx)?;

    // SAFETY: TurboQuant is a lossy approximation of the normalized child, so we intentionally
    // bypass the strict normalized-row and zero-row validation when reattaching the stored norms.
    Ok(unsafe { L2Denorm::new_array_unchecked(tq, norms, num_rows) }?.into_array())
}

/// Encode a non-nullable [`NormalizedVector`](crate::normalized_vector::NormalizedVector)
/// extension array into
/// a `ScalarFnArray(SorfTransform, [FSL(Dict(codes, centroids))])`, without validating the
/// unit-norm precondition.
///
/// Passing non-unit-norm vectors will not cause memory unsafety, but will produce silently
/// incorrect quantization results.
pub fn turboquant_encode_normalized(
    ext: ArrayView<Extension>,
    config: &TurboQuantConfig,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let ext_dtype = ext.dtype().clone();

    let vector_metadata = ext_dtype.as_extension().metadata::<AnyNormalizedVector>();
    let element_ptype = vector_metadata.element_ptype();
    let dimensions = vector_metadata.dimensions();

    // `NormalizedVector` storage is `Extension(Vector(FSL))`; drill past the inner `Vector` to
    // reach the underlying `FixedSizeList`.
    let inner_vector: ExtensionArray = ext.storage_array().clone().execute(ctx)?;
    let fsl: FixedSizeListArray = inner_vector.storage_array().clone().execute(ctx)?;

    vortex_ensure!(
        config.bit_width >= 1 && config.bit_width <= MAX_BIT_WIDTH,
        "bit_width must be 1-{MAX_BIT_WIDTH}, got {}",
        config.bit_width
    );
    vortex_ensure!(
        dimensions >= MIN_DIMENSION,
        "TurboQuant requires dimension >= {MIN_DIMENSION}, got {dimensions}",
    );

    let num_rows = fsl.len();
    let sorf_options = SorfOptions {
        seed: config.seed,
        num_rounds: config.num_rounds,
        dimensions,
        element_ptype,
    };

    if fsl.is_empty() {
        let padded_dim = dimensions.next_power_of_two();
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
        // SAFETY: An empty FSL contains no rows, so the unit-norm-or-zero invariant holds
        // vacuously.
        let empty_padded_vector =
            unsafe { NormalizedVector::new_unchecked(empty_fsl.into_array()) }?;

        return Ok(
            SorfTransform::try_new_array(&sorf_options, empty_padded_vector, 0)?.into_array(),
        );
    }

    let quantized_fsl = turboquant_quantize_fsl(&fsl, config.bit_width, &sorf_options, ctx)?;

    // NB: The quantized rows are approximately unit-norm by construction; downstream callers
    // (notably the enclosing `L2Denorm` wrapper) treat the stored-norm + NormalizedVector claim as
    // authoritative rather than decode-verified.

    // SAFETY: TurboQuant is a lossy approximation of the already-unit-norm input.
    let padded_vector = unsafe { NormalizedVector::new_unchecked(quantized_fsl) }?;

    Ok(SorfTransform::try_new_array(&sorf_options, padded_vector, num_rows)?.into_array())
}

/// Rotate and quantize already-normalized rows into a dict-encoded `FixedSizeList`.
///
/// The input `fsl` must contain non-nullable, unit-norm vectors of float values (already
/// L2-normalized). Null vectors are not supported and must be zeroed out before reaching this
/// function. The rotation and centroid lookup happen in f32.
///
/// The returned array is `FSL(DictArray(codes, centroids), padded_dim)`. The `FixedSizeList` has
/// Dict-encoded elements, where each row of `padded_dim` u8 codes indexes into the centroid
/// codebook.
///
/// This allows the FSL (via the Dict-encodede elements) to be independently sliced, taken, or
/// executed (dequantized) without knowledge of the rotation.
///
/// Internally, this function will:
///
/// 1. Builds a [`SorfMatrix`] structured rotation from the seed/rounds in `sorf_options`.
/// 2. For each row, zero-pads to the next power of 2, applies the rotation, and maps each rotated
///    coordinate to its nearest centroid index via binary search on precomputed boundaries.
/// 3. Packs the per-row centroid indices and the shared centroid codebook into a `DictArray`-backed
///    `FixedSizeListArray`.
fn turboquant_quantize_fsl(
    fsl: &FixedSizeListArray,
    bit_width: u8,
    sorf_options: &SorfOptions,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let dimensions = fsl.list_size() as usize;
    let num_rows = fsl.len();

    vortex_ensure!(!fsl.dtype().is_nullable());

    let rotation = SorfMatrix::try_new(
        sorf_options.seed,
        dimensions,
        sorf_options.num_rounds as usize,
    )?;
    let padded_dim = rotation.padded_dim();
    let padded_dim_u32 =
        u32::try_from(padded_dim).vortex_expect("padded_dim stays representable as u32");

    // Compute the centroids for the given (dimension, bit_width) combination (or retrieve it from a
    // previous computation)
    let centroids = compute_or_get_centroids(padded_dim_u32, bit_width)?;

    // Extract out the elements of the FSL and cast to f32. In the f64 case, we intentionally lose
    // information here because we are already going to be quantizing to a smaller set of centroids,
    // so we are fine with this loss.
    let elements_prim: PrimitiveArray = fsl.elements().clone().execute(ctx)?;
    let f32_elements = cast_to_f32(elements_prim)?;

    // Take the float values and quantize by finding the closest centroid in the codebook to each
    // and recording the index of that centroid.
    let all_indices = rotate_and_quantize(
        f32_elements.as_slice(),
        num_rows,
        dimensions,
        &rotation,
        &centroids,
    );

    // Build the Dict-encoded FSL from the centroid indices and codebook. Everything is non-null
    // since our input in non-null.
    let codes = PrimitiveArray::new::<u8>(all_indices, Validity::NonNullable);
    let values = PrimitiveArray::new::<f32>(centroids, Validity::NonNullable);
    let dict = DictArray::try_new(codes.into_array(), values.into_array())?;

    Ok(FixedSizeListArray::try_new(
        dict.into_array(),
        padded_dim_u32,
        Validity::NonNullable,
        num_rows,
    )?
    .into_array())
}

/// Rotate each row via the structured rotation and quantize every rotated coordinate to its nearest
/// centroid index via binary search on precomputed boundaries.
///
/// Returns a flat [`Buffer<u8>`] of length `num_rows * padded_dim` containing the per-coordinate
/// centroid indices.
fn rotate_and_quantize(
    f32_slice: &[f32],
    num_rows: usize,
    dimensions: usize,
    rotation: &SorfMatrix,
    centroids: &[f32],
) -> Buffer<u8> {
    let padded_dim = rotation.padded_dim();
    let boundaries = compute_centroid_boundaries(centroids);

    let mut all_indices = BufferMut::<u8>::with_capacity(num_rows * padded_dim);
    let mut padded = vec![0.0f32; padded_dim];
    let mut rotated = vec![0.0f32; padded_dim];

    for row in 0..num_rows {
        let x = &f32_slice[row * dimensions..][..dimensions];

        // Zero-pad to the next power of 2.
        padded[..dimensions].copy_from_slice(x);
        padded[dimensions..].fill(0.0);

        rotation.rotate(&padded, &mut rotated);

        for j in 0..padded_dim {
            all_indices.push(find_nearest_centroid(rotated[j], &boundaries));
        }
    }

    all_indices.freeze()
}
