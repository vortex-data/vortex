// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Block-aware TurboQuant encode pipeline.
//!
//! Each block of an input vector array is encoded independently: per-row L2 norm, per-row SORF
//! transform sized to the block, and per-row scalar quantization against the block's centroid
//! table. The output is one [`Block`] per block in `block_sizes`, each row-aligned to the
//! input row count and carrying the input's row validity.
//!
//! # Block slicing
//!
//! Block `i` covers input coordinates `[offset_i .. offset_i + block_sizes[i])`, where
//! `offset_i = sum(block_sizes[..i])`. When a block extends past `dimensions` its tail is
//! zero-padded; a block whose `offset_i >= dimensions` is entirely padding. Such overspilling
//! block lists are valid, not rejected; `resolve_block_sizes` emits a `tracing::warn!` only for a
//! block lying entirely past `dimensions` or a sum exceeding `2 * dimensions`.
//!
//! # Per-block algorithm
//!
//! For each block `i` of each valid input row, the encoder:
//!
//! 1. Slices the block out of the input, zero-padding any range that extends past `dimensions`.
//! 2. Computes the block's L2 norm and writes it into that block's `norms` column.
//! 3. Divides the slice by that norm to produce a unit-norm block.
//! 4. Applies a SORF transform of width `block_sizes[i]` seeded with
//!    `derive_block_seed(config.seed(), i)`, so every block has its own distinct rotation even
//!    when two blocks share the same width.
//! 5. Scalar-quantizes the rotated coordinates against a `2^bit_width`-entry centroid table built
//!    for width `block_sizes[i]` and writes the codes into that block's `codes` column.
//!
//! # Null and zero-norm rows
//!
//! Per-row null and zero-norm handling mirrors the previous single-block pipeline: a null row
//! writes zero placeholders into every block's `norms` and `codes`, and a valid row whose block
//! slice has zero norm writes zeros into that block's children only.

use half::f16;
use num_traits::Float;
use num_traits::FromPrimitive;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrays::fixed_size_list::FixedSizeListArrayExt;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PType;
use vortex_array::match_each_float_ptype;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_tensor::vector::AnyVector;

use crate::centroids::compute_or_get_codebook;
use crate::centroids::find_nearest_centroid;
use crate::sorf::splitmix64::derive_block_seed;
use crate::sorf::transform::SorfMatrix;
use crate::vector::storage::Block;

/// Per-block precomputed runtime state shared across rows.
///
/// Built once per encode call and reused for every row of the input array.
pub(crate) struct BlockRuntimeState {
    /// One [`SorfMatrix`] per block, sized to its block width and seeded from [`derive_block_seed`]
    /// `(global_seed, block_index)`.
    matrices: Vec<SorfMatrix>,
    /// Precomputed centroid boundaries used by [`find_nearest_centroid`], one cheap-to-clone
    /// reference-counted [`Buffer`] per block.
    boundaries: Vec<Buffer<f32>>,
}

/// Build the per-block SORF transforms and centroid tables for a given config and resolved block
/// list. Inexpensive when the centroid cache is warm.
pub(crate) fn prepare_block_state(
    seed: u64,
    num_rounds: u8,
    bit_width: u8,
    block_sizes: &[u32],
) -> VortexResult<BlockRuntimeState> {
    let mut matrices = Vec::with_capacity(block_sizes.len());
    let mut boundaries = Vec::with_capacity(block_sizes.len());

    for (index, &block) in block_sizes.iter().enumerate() {
        let block_usize = usize::try_from(block)
            .map_err(|_| vortex_err!("TurboQuant block {block} does not fit usize"))?;

        // Each block gets a distinct SORF rotation derived from the global seed and its index.
        let seed_i = derive_block_seed(seed, index);

        matrices.push(SorfMatrix::try_new(
            block_usize,
            num_rounds as usize,
            seed_i,
        )?);

        boundaries.push(
            compute_or_get_codebook(block, bit_width)?
                .boundaries
                .clone(),
        );
    }

    Ok(BlockRuntimeState {
        matrices,
        boundaries,
    })
}

/// Encode every block of `input` (the original `Vector` extension array that is not pre-normalized)
/// into its own `(norms, codes)` row-aligned pair.
///
/// Returns one [`Block`] per block in `block_sizes`, each carrying `num_vectors` rows.
pub(crate) fn turboquant_encode_blocks(
    input: ArrayRef,
    block_sizes: &[u32],
    state: &BlockRuntimeState,
    vector_validity: Validity,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Vec<Block>> {
    let num_vectors = input.len();
    let vector_metadata = input
        .dtype()
        .as_extension_opt()
        .and_then(|ext_dtype| ext_dtype.metadata_opt::<AnyVector>())
        .ok_or_else(|| vortex_err!("TurboQuant encode expects a Vector extension array"))?;

    let dimensions = usize::try_from(vector_metadata.dimensions())
        .map_err(|_| vortex_err!("TurboQuant dimensions does not fit usize"))?;
    let element_ptype = vector_metadata.element_ptype();

    let extension: ExtensionArray = input.execute(ctx)?;
    let storage: FixedSizeListArray = extension.storage_array().clone().execute(ctx)?;
    let elements: PrimitiveArray = storage.elements().clone().execute(ctx)?;
    let mask = vector_validity.execute_mask(num_vectors, ctx)?;

    // TODO(connor): It would be more "correct" to compute norms **before** casting to f32.
    let f32_input = cast_to_f32(elements)?;
    let f32_slice = f32_input.as_slice();

    // `encode_blocks_typed` is monomorphized per float ptype although its hot loop runs in f32, and
    // only the output norm column depends on `T`.
    let block_arrays = match_each_float_ptype!(element_ptype, |T| {
        encode_blocks_typed::<T>(
            f32_slice,
            dimensions,
            num_vectors,
            &mask,
            block_sizes,
            state,
            vector_validity.clone(),
        )?
    });

    Ok(block_arrays)
}

// TODO(connor): Clean up this function!
fn encode_blocks_typed<T>(
    input: &[f32],
    dimensions: usize,
    num_vectors: usize,
    mask: &Mask,
    block_sizes: &[u32],
    state: &BlockRuntimeState,
    vector_validity: Validity,
) -> VortexResult<Vec<Block>>
where
    T: NativePType + Float + FromPrimitive,
{
    let block_widths: Vec<usize> = block_sizes
        .iter()
        .map(|&b| {
            usize::try_from(b).map_err(|_| vortex_err!("TurboQuant block {b} does not fit usize"))
        })
        .collect::<VortexResult<Vec<_>>>()?;

    // `total_block_width` sizes the per-block scratch and the offset-invariant assert below; the
    // `sum >= dimensions` rule itself is enforced upstream by `validate_block_sum` (via
    // `resolve_block_sizes`), so it is not re-checked here.
    let total_block_width: usize = block_widths.iter().sum();

    // Per-block output buffers. `norms_out[b]` collects `num_vectors` block-norm values;
    // `codes_out[b]` collects `num_vectors * block_sizes[b]` u8 codes.
    let mut norms_out: Vec<BufferMut<T>> = block_sizes
        .iter()
        .map(|_| BufferMut::<T>::with_capacity(num_vectors))
        .collect();
    let mut codes_out: Vec<BufferMut<u8>> = block_widths
        .iter()
        .map(|&b| {
            let len = num_vectors
                .checked_mul(b)
                .ok_or_else(|| vortex_err!("TurboQuant codes length overflow"))?;
            Ok::<_, vortex_error::VortexError>(BufferMut::<u8>::with_capacity(len))
        })
        .collect::<VortexResult<_>>()?;

    // Per-block scratch buffers reused across rows.
    let mut padded_scratch: Vec<Vec<f32>> = block_widths.iter().map(|&b| vec![0.0f32; b]).collect();
    let mut transformed_scratch: Vec<Vec<f32>> =
        block_widths.iter().map(|&b| vec![0.0f32; b]).collect();

    for row in 0..num_vectors {
        let is_valid = mask.value(row);
        let row_input = &input[row * dimensions..][..dimensions];
        let mut offset = 0usize;
        for (block_index, &block) in block_widths.iter().enumerate() {
            if !is_valid {
                // SAFETY: norms_out[block_index] reserved `num_vectors` capacity at start.
                unsafe { norms_out[block_index].push_unchecked(T::zero()) };
                // SAFETY: codes_out[block_index] reserved `num_vectors * block` capacity.
                unsafe { codes_out[block_index].push_n_unchecked(0u8, block) };
                offset += block;
                continue;
            }
            // Copy the row's block slice into the scratch buffer, zero-padding the final block
            // when `offset + block > dimensions`.
            let take = block.min(dimensions.saturating_sub(offset));
            if take > 0 {
                padded_scratch[block_index][..take]
                    .copy_from_slice(&row_input[offset..offset + take]);
            }
            if take < block {
                padded_scratch[block_index][take..].fill(0.0);
            }
            // Computed in f32 to match the SORF transform precision. For f64 inputs this is an
            // intentional precision downgrade relative to the legacy per-input-ptype `L2Norm`,
            // accepted as part of the block-decomposition wire-format break.
            let norm_sq: f32 = padded_scratch[block_index]
                .iter()
                .map(|&v| v * v)
                .sum::<f32>();
            let norm_f32 = norm_sq.sqrt();
            let norm_value = T::from_f32(norm_f32)
                .vortex_expect("from_f32 is infallible for supported float types");
            // Reject a non-finite stored norm (an input magnitude out of the element type's range)
            // rather than emit an array the decoder cannot reconstruct.
            if !norm_value.is_finite() {
                vortex_bail!(
                    "TurboQuant block norm is not finite; an input magnitude is out of range"
                );
            }
            // SAFETY: capacity reserved above.
            unsafe { norms_out[block_index].push_unchecked(norm_value) };

            if norm_f32 == 0.0 {
                // SAFETY: capacity reserved above.
                unsafe { codes_out[block_index].push_n_unchecked(0u8, block) };
                offset += block;
                continue;
            }

            // Normalize in place by the block norm.
            for value in padded_scratch[block_index].iter_mut() {
                *value /= norm_f32;
            }
            state.matrices[block_index].transform(
                &padded_scratch[block_index],
                &mut transformed_scratch[block_index],
            );

            let boundaries = &state.boundaries[block_index];
            for &value in &transformed_scratch[block_index] {
                let code = find_nearest_centroid(value, boundaries);
                // SAFETY: capacity reserved above.
                unsafe { codes_out[block_index].push_unchecked(code) };
            }
            offset += block;
        }
        debug_assert_eq!(offset, total_block_width);
    }

    let mut result = Vec::with_capacity(block_sizes.len());
    for block_index in 0..block_sizes.len() {
        let norms_buf = std::mem::take(&mut norms_out[block_index]).freeze();
        let codes_buf = std::mem::take(&mut codes_out[block_index]).freeze();
        let norms = PrimitiveArray::new::<T>(norms_buf, vector_validity.clone());
        let codes = PrimitiveArray::new::<u8>(codes_buf, Validity::NonNullable);
        result.push(Block { norms, codes });
    }
    Ok(result)
}

/// Cast a float [`PrimitiveArray`] to a `Buffer<f32>`.
///
/// All in-loop arithmetic happens in f32 for SORF compatibility; the input element ptype is
/// lossily widened or narrowed once at the start.
fn cast_to_f32(prim: PrimitiveArray) -> VortexResult<Buffer<f32>> {
    match prim.ptype() {
        PType::F16 => Ok(prim
            .as_slice::<f16>()
            .iter()
            .map(|&v| f32::from(v))
            .collect()),
        PType::F32 => Ok(prim.into_buffer()),
        PType::F64 => Ok(prim
            .as_slice::<f64>()
            .iter()
            .map(|&v| {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "f64 values outside f32 range become infinity, matching tensor TQ"
                )]
                let v = v as f32;
                v
            })
            .collect()),
        other => vortex_bail!("expected float elements, got {other:?}"),
    }
}
