// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Block-aware TurboQuant encode pipeline.

// TODO(connor): More docs!

use num_traits::Float;
use num_traits::FromPrimitive;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::NativePType;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_tensor::vector::Vector;

use crate::TurboQuantMetadata;
use crate::sorf::transform::SorfMatrix;
use crate::vector::storage::Block;

/// Borrowed bundle of per-array decode inputs passed to the typed inner loop.
///
/// Packaged as a struct rather than positional arguments because [`decode_typed`] runs through
/// [`match_each_float_ptype!`] which expands once per supported element ptype. Each expansion
/// takes the same set of inputs, and the struct keeps the call site short.
pub(crate) struct DecodeInputs<'a> {
    /// TurboQuant metadata recovered from the input extension dtype.
    pub(crate) metadata: &'a TurboQuantMetadata,

    /// Block widths in `usize`, parallel to `metadata.block_sizes`. Cached to avoid repeated
    /// `usize::try_from` in the row loop.
    pub(crate) block_sizes: &'a [usize],

    /// Sum of `block_sizes`. The decode loop's row-aligned scratch buffer is this wide.
    pub(crate) total_width: usize,

    /// One `SorfMatrix` per block, seeded via `derive_block_seed(metadata.seed, i)`.
    pub(crate) sorf_matrices: &'a [SorfMatrix],

    /// One centroid table per block, keyed on `(block_sizes[i], bit_width)`.
    pub(crate) centroid_tables: &'a [Buffer<f32>],

    /// Per-block executed `(norms, codes)` storage children, in block order.
    pub(crate) block_storages: &'a [Block],
}

// TODO(connor): Clean up this function!
pub(crate) fn decode_typed<T>(
    decode: DecodeInputs<'_>,
    vector_validity: Validity,
    num_vectors: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef>
where
    T: NativePType + Float + FromPrimitive,
{
    let metadata = decode.metadata;
    let dimensions = usize::try_from(metadata.dimensions)
        .vortex_expect("dimensions stays representable as usize");
    let mask = vector_validity.execute_mask(num_vectors, ctx)?;

    let output_len = num_vectors
        .checked_mul(dimensions)
        .ok_or_else(|| vortex_err!("TurboQuant decoded vector length overflow"))?;
    let mut output = BufferMut::<T>::with_capacity(output_len);

    let mut decoded_blocks: Vec<Vec<f32>> =
        decode.block_sizes.iter().map(|&b| vec![0.0; b]).collect();
    let mut inverse_blocks: Vec<Vec<f32>> =
        decode.block_sizes.iter().map(|&b| vec![0.0; b]).collect();
    // `total_width == sum(block_sizes) >= dimensions` (enforced by validate_block_sum at metadata
    // construction/deserialize), so the `row_scratch[..dimensions]` copy below is always in bounds.
    let mut row_scratch = vec![0.0f32; decode.total_width];

    let block_norms: Vec<&[T]> = decode
        .block_storages
        .iter()
        .map(|bs| bs.norms.as_slice::<T>())
        .collect();
    let block_codes: Vec<&[u8]> = decode
        .block_storages
        .iter()
        .map(|bs| bs.codes.as_slice::<u8>())
        .collect();

    // `decode_row` is fallible and validates each code against its block's centroid table at the
    // lookup site. Validation happens here rather than up front over every physical row so that
    // null / masked-out rows, whose placeholder codes are never decoded, cannot trip a bounds
    // error: the closure is only invoked for rows selected by `mask` below.
    let mut decode_row = |output: &mut BufferMut<T>, row: usize| -> VortexResult<()> {
        let mut offset = 0usize;
        for (block_index, &block) in decode.block_sizes.iter().enumerate() {
            let code_row = &block_codes[block_index][row * block..][..block];
            let centroids = decode.centroid_tables[block_index].as_slice();
            for (dst, &code) in decoded_blocks[block_index].iter_mut().zip(code_row.iter()) {
                *dst = *centroids.get(code as usize).ok_or_else(|| {
                    vortex_err!(
                        "TurboQuant code {code} exceeds centroid count {} for block {block_index}",
                        centroids.len()
                    )
                })?;
            }
            decode.sorf_matrices[block_index].inverse_transform(
                &decoded_blocks[block_index],
                &mut inverse_blocks[block_index],
            );

            let norm = block_norms[block_index][row];
            let norm_f32 = norm
                .to_f32()
                .vortex_expect("to_f32 is infallible for supported float types");
            // A stored norm must be a finite, non-negative magnitude; reject malformed storage that
            // would otherwise scale the reconstruction by garbage (or sign-flip it).
            vortex_ensure!(
                norm_f32.is_finite() && norm_f32 >= 0.0,
                "TurboQuant stored block norm is not a valid magnitude, got {norm_f32}"
            );
            for (dst, &value) in row_scratch[offset..offset + block]
                .iter_mut()
                .zip(inverse_blocks[block_index].iter())
            {
                *dst = value * norm_f32;
            }
            offset += block;
        }
        debug_assert_eq!(offset, decode.total_width);

        for &value in &row_scratch[..dimensions] {
            let value = T::from_f32(value)
                .vortex_expect("from_f32 is infallible for supported float types");
            // SAFETY: total pushes equal `output_len` across all match arms below.
            unsafe { output.push_unchecked(value) };
        }
        Ok(())
    };

    match &mask {
        Mask::AllFalse(_) => {
            // SAFETY: `output` has capacity `output_len` and this writes exactly `output_len`
            // zero placeholders, so the push stays within the reserved capacity.
            unsafe { output.push_n_unchecked(T::zero(), output_len) };
        }
        Mask::AllTrue(_) => {
            for row in 0..num_vectors {
                decode_row(&mut output, row)?;
            }
        }
        Mask::Values(values_mask) => {
            let mut cursor = 0;
            for &(start, end) in values_mask.slices() {
                if start > cursor {
                    // SAFETY: total pushes across all arms equal `output_len`.
                    unsafe { output.push_n_unchecked(T::zero(), (start - cursor) * dimensions) };
                }
                for row in start..end {
                    decode_row(&mut output, row)?;
                }
                cursor = end;
            }
            if cursor < num_vectors {
                // SAFETY: total pushes across all arms equal `output_len`.
                unsafe { output.push_n_unchecked(T::zero(), (num_vectors - cursor) * dimensions) };
            }
        }
    }

    let elements = PrimitiveArray::new::<T>(output.freeze(), Validity::NonNullable);
    let fsl = FixedSizeListArray::try_new(
        elements.into_array(),
        metadata.dimensions,
        vector_validity,
        num_vectors,
    )?;

    Vector::try_new_vector_array(fsl.into_array())
}
