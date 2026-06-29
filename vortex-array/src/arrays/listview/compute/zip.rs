// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem::MaybeUninit;
use std::ops::BitAnd;
use std::ops::BitOr;
use std::ops::Not;

use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Chunked;
use crate::arrays::ChunkedArray;
use crate::arrays::ListView;
use crate::arrays::ListViewArray;
use crate::arrays::chunked::ChunkedArrayExt;
use crate::arrays::listview::ListViewArrayExt;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::scalar_fn::fns::zip::ZipKernel;
use crate::validity::Validity;

/// Zip two [`ListViewArray`]s by selecting whole list views per row.
///
/// A [`ListViewArray`] addresses each list by an `(offset, size)` pair into a shared `elements`
/// array, and unlike [`ListArray`](crate::arrays::ListArray) it does not require lists to be stored
/// contiguously or in order. Zipping two list views is therefore a metadata-only operation over the
/// `offsets`, `sizes` and `validity` child arrays: we concatenate the two `elements` arrays
/// (without rewriting them) and, for each row, select the `(offset, size)` pair from `if_true` or
/// `if_false` per the mask. `if_false` views are shifted past the end of `if_true`'s elements so
/// they continue to address the correct half of the concatenated elements array.
impl ZipKernel for ListView {
    fn zip(
        if_true: ArrayView<'_, ListView>,
        if_false: &ArrayRef,
        mask: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(if_false) = if_false.as_opt::<ListView>() else {
            return Ok(None);
        };

        // Null mask entries select `if_false`, matching `Zip`'s SQL ELSE semantics.
        let mask = mask.try_to_mask_fill_null_false(ctx)?;
        match &mask {
            // Defer the trivial masks to the generic zip, which just casts one side.
            Mask::AllTrue(_) | Mask::AllFalse(_) => return Ok(None),
            Mask::Values(_) => {}
        }

        let len = if_true.len();

        let result_elements_dtype = if_true
            .elements()
            .dtype()
            .union_nullability(if_false.elements().dtype().nullability());

        // `if_false`'s elements share the element dtype up to nullability; normalize so both chunks
        // of the concatenated elements array have an identical dtype.
        let true_elements = if_true.elements().cast(result_elements_dtype.clone())?;
        let false_elements = if_false.elements().cast(result_elements_dtype.clone())?;

        // `if_false` views index into the second half of the concatenated elements.
        let false_shift = true_elements.len() as u64;

        // Concatenate the two `elements` arrays without copying. If either side is already a
        // `ChunkedArray` (e.g. the result of a previous list-view zip), splice its chunks in
        // directly rather than nesting chunked arrays.
        let mut chunks = Vec::with_capacity(2);
        push_element_chunks(true_elements, &mut chunks);
        push_element_chunks(false_elements, &mut chunks);
        let elements = ChunkedArray::try_new(chunks, result_elements_dtype)?.into_array();

        let true_offsets = to_u64(if_true.offsets(), ctx)?;
        let true_sizes = to_u64(if_true.sizes(), ctx)?;
        let false_offsets = to_u64(if_false.offsets(), ctx)?;
        let false_sizes = to_u64(if_false.sizes(), ctx)?;

        let mut offsets = BufferMut::<u64>::with_capacity(len);
        let mut sizes = BufferMut::<u64>::with_capacity(len);
        {
            let true_offsets = true_offsets.as_slice();
            let true_sizes = true_sizes.as_slice();
            let false_offsets = false_offsets.as_slice();
            let false_sizes = false_sizes.as_slice();

            let offsets_out = offsets.spare_capacity_mut();
            let sizes_out = sizes.spare_capacity_mut();

            // We matched `Mask::Values` above, so the bit buffer is materialized. `unaligned_chunks`
            // iterates faster than `chunks`: it exposes the byte-aligned body as a plain `&[u64]`
            // with no per-word reshifting, isolating any bit misalignment into a leading `prefix`
            // and trailing `suffix` word. We blend both sides branchlessly per row so the compiler
            // vectorizes the inner select instead of mispredicting a data-dependent branch.
            let mask_bits = mask
                .values()
                .vortex_expect("mask is Mask::Values")
                .bit_buffer();
            let unaligned = mask_bits.unaligned_chunks();
            // The prefix word's low `lead` bits are padding; shifting them out aligns row 0 to bit 0,
            // after which every chunk and the suffix start cleanly on a row boundary.
            let lead = unaligned.lead_padding();

            let mut select_block = |word: u64, base: usize, n: usize| {
                let end = base + n;
                // `if_false` views address the second half of the concatenated elements, so shift
                // their offsets by `false_shift`; sizes are taken verbatim from the chosen side.
                select_column(
                    word,
                    &true_offsets[base..end],
                    &false_offsets[base..end],
                    false_shift,
                    &mut offsets_out[base..end],
                );
                select_column(
                    word,
                    &true_sizes[base..end],
                    &false_sizes[base..end],
                    0,
                    &mut sizes_out[base..end],
                );
            };

            let mut base = 0;
            if let Some(prefix) = unaligned.prefix() {
                let n = (64 - lead).min(len);
                select_block(prefix >> lead, base, n);
                base += n;
            }
            for &word in unaligned.chunks() {
                select_block(word, base, 64);
                base += 64;
            }
            if let Some(suffix) = unaligned.suffix() {
                select_block(suffix, base, len - base);
            }
        }

        // SAFETY: `select_column` initialized exactly `len` slots in both buffers.
        unsafe {
            offsets.set_len(len);
            sizes.set_len(len);
        }

        let validity = zip_validity(if_true.validity()?, if_false.validity()?, &mask, ctx)?;

        Ok(Some(
            ListViewArray::try_new(
                elements,
                offsets.freeze().into_array(),
                sizes.freeze().into_array(),
                validity,
            )?
            .into_array(),
        ))
    }
}

/// Branchlessly select one `u64` column per row from `if_true` or `if_false`.
///
/// `word` holds the mask bits for this block, bit `j` (LSB-first) selecting row `j`: a set bit keeps
/// `true_vals[j]`, an unset bit keeps `false_vals[j] + false_add`. The bit is expanded to a
/// full-width lane mask and blended, so the inner loop is branch-free and auto-vectorizable. Inputs
/// are sliced to the output length up front so the compiler can elide bounds checks across the block.
#[inline]
fn select_column(
    word: u64,
    true_vals: &[u64],
    false_vals: &[u64],
    false_add: u64,
    out: &mut [MaybeUninit<u64>],
) {
    let n = out.len();
    let true_vals = &true_vals[..n];
    let false_vals = &false_vals[..n];
    for j in 0..n {
        // 0 for an unset bit, `u64::MAX` for a set bit.
        let lane = 0u64.wrapping_sub((word >> j) & 1);
        out[j].write((true_vals[j] & lane) | ((false_vals[j] + false_add) & !lane));
    }
}

/// Appends `array`'s element chunks to `chunks`, flattening a top-level [`ChunkedArray`] so the
/// concatenated elements never nest chunked arrays.
fn push_element_chunks(array: ArrayRef, chunks: &mut Vec<ArrayRef>) {
    match array.as_opt::<Chunked>() {
        Some(chunked) => chunks.extend(chunked.iter_chunks().cloned()),
        None => chunks.push(array),
    }
}

/// Read a non-nullable integer array into a `u64` buffer.
fn to_u64(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Buffer<u64>> {
    array
        .clone()
        .cast(DType::Primitive(PType::U64, Nullability::NonNullable))?
        .execute::<Buffer<u64>>(ctx)
}

/// Combine the two list-level validities, taking `if_true`'s validity where `mask` is set and
/// `if_false`'s where it is not.
fn zip_validity(
    if_true: Validity,
    if_false: Validity,
    mask: &Mask,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Validity> {
    Ok(match (&if_true, &if_false) {
        (Validity::NonNullable, Validity::NonNullable) => Validity::NonNullable,
        (Validity::AllValid, Validity::AllValid) => Validity::AllValid,
        (Validity::AllInvalid, Validity::AllInvalid) => Validity::AllInvalid,
        _ => {
            let true_mask = if_true.execute_mask(mask.len(), ctx)?;
            let false_mask = if_false.execute_mask(mask.len(), ctx)?;
            let combined = true_mask
                .bitand(mask)
                .bitor(&false_mask.bitand(&mask.not()));
            Validity::from_mask(combined, if_true.nullability() | if_false.nullability())
        }
    })
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::cast_possible_truncation,
        reason = "test fixtures use small indices that fit the target widths"
    )]

    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_mask::Mask;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::arrays::BoolArray;
    use crate::arrays::Chunked;
    use crate::arrays::ChunkedArray;
    use crate::arrays::ListView;
    use crate::arrays::ListViewArray;
    use crate::arrays::chunked::ChunkedArrayExt;
    use crate::arrays::listview::ListViewArrayExt;
    use crate::assert_arrays_eq;
    use crate::builtins::ArrayBuiltins;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::validity::Validity;

    fn list_view(
        elements: ArrayRef,
        offsets: ArrayRef,
        sizes: ArrayRef,
        validity: Validity,
    ) -> ArrayRef {
        ListViewArray::try_new(elements, offsets, sizes, validity)
            .unwrap()
            .into_array()
    }

    /// `zip` of two list views selects whole lists per the mask and keeps the list encoding.
    #[test]
    fn zip_selects_lists() -> VortexResult<()> {
        // [[1, 2], [3], [4, 5, 6]]
        let if_true = list_view(
            buffer![1i32, 2, 3, 4, 5, 6].into_array(),
            buffer![0u32, 2, 3].into_array(),
            buffer![2u32, 1, 3].into_array(),
            Validity::NonNullable,
        );
        // [[10], [20, 21], [30]]
        let if_false = list_view(
            buffer![10i32, 20, 21, 30].into_array(),
            buffer![0u32, 1, 3].into_array(),
            buffer![1u32, 2, 1].into_array(),
            Validity::NonNullable,
        );
        let mask = Mask::from_iter([true, false, true]);

        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let result = mask
            .into_array()
            .zip(if_true, if_false)?
            .execute::<ArrayRef>(&mut ctx)?;

        // The kernel should keep the list-view encoding rather than canonicalizing.
        assert!(result.is::<ListView>());

        // Expected: [[1, 2], [20, 21], [4, 5, 6]]
        let expected = list_view(
            buffer![1i32, 2, 20, 21, 4, 5, 6].into_array(),
            buffer![0u32, 2, 4].into_array(),
            buffer![2u32, 2, 3].into_array(),
            Validity::NonNullable,
        );
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    /// `zip` selects list-level validity from the chosen side and widens nullability.
    #[test]
    fn zip_selects_validity() -> VortexResult<()> {
        // [[1], null, [2]] (list-level nulls)
        let if_true = list_view(
            buffer![1i32, 2].into_array(),
            buffer![0u32, 1, 1].into_array(),
            buffer![1u32, 0, 1].into_array(),
            Validity::Array(BoolArray::from_iter([true, false, true]).into_array()),
        );
        // [[10], [20], null]
        let if_false = list_view(
            buffer![10i32, 20].into_array(),
            buffer![0u32, 1, 2].into_array(),
            buffer![1u32, 1, 0].into_array(),
            Validity::Array(BoolArray::from_iter([true, true, false]).into_array()),
        );
        // true -> if_true, false -> if_false
        let mask = Mask::from_iter([false, true, true]);

        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let result = mask
            .into_array()
            .zip(if_true, if_false)?
            .execute::<ArrayRef>(&mut ctx)?;

        // Row 0 -> if_false[0] = [10]; row 1 -> if_true[1] = null; row 2 -> if_true[2] = [2]
        let expected = list_view(
            buffer![10i32, 2].into_array(),
            buffer![0u32, 1, 1].into_array(),
            buffer![1u32, 0, 1].into_array(),
            Validity::Array(BoolArray::from_iter([true, false, true]).into_array()),
        );
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    /// `zip` handles out-of-order/non-contiguous offsets and widens nullability when only one side
    /// is nullable.
    #[test]
    fn zip_out_of_order_offsets_and_widening() -> VortexResult<()> {
        // [[5, 6], [7], [8, 9]] expressed with out-of-order offsets.
        let if_true = list_view(
            buffer![7i32, 8, 9, 5, 6].into_array(),
            buffer![3u32, 0, 1].into_array(),
            buffer![2u32, 1, 2].into_array(),
            Validity::NonNullable,
        );
        // [[100], null, [200, 201]]
        let if_false = list_view(
            buffer![100i32, 200, 201].into_array(),
            buffer![0u32, 1, 1].into_array(),
            buffer![1u32, 0, 2].into_array(),
            Validity::Array(BoolArray::from_iter([true, false, true]).into_array()),
        );
        let mask = Mask::from_iter([true, true, false]);

        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let result = mask
            .into_array()
            .zip(if_true, if_false)?
            .execute::<ArrayRef>(&mut ctx)?;
        assert!(result.is::<ListView>());

        // [[5, 6], [7], [200, 201]], all valid but nullable (widened by if_false).
        let expected = list_view(
            buffer![5i32, 6, 7, 200, 201].into_array(),
            buffer![0u32, 2, 3].into_array(),
            buffer![2u32, 1, 2].into_array(),
            Validity::AllValid,
        );
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    /// Zipping more rows than fit in a single 64-bit mask chunk exercises both the chunked select
    /// loop and the trailing remainder, including the `false_shift` applied to `if_false` views.
    #[test]
    fn zip_spans_multiple_mask_chunks() -> VortexResult<()> {
        // 130 single-element lists per side: `if_true[i] = [i]`, `if_false[i] = [1000 + i]`.
        let len = 130usize;
        let true_elements: Vec<i32> = (0..len as i32).collect();
        let false_elements: Vec<i32> = (0..len as i32).map(|i| 1000 + i).collect();
        let offsets: Vec<u64> = (0..len as u64).collect();
        let sizes: Vec<u64> = vec![1; len];

        let if_true = list_view(
            true_elements
                .iter()
                .copied()
                .collect::<Buffer<i32>>()
                .into_array(),
            offsets
                .iter()
                .copied()
                .collect::<Buffer<u64>>()
                .into_array(),
            sizes.iter().copied().collect::<Buffer<u64>>().into_array(),
            Validity::NonNullable,
        );
        let if_false = list_view(
            false_elements
                .iter()
                .copied()
                .collect::<Buffer<i32>>()
                .into_array(),
            offsets
                .iter()
                .copied()
                .collect::<Buffer<u64>>()
                .into_array(),
            sizes.iter().copied().collect::<Buffer<u64>>().into_array(),
            Validity::NonNullable,
        );

        // A non-trivial pattern that straddles the chunk boundary (index 63/64) and the remainder.
        let mask_bits: Vec<bool> = (0..len).map(|i| i.is_multiple_of(3) || i == 64).collect();
        let mask = Mask::from_iter(mask_bits.iter().copied());

        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let result = mask
            .into_array()
            .zip(if_true, if_false)?
            .execute::<ArrayRef>(&mut ctx)?;
        assert!(result.is::<ListView>());

        // Each row collapses to a single element: `i` when the mask is set, else `1000 + i`.
        let expected_elements: Vec<i32> = (0..len)
            .map(|i| {
                if mask_bits[i] {
                    i as i32
                } else {
                    1000 + i as i32
                }
            })
            .collect();
        let expected = list_view(
            expected_elements
                .iter()
                .copied()
                .collect::<Buffer<i32>>()
                .into_array(),
            offsets
                .iter()
                .copied()
                .collect::<Buffer<u64>>()
                .into_array(),
            sizes.iter().copied().collect::<Buffer<u64>>().into_array(),
            Validity::NonNullable,
        );
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    /// A mask whose bit buffer starts at a non-byte-aligned offset (here from slicing a bool array)
    /// has non-zero `unaligned_chunks` lead padding, exercising the prefix word alongside the
    /// aligned chunk body and the suffix.
    #[test]
    fn zip_handles_offset_mask() -> VortexResult<()> {
        // 200 single-element lists per side: `if_true[i] = [i]`, `if_false[i] = [1000 + i]`. With a
        // 3-bit lead offset the mask spans more than 16 bytes, so `unaligned_chunks` exposes a
        // non-empty aligned `chunks` body between the prefix and suffix words.
        let len = 200usize;
        let true_elements: Vec<i32> = (0..len as i32).collect();
        let false_elements: Vec<i32> = (0..len as i32).map(|i| 1000 + i).collect();
        let offsets: Vec<u64> = (0..len as u64).collect();
        let sizes: Vec<u64> = vec![1; len];

        let single_element_view = |elements: &[i32]| {
            list_view(
                elements
                    .iter()
                    .copied()
                    .collect::<Buffer<i32>>()
                    .into_array(),
                offsets
                    .iter()
                    .copied()
                    .collect::<Buffer<u64>>()
                    .into_array(),
                sizes.iter().copied().collect::<Buffer<u64>>().into_array(),
                Validity::NonNullable,
            )
        };
        let if_true = single_element_view(&true_elements);
        let if_false = single_element_view(&false_elements);

        // Slice off the first `offset` bits so the mask's bit buffer keeps a sub-byte offset while
        // remaining `len` rows long. A non-trivial pattern straddles the prefix/body and chunk
        // boundaries within the sliced window.
        let offset = 3usize;
        let mask_bits: Vec<bool> = (0..offset + len)
            .map(|i| i.is_multiple_of(3) || i == offset + 64)
            .collect();
        let mask = BoolArray::from_iter(mask_bits.iter().copied())
            .into_array()
            .slice(offset..offset + len)?;

        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let result = mask.zip(if_true, if_false)?.execute::<ArrayRef>(&mut ctx)?;
        assert!(result.is::<ListView>());

        // Each row collapses to a single element: `i` when the sliced mask is set, else `1000 + i`.
        let expected_elements: Vec<i32> = (0..len)
            .map(|i| {
                if mask_bits[offset + i] {
                    i as i32
                } else {
                    1000 + i as i32
                }
            })
            .collect();
        let expected = single_element_view(&expected_elements);
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    /// When an input's `elements` is already a [`ChunkedArray`], its chunks are spliced in rather
    /// than nesting a chunked array inside the concatenated elements.
    #[test]
    fn zip_flattens_chunked_elements() -> VortexResult<()> {
        // elements [1, 2, 3] stored as two chunks; lists [[1, 2], [3]].
        let chunked_elements = ChunkedArray::try_new(
            vec![buffer![1i32, 2].into_array(), buffer![3i32].into_array()],
            DType::Primitive(PType::I32, Nullability::NonNullable),
        )?
        .into_array();
        let if_true = list_view(
            chunked_elements,
            buffer![0u32, 2].into_array(),
            buffer![2u32, 1].into_array(),
            Validity::NonNullable,
        );
        // [[10], [20]]
        let if_false = list_view(
            buffer![10i32, 20].into_array(),
            buffer![0u32, 1].into_array(),
            buffer![1u32, 1].into_array(),
            Validity::NonNullable,
        );
        let mask = Mask::from_iter([true, false]);

        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let result = mask
            .into_array()
            .zip(if_true, if_false)?
            .execute::<ArrayRef>(&mut ctx)?;

        // The concatenated elements are chunked, but no chunk is itself a `ChunkedArray`.
        let result_lv = result
            .as_opt::<ListView>()
            .expect("zip keeps the list-view encoding");
        let chunked = result_lv
            .elements()
            .as_opt::<Chunked>()
            .expect("zip concatenates elements into a chunked array");
        assert!(
            chunked.iter_chunks().all(|chunk| !chunk.is::<Chunked>()),
            "chunked elements must be flattened, not nested",
        );

        // [[1, 2], [20]]
        let expected = list_view(
            buffer![1i32, 2, 20].into_array(),
            buffer![0u32, 2].into_array(),
            buffer![2u32, 1].into_array(),
            Validity::NonNullable,
        );
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }
}
