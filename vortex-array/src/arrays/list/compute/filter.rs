// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use num_traits::Zero;
use vortex_buffer::BitBufferMut;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_mask::MaskIter;
use vortex_mask::MaskValues;

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::ConstantArray;
use crate::arrays::List;
use crate::arrays::ListArray;
use crate::arrays::ListViewArray;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::filter::FilterKernel;
use crate::arrays::filter::FilterReduce;
use crate::arrays::list::ListArrayExt;
use crate::arrays::primitive::PrimitiveArrayExt;
use crate::dtype::IntegerPType;
use crate::match_each_integer_ptype;
use crate::validity::Validity;

/// Density threshold for choosing between indices and slices representation when expanding masks.
///
/// When the mask density is below this threshold, we use indices. Otherwise, we use slices.
///
/// Note that this is somewhat arbitrarily chosen...
const MASK_EXPANSION_DENSITY_THRESHOLD: f64 = 0.05;

/// Construct an element mask from contiguous list offsets and a selection mask.
pub fn element_mask_from_offsets<O: IntegerPType>(
    offsets: &[O],
    selection: &Arc<MaskValues>,
) -> Mask {
    let first_offset = offsets.first().map_or(0, |first_offset| first_offset.as_());
    let last_offset = offsets.last().map_or(0, |last_offset| last_offset.as_());
    let len = last_offset - first_offset;

    let mut mask_builder = BitBufferMut::with_capacity(len);

    match selection.threshold_iter(MASK_EXPANSION_DENSITY_THRESHOLD) {
        MaskIter::Slices(slices) => {
            // Dense iteration: process ranges of consecutive selected lists.
            for &(start, end) in slices {
                // Optimization: for dense ranges, we can process the elements mask more efficiently.
                let elems_start = offsets[start].as_() - first_offset;
                let elems_end = offsets[end].as_() - first_offset;

                // Process the entire range of elements at once.
                process_element_range(elems_start, elems_end, &mut mask_builder);
            }
        }
        MaskIter::Indices(indices) => {
            // Sparse iteration: process individual selected lists.
            for &idx in indices {
                let list_start = offsets[idx].as_() - first_offset;
                let list_end = offsets[idx + 1].as_() - first_offset;

                // Process the elements for this list.
                process_element_range(list_start, list_end, &mut mask_builder);
            }
        }
    }

    // Pad to full length if necessary.
    mask_builder.append_n(false, len - mask_builder.len());

    Mask::from_buffer(mask_builder.freeze())
}

/// Process a range of elements for filtering.
fn process_element_range(
    elems_start: usize,
    elems_end: usize,
    new_mask_builder: &mut BitBufferMut,
) {
    let elems_len = elems_end - elems_start;

    // Only process if there are elements to mark.
    if elems_len > 0 {
        // Fill any gaps before this range.
        if elems_start > new_mask_builder.len() {
            new_mask_builder.append_n(false, elems_start - new_mask_builder.len());
        }
        // Keep all elements in this range.
        new_mask_builder.append_n(true, elems_len);
    }
}

impl FilterKernel for List {
    fn filter(
        array: ArrayView<'_, List>,
        mask: &Mask,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let selection = match mask {
            Mask::AllTrue(_) | Mask::AllFalse(_) => return Ok(None),
            Mask::Values(v) => v,
        };

        let new_validity = match array.validity()? {
            Validity::NonNullable => Validity::NonNullable,
            Validity::AllValid => Validity::AllValid,
            Validity::AllInvalid => {
                let elements = Canonical::empty(array.element_dtype()).into_array();
                let offsets = ConstantArray::new(0u64, selection.true_count() + 1).into_array();
                return Ok(Some(unsafe {
                    ListArray::new_unchecked(elements, offsets, Validity::AllInvalid).into_array()
                }));
            }
            Validity::Array(a) => Validity::Array(a.filter(mask.clone())?),
        };

        // TODO(ngates): for ultra-sparse masks, we don't need to optimize the entire offsets.
        let offsets = array.offsets().clone();

        let (new_offsets, element_mask) =
            match_each_integer_ptype!(offsets.dtype().as_ptype(), |O| {
                let offsets_buffer = offsets.execute::<Buffer<O>>(ctx)?;
                let offsets = offsets_buffer.as_slice();
                let mut new_offsets = BufferMut::<O>::with_capacity(selection.true_count() + 1);

                let mut offset = O::zero();
                unsafe { new_offsets.push_unchecked(offset) };
                for idx in selection.indices() {
                    let size = offsets[idx + 1] - offsets[*idx];
                    offset += size;
                    unsafe { new_offsets.push_unchecked(offset) };
                }

                // TODO(ngates): for very dense masks, there may be no point in filtering the elements,
                //  and instead we should construct a view against the unfiltered elements.
                let element_mask = element_mask_from_offsets::<O>(offsets, selection);

                (new_offsets.freeze().into_array(), element_mask)
            });

        let new_elements = array.sliced_elements()?.filter(element_mask)?;

        // SAFETY: new_offsets are monotonically increasing starting from 0 with length
        // true_count + 1, and the elements have been filtered to match.
        Ok(Some(unsafe {
            ListArray::new_unchecked(new_elements, new_offsets, new_validity).into_array()
        }))
    }
}

/// Metadata-only filter for `ListArray` that returns a `ListViewArray`.
///
/// The `FilterKernel` impl above eagerly filters the elements buffer to keep it compact (`List`
/// requires sequential offsets). For destinations where the per-element export cost is zero —
/// DuckDB primitive / decimal / fixed-size children, Arrow → Arrow for any canonical child —
/// that element compaction is pure overhead: the downstream export would pass the elements
/// buffer through zero-copy regardless of whether it's compact.
///
/// This reduce produces a `ListViewArray` instead: original `elements` (zero-copy), plus
/// per-row `(offset, size)` taken from the surviving rows. The export-time prune can still
/// fire on the result when it would actually save work; otherwise the export is free and we
/// have skipped an `O(elements_len)` mask scan plus an `O(num_kept_elements)` copy.
///
/// The reduce only takes the fast path when offsets are host-primitive (the canonical case);
/// otherwise it returns `None` and the `FilterKernel` runs.
impl FilterReduce for List {
    fn filter(array: ArrayView<'_, List>, mask: &Mask) -> VortexResult<Option<ArrayRef>> {
        let selection = match mask {
            Mask::AllTrue(_) | Mask::AllFalse(_) => return Ok(None),
            Mask::Values(v) => v,
        };

        // Only handle the host-primitive offsets case; fall back to the kernel for anything
        // else (e.g. compressed-encoded offsets).
        let Some(offsets_prim) = array.offsets().as_opt::<Primitive>() else {
            return Ok(None);
        };
        if !array.offsets().is_host() {
            return Ok(None);
        }

        let new_validity = match array.validity()? {
            Validity::NonNullable => Validity::NonNullable,
            Validity::AllValid => Validity::AllValid,
            Validity::AllInvalid => Validity::AllInvalid,
            Validity::Array(a) => Validity::Array(a.filter(mask.clone())?),
        };

        let kept = selection.true_count();
        let new_offsets: ArrayRef;
        let new_sizes: ArrayRef;
        match_each_integer_ptype!(offsets_prim.ptype(), |O| {
            let offsets_slice = offsets_prim.as_slice::<O>();
            let mut off_buf = BufferMut::<O>::with_capacity(kept);
            let mut sz_buf = BufferMut::<O>::with_capacity(kept);
            for &i in selection.indices() {
                let start = offsets_slice[i];
                let stop = offsets_slice[i + 1];
                off_buf.push(start);
                sz_buf.push(stop - start);
            }
            new_offsets = PrimitiveArray::new(off_buf.freeze(), Validity::NonNullable).into_array();
            new_sizes = PrimitiveArray::new(sz_buf.freeze(), Validity::NonNullable).into_array();
        });

        // SAFETY:
        // - `new_offsets`/`new_sizes` are non-nullable and have the same length (`kept`).
        // - Each `(offset, size)` pair indexes into a valid range of `elements` because the
        //   pair was taken from a valid `ListArray`'s monotonic offsets.
        let listview = unsafe {
            ListViewArray::new_unchecked(
                array.elements().clone(),
                new_offsets,
                new_sizes,
                new_validity,
            )
        };
        Ok(Some(listview.into_array()))
    }
}
