// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_buffer::BitBufferMut;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_mask::MaskIter;

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::ConstantArray;
use crate::arrays::List;
use crate::arrays::ListArray;
use crate::arrays::filter::FilterKernel;
use crate::arrays::list::ListArrayExt;
use crate::arrays::list::ListArraySlotsExt;
use crate::dtype::IntegerPType;
use crate::match_each_integer_ptype;
use crate::validity::Validity;

/// Density threshold for choosing between indices and slices representation when expanding masks.
///
/// When the mask density is below this threshold, we use indices. Otherwise, we use slices.
///
/// Note that this is somewhat arbitrarily chosen...
const MASK_EXPANSION_DENSITY_THRESHOLD: f64 = 0.05;

/// Minimum average list length at which filtering trims unselected leading and trailing elements.
const ELEMENT_RANGE_CROP_MIN_AVERAGE_LIST_LENGTH: usize = 1024;

/// Return the element range to slice before filtering, if doing so removes a sufficiently large
/// unselected prefix or suffix.
fn slice_elements_before_filter<O: IntegerPType>(
    offsets: &[O],
    selection: &MaskIter<'_>,
) -> Option<Range<usize>> {
    let first_offset = offsets.first().map_or(0, |first_offset| first_offset.as_());
    let last_offset = offsets.last().map_or(0, |last_offset| last_offset.as_());
    let element_count = last_offset - first_offset;
    let list_count = offsets.len() - 1;

    if element_count <= list_count.saturating_mul(ELEMENT_RANGE_CROP_MIN_AVERAGE_LIST_LENGTH) {
        return None;
    }

    let (first_index, last_index) = match selection {
        MaskIter::Indices(indices) => (indices[0], indices[indices.len() - 1]),
        MaskIter::Slices(slices) => (slices[0].0, slices[slices.len() - 1].1 - 1),
    };
    let element_range = offsets[first_index].as_()..offsets[last_index + 1].as_();
    let full_element_range = first_offset..last_offset;

    (element_range != full_element_range).then_some(element_range)
}

/// Construct an element mask from contiguous list offsets and an outer-row selection mask. If
/// `element_slice` is present, construct the mask relative to that slice instead of the complete
/// logical element range.
pub fn element_mask_from_offsets<O: IntegerPType>(
    offsets: &[O],
    selection: MaskIter<'_>,
    element_slice: Option<&Range<usize>>,
) -> Mask {
    let (first_offset, last_offset) = element_slice.map_or_else(
        || {
            (
                offsets.first().map_or(0, |offset| offset.as_()),
                offsets.last().map_or(0, |offset| offset.as_()),
            )
        },
        |range| (range.start, range.end),
    );
    let len = last_offset - first_offset;

    let mut mask_builder = BitBufferMut::with_capacity(len);

    match selection {
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

fn append_selected_list_offset<O: IntegerPType>(
    offsets: &[O],
    index: usize,
    offset: &mut O,
    new_offsets: &mut BufferMut<O>,
) {
    *offset += offsets[index + 1] - offsets[index];
    unsafe { new_offsets.push_unchecked(*offset) };
}

fn filtered_offsets<O: IntegerPType>(
    offsets: &[O],
    selection: &MaskIter<'_>,
    selected_count: usize,
) -> Buffer<O> {
    let mut new_offsets = BufferMut::<O>::with_capacity(selected_count + 1);
    let mut offset = O::zero();
    unsafe { new_offsets.push_unchecked(offset) };

    match selection {
        MaskIter::Indices(indices) => {
            for &index in *indices {
                append_selected_list_offset(offsets, index, &mut offset, &mut new_offsets);
            }
        }
        MaskIter::Slices(slices) => {
            for &(start, end) in *slices {
                for index in start..end {
                    append_selected_list_offset(offsets, index, &mut offset, &mut new_offsets);
                }
            }
        }
    }

    new_offsets.freeze()
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
            Mask::Values(values) => values,
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

        let (new_offsets, element_slice, element_mask) =
            match_each_integer_ptype!(offsets.dtype().as_ptype(), |O| {
                let offsets_buffer = offsets.execute::<Buffer<O>>(ctx)?;
                let offsets = offsets_buffer.as_slice();
                let selected_lists = selection.threshold_iter(MASK_EXPANSION_DENSITY_THRESHOLD);
                let new_offsets =
                    filtered_offsets(offsets, &selected_lists, selection.true_count());

                // TODO(ngates): for very dense masks, there may be no point in filtering the elements,
                //  and instead we should construct a view against the unfiltered elements.
                let element_slice = slice_elements_before_filter::<O>(offsets, &selected_lists);
                let element_mask =
                    element_mask_from_offsets::<O>(offsets, selected_lists, element_slice.as_ref());

                (new_offsets.into_array(), element_slice, element_mask)
            });

        let elements = match element_slice {
            Some(range) => array.elements().slice(range)?,
            None => array.sliced_elements()?,
        };
        let new_elements = elements.filter(element_mask)?;

        // SAFETY: new_offsets are monotonically increasing starting from 0 with length
        // true_count + 1, and the elements have been filtered to match.
        Ok(Some(unsafe {
            ListArray::new_unchecked(new_elements, new_offsets, new_validity).into_array()
        }))
    }
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;
    use vortex_error::vortex_bail;
    use vortex_mask::Mask;

    use super::MASK_EXPANSION_DENSITY_THRESHOLD;
    use super::element_mask_from_offsets;
    use super::slice_elements_before_filter;

    #[test]
    fn element_mask_excludes_unselected_prefix_and_suffix() -> VortexResult<()> {
        let Mask::Values(selection) = Mask::from_indices(5, [2]) else {
            vortex_bail!("a partially selective mask uses Mask::Values")
        };

        let offsets = [10u32, 2010, 4010, 6010, 8010, 10010];
        let selected_lists = selection.threshold_iter(MASK_EXPANSION_DENSITY_THRESHOLD);
        let range = slice_elements_before_filter(&offsets, &selected_lists);
        let element_mask = element_mask_from_offsets(&offsets, selected_lists, range.as_ref());

        assert_eq!(range, Some(4010..6010));
        assert!(element_mask.all_true());
        assert_eq!(element_mask.len(), 2000);
        Ok(())
    }

    #[test]
    fn element_mask_retains_gaps_between_selected_lists() -> VortexResult<()> {
        let Mask::Values(selection) = Mask::from_indices(5, [1, 3]) else {
            vortex_bail!("a partially selective mask uses Mask::Values")
        };

        let offsets = [10u32, 2010, 4010, 6010, 8010, 10010];
        let selected_lists = selection.threshold_iter(MASK_EXPANSION_DENSITY_THRESHOLD);
        let range = slice_elements_before_filter(&offsets, &selected_lists);
        let element_mask = element_mask_from_offsets(&offsets, selected_lists, range.as_ref());

        assert_eq!(range, Some(2010..8010));
        assert_eq!(element_mask.len(), 6000);
        assert_eq!(element_mask.true_count(), 4000);
        Ok(())
    }

    #[test]
    fn element_mask_preserves_complete_range_for_short_lists() -> VortexResult<()> {
        let Mask::Values(selection) = Mask::from_indices(5, [2]) else {
            vortex_bail!("a partially selective mask uses Mask::Values")
        };

        let offsets = [10u32, 20, 30, 40, 50, 60];
        let selected_lists = selection.threshold_iter(MASK_EXPANSION_DENSITY_THRESHOLD);
        let range = slice_elements_before_filter(&offsets, &selected_lists);
        let element_mask = element_mask_from_offsets(&offsets, selected_lists, range.as_ref());

        assert_eq!(range, None);
        assert_eq!(element_mask.len(), 50);
        assert_eq!(element_mask.true_count(), 10);
        Ok(())
    }

    #[test]
    fn element_mask_preserves_complete_range_for_edge_spanning_selection() -> VortexResult<()> {
        let Mask::Values(selection) = Mask::from_indices(5, [0, 4]) else {
            vortex_bail!("a partially selective mask uses Mask::Values")
        };

        let offsets = [10u32, 2010, 4010, 6010, 8010, 10010];
        let selected_lists = selection.threshold_iter(MASK_EXPANSION_DENSITY_THRESHOLD);
        let range = slice_elements_before_filter(&offsets, &selected_lists);
        let element_mask = element_mask_from_offsets(&offsets, selected_lists, range.as_ref());

        assert_eq!(range, None);
        assert_eq!(element_mask.len(), 10000);
        assert_eq!(element_mask.true_count(), 4000);
        Ok(())
    }
}
