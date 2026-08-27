// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use num_traits::Zero;
use vortex_buffer::BitBufferMut;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_mask::MaskIter;
use vortex_mask::MaskValuesRef;

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

/// Construct the element range to filter.
fn element_range_from_offsets<O: IntegerPType>(
    offsets: &[O],
    selection: &MaskValuesRef,
) -> Range<usize> {
    let full_first_offset = offsets[0].as_();
    let full_last_offset = offsets[offsets.len() - 1].as_();
    let element_count = full_last_offset - full_first_offset;
    let list_count = offsets.len() - 1;

    let selected_indices = selection.indices();
    let first_index = selected_indices[0];
    let last_index = selected_indices[selected_indices.len() - 1];
    let crop_element_range =
        element_count > list_count.saturating_mul(ELEMENT_RANGE_CROP_MIN_AVERAGE_LIST_LENGTH);
    let (first_offset, last_offset) = if crop_element_range {
        (offsets[first_index].as_(), offsets[last_index + 1].as_())
    } else {
        (full_first_offset, full_last_offset)
    };
    first_offset..last_offset
}

/// Construct an element mask relative to `element_range` from contiguous list offsets and an
/// outer-row selection mask.
pub fn element_mask_from_offsets<O: IntegerPType>(
    offsets: &[O],
    selection: &MaskValuesRef,
    element_range: &Range<usize>,
) -> Mask {
    let first_offset = element_range.start;
    let len = element_range.end - first_offset;

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

        let (new_offsets, element_range, element_mask) =
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
                let element_range = element_range_from_offsets::<O>(offsets, selection);
                let element_mask =
                    element_mask_from_offsets::<O>(offsets, selection, &element_range);

                (
                    new_offsets.freeze().into_array(),
                    element_range,
                    element_mask,
                )
            });

        let new_elements = array
            .elements()
            .slice(element_range)?
            .filter(element_mask)?;

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

    use super::element_mask_from_offsets;
    use super::element_range_from_offsets;

    #[test]
    fn element_mask_excludes_unselected_prefix_and_suffix() -> VortexResult<()> {
        let Mask::Values(selection) = Mask::from_indices(5, [2]) else {
            vortex_bail!("a partially selective mask uses Mask::Values")
        };

        let offsets = [10u32, 2010, 4010, 6010, 8010, 10010];
        let range = element_range_from_offsets(&offsets, &selection);
        let element_mask = element_mask_from_offsets(&offsets, &selection, &range);

        assert_eq!(range, 4010..6010);
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
        let range = element_range_from_offsets(&offsets, &selection);
        let element_mask = element_mask_from_offsets(&offsets, &selection, &range);

        assert_eq!(range, 2010..8010);
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
        let range = element_range_from_offsets(&offsets, &selection);
        let element_mask = element_mask_from_offsets(&offsets, &selection, &range);

        assert_eq!(range, 10..60);
        assert_eq!(element_mask.len(), 50);
        assert_eq!(element_mask.true_count(), 10);
        Ok(())
    }
}
