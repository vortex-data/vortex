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

/// Crop when at least `1 / threshold` of referenced elements are unselected.
const PERCENTAGE_REFERENCED_UNSELECTED_ELEMENTS_THRESHOLD: usize = 20;

/// Minimum percentage of referenced-but-unselected prefix and suffix elements required before
/// cropping.
const N_REFERENCED_UNSELECTED_ELEMENTS_THRESHOLD: usize = 1024;

/// Return the element range to construct new mask over and to subsequently filter. In the general case this will be the range
/// of elements referenced by a sublist.
///
/// If there are enough elements in head or tail that are referenced but not selected, it is more efficient
/// to bound the element range to the first and last selected indices. We can then expand the
/// mask only over this subset of elements, slice the elements array, and then filter. This avoids the
/// overhead of potentially wasteful mask reconstruction.
///
/// Returns the range and a flag indicating whether the range is a subinterval of the referenced element range.
fn element_range_from_offsets<O: IntegerPType>(
    offsets: &[O],
    selection: &MaskValuesRef,
) -> (Range<usize>, bool) {
    let referenced_elements_range = offsets[0].as_()..offsets[offsets.len() - 1].as_();

    let selected_indices = selection.indices();
    let first_selected_sublist_index = selected_indices[0];
    let last_selected_sublist_index = selected_indices[selected_indices.len() - 1];
    let selected_elements_range =
        offsets[first_selected_sublist_index].as_()..offsets[last_selected_sublist_index + 1].as_();
    let trimmed_element_count = referenced_elements_range.len() - selected_elements_range.len();

    if trimmed_element_count >= N_REFERENCED_UNSELECTED_ELEMENTS_THRESHOLD
        && trimmed_element_count.saturating_mul(PERCENTAGE_REFERENCED_UNSELECTED_ELEMENTS_THRESHOLD)
            >= referenced_elements_range.len()
    {
        (selected_elements_range, true)
    } else {
        (referenced_elements_range, false)
    }
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

        let (new_offsets, element_range, range_is_subinterval, element_mask) =
            match_each_integer_ptype!(offsets.dtype().as_ptype(), |O| {
                let offsets_buffer = offsets.execute::<Buffer<O>>(ctx)?;
                let offsets = offsets_buffer.as_slice();
                let mut new_offsets = BufferMut::<O>::with_capacity(selection.true_count() + 1);

                let mut offset = O::zero();
                unsafe { new_offsets.push_unchecked(offset) };
                for &index in selection.indices() {
                    offset += offsets[index + 1] - offsets[index];
                    unsafe { new_offsets.push_unchecked(offset) };
                }

                let (element_range, range_is_subinterval) =
                    element_range_from_offsets::<O>(offsets, selection);
                let element_mask =
                    element_mask_from_offsets::<O>(offsets, selection, &element_range);

                (
                    new_offsets.freeze().into_array(),
                    element_range,
                    range_is_subinterval,
                    element_mask,
                )
            });

        let new_elements = if range_is_subinterval {
            array
                .elements()
                .slice(element_range)?
                .filter(element_mask)?
        } else {
            array.sliced_elements()?.filter(element_mask)?
        };

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

        let offsets = [10u32, 20010, 40010, 60010, 80010, 100010];
        let (range, range_is_subinterval) = element_range_from_offsets(&offsets, &selection);
        let element_mask = element_mask_from_offsets(&offsets, &selection, &range);

        assert_eq!(range, 40010..60010);
        assert!(range_is_subinterval);
        assert!(element_mask.all_true());
        assert_eq!(element_mask.len(), 20000);
        Ok(())
    }

    #[test]
    fn element_mask_retains_gaps_between_selected_lists() -> VortexResult<()> {
        let Mask::Values(selection) = Mask::from_indices(5, [1, 3]) else {
            vortex_bail!("a partially selective mask uses Mask::Values")
        };

        let offsets = [10u32, 20010, 40010, 60010, 80010, 100010];
        let (range, range_is_subinterval) = element_range_from_offsets(&offsets, &selection);
        let element_mask = element_mask_from_offsets(&offsets, &selection, &range);

        assert_eq!(range, 20010..80010);
        assert!(range_is_subinterval);
        assert_eq!(element_mask.len(), 60000);
        assert_eq!(element_mask.true_count(), 40000);
        Ok(())
    }

    #[test]
    fn element_range_preserves_complete_range_for_short_lists() -> VortexResult<()> {
        let Mask::Values(selection) = Mask::from_indices(5, [2]) else {
            vortex_bail!("a partially selective mask uses Mask::Values")
        };

        let offsets = [10u32, 20, 30, 40, 50, 60];
        let (range, range_is_subinterval) = element_range_from_offsets(&offsets, &selection);
        let element_mask = element_mask_from_offsets(&offsets, &selection, &range);

        assert_eq!(range, 10..60);
        assert!(!range_is_subinterval);
        assert_eq!(element_mask.len(), 50);
        assert_eq!(element_mask.true_count(), 10);
        Ok(())
    }

    #[test]
    fn element_range_requires_minimum_savings() -> VortexResult<()> {
        let Mask::Values(selection) = Mask::from_indices(2, [0]) else {
            vortex_bail!("a partially selective mask uses Mask::Values")
        };

        let offsets = [0u32, 512, 1024];
        assert_eq!(
            element_range_from_offsets(&offsets, &selection),
            (0..1024, false)
        );
        Ok(())
    }

    #[test]
    fn element_range_requires_sufficient_savings_ratio() -> VortexResult<()> {
        let Mask::Values(selection) = Mask::from_slices(100, vec![(0, 96)]) else {
            vortex_bail!("a partially selective mask uses Mask::Values")
        };
        let offsets = (0..=100).map(|index| index * 1000).collect::<Vec<u32>>();

        assert_eq!(
            element_range_from_offsets(&offsets, &selection),
            (0..100_000, false)
        );
        Ok(())
    }

    #[test]
    fn element_range_crops_at_sufficient_savings_ratio() -> VortexResult<()> {
        let Mask::Values(selection) = Mask::from_slices(100, vec![(0, 95)]) else {
            vortex_bail!("a partially selective mask uses Mask::Values")
        };
        let offsets = (0..=100).map(|index| index * 1000).collect::<Vec<u32>>();

        assert_eq!(
            element_range_from_offsets(&offsets, &selection),
            (0..95_000, true)
        );
        Ok(())
    }

    #[test]
    fn element_mask_preserves_complete_range_for_edge_spanning_selection() -> VortexResult<()> {
        let Mask::Values(selection) = Mask::from_indices(5, [0, 4]) else {
            vortex_bail!("a partially selective mask uses Mask::Values")
        };

        let offsets = [10u32, 20010, 40010, 60010, 80010, 100010];
        let (range, range_is_subinterval) = element_range_from_offsets(&offsets, &selection);
        let element_mask = element_mask_from_offsets(&offsets, &selection, &range);

        assert_eq!(range, 10..100010);
        assert!(!range_is_subinterval);
        assert_eq!(element_mask.len(), 100000);
        assert_eq!(element_mask.true_count(), 40000);
        Ok(())
    }
}
