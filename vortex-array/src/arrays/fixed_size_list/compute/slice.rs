// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::FixedSizeList;
use crate::arrays::FixedSizeListArray;
use crate::arrays::filter::FilterReduce;
use crate::arrays::slice::SliceReduce;
use crate::vtable::ValidityHelper;

impl SliceReduce for FixedSizeList {
    fn slice(array: &Self::Array, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        let new_len = range.len();
        let list_size = array.list_size() as usize;

        // SAFETY: Slicing preserves FixedSizeListArray invariants
        Ok(Some(
            unsafe {
                FixedSizeListArray::new_unchecked(
                    array
                        .elements()
                        .slice(range.start * list_size..range.end * list_size)?,
                    array.list_size(),
                    array.validity().slice(range)?,
                    new_len,
                )
            }
            .into_array(),
        ))
    }
}

impl FilterReduce for FixedSizeList {
    fn filter(array: &FixedSizeListArray, mask: &Mask) -> VortexResult<Option<ArrayRef>> {
        let list_size = array.list_size() as usize;
        let new_len = mask.true_count();

        let filtered_elements = if list_size == 0 {
            // Degenerate case: elements array is empty regardless of filter.
            array.elements().clone()
        } else {
            let elements_len = array.elements().len();
            let expanded_slices: Vec<(usize, usize)> = mask
                .slices()
                .unwrap_or_else(|| unreachable!(), || unreachable!())
                .iter()
                .map(|&(s, e)| (s * list_size, e * list_size))
                .collect();
            let elements_mask = Mask::from_slices(elements_len, expanded_slices);
            array.elements().filter(elements_mask)?
        };

        // SAFETY: Filtering preserves FixedSizeListArray invariants — each selected list's
        // elements are contiguously preserved, maintaining elements.len() == new_len * list_size.
        Ok(Some(
            unsafe {
                FixedSizeListArray::new_unchecked(
                    filtered_elements,
                    array.list_size(),
                    array.validity().filter(mask)?,
                    new_len,
                )
            }
            .into_array(),
        ))
    }
}
