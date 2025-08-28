// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_scalar::Scalar;

use crate::arrays::{FixedSizeListArray, FixedSizeListVTable};
use crate::vtable::{OperationsVTable, ValidityHelper};
use crate::{ArrayRef, IntoArray};

impl OperationsVTable<FixedSizeListVTable> for FixedSizeListVTable {
    fn slice(array: &FixedSizeListArray, range: Range<usize>) -> ArrayRef {
        let start = range.start;
        let end = range.end;

        debug_assert!(
            start <= end && end <= array.len(),
            "slice [{start}..{end}) out of bounds: then len is {}",
            array.len()
        );

        let new_len = end - start;
        let list_size = array.list_size() as usize;

        // SAFETY:
        // - If the `list_size` is 0, then the elements slice has length 0
        // - The length of the sliced elements must be a multiple of the `list_size` since we
        //   multiply both ends by `list_size`
        // - The validity is sliced with equal length to `new_len`
        unsafe {
            FixedSizeListArray::new_unchecked(
                array.elements().slice(start * list_size..end * list_size),
                array.list_size(),
                array.validity().slice(range),
                new_len,
            )
        }
        .into_array()
    }

    fn scalar_at(array: &FixedSizeListArray, index: usize) -> Scalar {
        let list = array.fixed_size_list_at(index);
        let children_elements: Vec<Scalar> = (0..list.len()).map(|i| list.scalar_at(i)).collect();

        debug_assert_eq!(children_elements.len(), array.list_size() as usize);

        Scalar::fixed_size_list(
            list.dtype().clone(),
            children_elements,
            array.dtype.nullability(),
        )
    }
}
