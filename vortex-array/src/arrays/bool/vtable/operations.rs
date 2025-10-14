// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_scalar::Scalar;

use crate::arrays::{BoolArray, BoolVTable};
use crate::vtable::{OperationsVTable, ValidityHelper};
use crate::{ArrayRef, IntoArray};

impl OperationsVTable<BoolVTable> for BoolVTable {
    fn slice(array: &BoolArray, range: Range<usize>) -> ArrayRef {
        BoolArray::from_bool_buffer(
            array.boolean_buffer().slice(range.start, range.len()),
            array.validity().slice(range),
        )
        .into_array()
    }

    fn scalar_at(array: &BoolArray, index: usize) -> Scalar {
        Scalar::bool(
            array.boolean_buffer().value(index),
            array.dtype().nullability(),
        )
    }
}

#[cfg(test)]
mod tests {
    use std::iter;

    use super::*;
    use crate::ToCanonical;

    #[test]
    fn test_slice_hundred_elements() {
        let arr = BoolArray::from_iter(iter::repeat_n(Some(true), 100));
        let sliced_arr = arr.slice(8..16).to_bool();
        assert_eq!(sliced_arr.len(), 8);
        assert_eq!(sliced_arr.boolean_buffer().len(), 8);
        assert_eq!(sliced_arr.boolean_buffer().offset(), 0);
    }

    #[test]
    fn test_slice() {
        let arr = BoolArray::from_iter([Some(true), Some(true), None, Some(false), None]);
        let sliced_arr = arr.slice(1..4).to_bool();

        assert_eq!(sliced_arr.len(), 3);

        let s = sliced_arr.scalar_at(0);
        assert_eq!(s.as_bool().value(), Some(true));

        let s = sliced_arr.scalar_at(1);
        assert!(!sliced_arr.is_valid(1));
        assert!(s.is_null());

        let s = sliced_arr.scalar_at(2);
        assert_eq!(s.as_bool().value(), Some(false));
    }
}
