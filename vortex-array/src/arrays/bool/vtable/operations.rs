// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_scalar::Scalar;

use crate::arrays::BoolArray;
use crate::arrays::BoolVTable;
use crate::vtable::OperationsVTable;

impl OperationsVTable<BoolVTable> for BoolVTable {
    fn scalar_at(array: &BoolArray, index: usize) -> Scalar {
        Scalar::bool(array.bit_buffer().value(index), array.dtype().nullability())
    }
}

#[cfg(test)]
mod tests {
    use std::iter;

    use super::*;
    use crate::ToCanonical;
    use crate::assert_arrays_eq;

    #[test]
    fn test_slice_hundred_elements() {
        let arr = BoolArray::from_iter(iter::repeat_n(Some(true), 100));
        let sliced_arr = arr.slice(8..16).to_bool();
        assert_eq!(sliced_arr.len(), 8);
        assert_eq!(sliced_arr.bit_buffer().len(), 8);
        assert_eq!(sliced_arr.bit_buffer().offset(), 0);
    }

    #[test]
    fn test_slice() {
        let arr = BoolArray::from_iter([Some(true), Some(true), None, Some(false), None]);
        let sliced_arr = arr.slice(1..4).to_bool();

        assert_arrays_eq!(
            sliced_arr,
            BoolArray::from_iter([Some(true), None, Some(false)])
        );
    }
}
