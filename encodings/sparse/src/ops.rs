// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::arrays::ConstantArray;
use vortex_array::vtable::OperationsVTable;
use vortex_array::{Array, ArrayRef, IntoArray};
use vortex_scalar::Scalar;

use crate::{SparseArray, SparseVTable};

impl OperationsVTable<SparseVTable> for SparseVTable {
    fn slice(array: &SparseArray, range: Range<usize>) -> ArrayRef {
        let new_patches = array.patches().slice(range.clone());

        let Some(new_patches) = new_patches else {
            return ConstantArray::new(array.fill_scalar().clone(), range.len()).into_array();
        };

        // If the number of values in the sparse array matches the array length, then all
        // values are in fact patches, since patches are sorted this is the correct values.
        if new_patches.array_len() == new_patches.values().len() {
            return new_patches.into_values();
        }

        // SAFETY:
        unsafe { SparseArray::new_unchecked(new_patches, array.fill_scalar().clone()).into_array() }
    }

    fn scalar_at(array: &SparseArray, index: usize) -> Scalar {
        array
            .patches()
            .get_patched(index)
            .unwrap_or_else(|| array.fill_scalar().clone())
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::{IntoArray, ToCanonical, assert_arrays_eq};
    use vortex_buffer::buffer;

    use super::*;

    #[test]
    fn slice_partially_invalid() {
        let values = buffer![0u64].into_array();
        let indices = buffer![0u8].into_array();

        let sparse = SparseArray::try_new(indices, values, 1000, 999u64.into()).unwrap();
        let sliced = sparse.slice(0..1000);
        let mut expected = vec![999u64; 1000];
        expected[0] = 0;

        let values = sliced.to_primitive();
        assert_arrays_eq!(values, PrimitiveArray::from_iter(expected));
    }
}
