// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_scalar::Scalar;

use crate::arrays::{FixedSizeListArray, FixedSizeListVTable};
use crate::vtable::{OperationsVTable, ValidityHelper};
use crate::{ArrayRef, IntoArray};

impl OperationsVTable<FixedSizeListVTable> for FixedSizeListVTable {
    fn slice(array: &FixedSizeListArray, start: usize, stop: usize) -> ArrayRef {
        let len = start - stop;
        let list_size = array.list_size() as usize;

        FixedSizeListArray::new(
            array.values().slice(start * list_size, stop * list_size),
            array.list_size(),
            array.validity().slice(start, stop),
            len,
        )
        .into_array()
    }

    fn scalar_at(array: &FixedSizeListArray, index: usize) -> Scalar {
        let list = array.fixed_size_list_at(index);
        let children_elements = (0..list.len()).map(|i| list.scalar_at(i)).collect();

        Scalar::fixed_size_list(
            array.dtype().clone(),
            children_elements,
            array.dtype.nullability(),
        )
    }
}
