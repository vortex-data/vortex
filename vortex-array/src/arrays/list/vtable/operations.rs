// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::Arc;

use vortex_scalar::Scalar;

use crate::arrays::{ListArray, ListVTable};
use crate::vtable::{OperationsVTable, ValidityHelper};
use crate::{ArrayRef, IntoArray};

impl OperationsVTable<ListVTable> for ListVTable {
    fn slice(array: &ListArray, range: Range<usize>) -> ArrayRef {
        ListArray::new(
            array.elements().clone(),
            array.offsets().slice(range.start..range.end + 1),
            array.validity().slice(range),
        )
        .into_array()
    }

    fn scalar_at(array: &ListArray, index: usize) -> Scalar {
        // By the preconditions we know that the list scalar is not null.
        let elems = array.list_elements_at(index);
        let scalars: Vec<Scalar> = (0..elems.len()).map(|i| elems.scalar_at(i)).collect();

        Scalar::list(
            Arc::new(elems.dtype().clone()),
            scalars,
            array.dtype().nullability(),
        )
    }
}
