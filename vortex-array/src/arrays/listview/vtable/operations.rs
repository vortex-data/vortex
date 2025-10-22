// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::Arc;

use vortex_scalar::Scalar;

use crate::arrays::{ListViewArray, ListViewVTable};
use crate::vtable::OperationsVTable;
use crate::{ArrayRef, IntoArray};

impl OperationsVTable<ListViewVTable> for ListViewVTable {
    fn slice(array: &ListViewArray, range: Range<usize>) -> ArrayRef {
        let start = range.start;
        let end = range.end;

        // We implement slice by simply slicing the views. We leave the child `elements` array alone
        // since slicing could potentially require calculating which elements are referenced by the
        // new set of views.

        // SAFETY: The preconditions of `slice` mean that the bounds have already been checked, and
        // slicing the components of an existing valid array is still valid.
        unsafe {
            ListViewArray::new_unchecked(
                array.elements().clone(),
                array.offsets().slice(start..end),
                array.sizes().slice(start..end),
                array.validity.slice(start..end),
            )
        }
        .into_array()
    }

    fn scalar_at(array: &ListViewArray, index: usize) -> Scalar {
        // By the preconditions we know that the list scalar is not null.
        let list = array.list_elements_at(index);
        let children: Vec<Scalar> = (0..list.len()).map(|i| list.scalar_at(i)).collect();

        Scalar::list(
            Arc::new(list.dtype().clone()),
            children,
            array.dtype.nullability(),
        )
    }
}
