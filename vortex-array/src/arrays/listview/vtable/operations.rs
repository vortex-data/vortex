// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::Arc;

use vortex_scalar::Scalar;

use crate::ArrayRef;
use crate::arrays::ListViewArray;
use crate::arrays::ListViewVTable;
use crate::vtable::OperationsVTable;

impl OperationsVTable<ListViewVTable> for ListViewVTable {
    fn slice(_array: &ListViewArray, _range: Range<usize>) -> ArrayRef {
        unreachable!("replaced with SliceArray")
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
