// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use itertools::Itertools;
use vortex_scalar::Scalar;

use crate::ArrayRef;
use crate::arrays::struct_::StructArray;
use crate::arrays::struct_::StructVTable;
use crate::vtable::OperationsVTable;

impl OperationsVTable<StructVTable> for StructVTable {
    fn slice(_array: &StructArray, _range: Range<usize>) -> ArrayRef {
        unreachable!("replaced with SliceArray")
    }

    fn scalar_at(array: &StructArray, index: usize) -> Scalar {
        Scalar::struct_(
            array.dtype().clone(),
            array
                .fields()
                .iter()
                .map(|field| field.scalar_at(index))
                .collect_vec(),
        )
    }
}
