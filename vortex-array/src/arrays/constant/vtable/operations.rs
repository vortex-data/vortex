// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_scalar::Scalar;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::ConstantVTable;
use crate::vtable::OperationsVTable;

impl OperationsVTable<ConstantVTable> for ConstantVTable {
    fn slice(array: &ConstantArray, range: Range<usize>) -> ArrayRef {
        ConstantArray::new(array.scalar.clone(), range.len()).into_array()
    }

    fn scalar_at(array: &ConstantArray, _index: usize) -> Scalar {
        array.scalar.clone()
    }
}
