// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_scalar::Scalar;

use crate::arrays::{
    ConstantArray,
    ConstantVTable,
};
use crate::vtable::OperationsVTable;
use crate::{
    ArrayRef,
    IntoArray,
};

impl OperationsVTable<ConstantVTable> for ConstantVTable {
    fn slice(array: &ConstantArray, range: Range<usize>) -> ArrayRef {
        ConstantArray::new(array.scalar.clone(), range.len()).into_array()
    }

    fn scalar_at(array: &ConstantArray, _index: usize) -> Scalar {
        array.scalar.clone()
    }
}
