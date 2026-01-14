// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_dtype::match_each_native_ptype;
use vortex_scalar::Scalar;

use crate::ArrayRef;
use crate::arrays::PrimitiveArray;
use crate::arrays::PrimitiveVTable;
use crate::vtable::OperationsVTable;

impl OperationsVTable<PrimitiveVTable> for PrimitiveVTable {
    fn slice(_array: &PrimitiveArray, _range: Range<usize>) -> ArrayRef {
        unreachable!("replaced with SliceArray")
    }

    fn scalar_at(array: &PrimitiveArray, index: usize) -> Scalar {
        match_each_native_ptype!(array.ptype(), |T| {
            Scalar::primitive(array.as_slice::<T>()[index], array.dtype().nullability())
        })
    }
}
