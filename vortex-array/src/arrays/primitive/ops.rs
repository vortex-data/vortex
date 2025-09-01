// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_dtype::match_each_native_ptype;
use vortex_scalar::Scalar;

use crate::arrays::{PrimitiveArray, PrimitiveVTable};
use crate::vtable::{OperationsVTable, ValidityHelper};
use crate::{ArrayRef, IntoArray};

impl OperationsVTable<PrimitiveVTable> for PrimitiveVTable {
    fn slice(array: &PrimitiveArray, range: Range<usize>) -> ArrayRef {
        match_each_native_ptype!(array.ptype(), |T| {
            PrimitiveArray::new(
                array.buffer::<T>().slice(range.clone()),
                array.validity().slice(range),
            )
            .into_array()
        })
    }

    fn scalar_at(array: &PrimitiveArray, index: usize) -> Scalar {
        match_each_native_ptype!(array.ptype(), |T| {
            Scalar::primitive(array.as_slice::<T>()[index], array.dtype().nullability())
        })
    }
}
