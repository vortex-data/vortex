// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_dtype::match_each_native_ptype;
use vortex_scalar::Scalar;

use crate::ArrayRef;
use crate::IntoArray;
use crate::LEGACY_SESSION;
use crate::VortexSessionExecute;
use crate::arrays::PrimitiveArray;
use crate::arrays::PrimitiveVTable;
use crate::vtable::OperationsVTable;
use crate::vtable::ValidityHelper;

impl OperationsVTable<PrimitiveVTable> for PrimitiveVTable {
    fn slice(array: &PrimitiveArray, range: Range<usize>) -> ArrayRef {
        let ctx = LEGACY_SESSION.create_execution_ctx();
        match_each_native_ptype!(array.ptype(), |T| {
            PrimitiveArray::new(
                array.buffer::<T>(&ctx).slice(range.clone()),
                array.validity().slice(range),
            )
            .into_array()
        })
    }

    fn scalar_at(array: &PrimitiveArray, index: usize) -> Scalar {
        let ctx = LEGACY_SESSION.create_execution_ctx();
        match_each_native_ptype!(array.ptype(), |T| {
            Scalar::primitive(
                array.as_slice::<T>(&ctx)[index],
                array.dtype().nullability(),
            )
        })
    }
}
