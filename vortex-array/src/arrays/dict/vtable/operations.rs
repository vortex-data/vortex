// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::AsPrimitive;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use super::DictVTable;
use crate::Array;
use crate::ArrayRef;
use crate::arrays::PrimitiveVTable;
use crate::arrays::dict::DictArray;
use crate::match_each_integer_ptype;
use crate::scalar::Scalar;
use crate::vtable::OperationsVTable;
use crate::vtable::ValidityHelper;

/// Returns the code at `index` as a `usize`, or `None` if the code is null.
///
/// Fast path: if `codes` is a `PrimitiveArray`, reads directly from the typed slice
/// without allocating a temporary `Scalar`. Falls back to `scalar_at` otherwise.
fn code_at(codes: &ArrayRef, index: usize) -> VortexResult<Option<usize>> {
    if let Some(p) = codes.as_opt::<PrimitiveVTable>() {
        if !p.validity().is_valid(index)? {
            return Ok(None);
        }
        Ok(Some(match_each_integer_ptype!(p.ptype(), |T| {
            p.as_slice::<T>()[index].as_()
        })))
    } else {
        Ok(codes.scalar_at(index)?.as_primitive().as_::<usize>())
    }
}

impl OperationsVTable<DictVTable> for DictVTable {
    fn scalar_at(array: &DictArray, index: usize) -> VortexResult<Scalar> {
        let Some(dict_index) = code_at(array.codes(), index)? else {
            return Ok(Scalar::null(array.dtype().clone()));
        };

        Ok(array
            .values()
            .scalar_at(dict_index)?
            .cast(array.dtype())
            .vortex_expect("Array dtype will only differ by nullability"))
    }
}
