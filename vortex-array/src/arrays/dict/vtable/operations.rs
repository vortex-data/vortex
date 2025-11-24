// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_error::VortexExpect;
use vortex_scalar::Scalar;

use super::DictVTable;
use crate::arrays::dict::DictArray;
use crate::arrays::{ConstantArray, ConstantVTable};
use crate::vtable::OperationsVTable;
use crate::{Array, ArrayRef, IntoArray};

impl OperationsVTable<DictVTable> for DictVTable {
    fn slice(array: &DictArray, range: Range<usize>) -> ArrayRef {
        let sliced_code = array.codes().slice(range);
        if sliced_code.is::<ConstantVTable>() {
            let code = &sliced_code.scalar_at(0).as_primitive().as_::<usize>();
            return if let Some(code) = code {
                ConstantArray::new(array.values().scalar_at(*code), sliced_code.len()).into_array()
            } else {
                ConstantArray::new(Scalar::null(array.dtype().clone()), sliced_code.len())
                    .to_array()
            };
        }
        // SAFETY: slicing the codes preserves invariants.
        unsafe { DictArray::new_unchecked(sliced_code, array.values().clone()).into_array() }
    }

    fn scalar_at(array: &DictArray, index: usize) -> Scalar {
        let Some(dict_index) = array.codes().scalar_at(index).as_primitive().as_::<usize>() else {
            return Scalar::null(array.dtype().clone());
        };

        array
            .values()
            .scalar_at(dict_index)
            .cast(array.dtype())
            .vortex_expect("Array dtype will only differ by nullability")
    }
}
