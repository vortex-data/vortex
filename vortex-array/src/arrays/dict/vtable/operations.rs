// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_scalar::Scalar;

use super::DictVTable;
use crate::Array;
use crate::arrays::dict::DictArray;
use crate::vtable::OperationsVTable;

impl OperationsVTable<DictVTable> for DictVTable {
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
