// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_scalar::Scalar;

use crate::arrays::struct_::StructArray;
use crate::arrays::struct_::StructVTable;
use crate::vtable::OperationsVTable;

impl OperationsVTable<StructVTable> for StructVTable {
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
