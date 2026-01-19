// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_scalar::Scalar;

use crate::arrays::ConstantArray;
use crate::arrays::ConstantVTable;
use crate::vtable::OperationsVTable;

impl OperationsVTable<ConstantVTable> for ConstantVTable {
    fn scalar_at(array: &ConstantArray, _index: usize) -> Scalar {
        array.scalar.clone()
    }
}
