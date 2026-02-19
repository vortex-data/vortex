// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::arrays::ConstantArray;
use crate::arrays::ConstantVTable;
use crate::scalar::Scalar;
use crate::vtable::OperationsVTable;

impl OperationsVTable<ConstantVTable> for ConstantVTable {
    fn scalar_at(array: &ConstantArray, _index: usize) -> VortexResult<Scalar> {
        Ok(array.scalar.clone())
    }
}
