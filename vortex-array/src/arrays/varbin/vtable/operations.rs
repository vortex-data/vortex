// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::arrays::varbin::VarBinArray;
use crate::arrays::varbin::VarBinVTable;
use crate::arrays::varbin::varbin_scalar;
use crate::scalar::Scalar;
use crate::vtable::OperationsVTable;

impl OperationsVTable<VarBinVTable> for VarBinVTable {
    fn scalar_at(array: &VarBinArray, index: usize) -> VortexResult<Scalar> {
        Ok(varbin_scalar(array.bytes_at(index), array.dtype()))
    }
}
