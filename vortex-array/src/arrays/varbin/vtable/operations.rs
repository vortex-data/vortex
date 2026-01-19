// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_scalar::Scalar;

use crate::arrays::VarBinArray;
use crate::arrays::VarBinVTable;
use crate::arrays::varbin_scalar;
use crate::vtable::OperationsVTable;

impl OperationsVTable<VarBinVTable> for VarBinVTable {
    fn scalar_at(array: &VarBinArray, index: usize) -> Scalar {
        varbin_scalar(array.bytes_at(index), array.dtype())
    }
}
