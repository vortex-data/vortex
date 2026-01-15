// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_scalar::Scalar;

use crate::arrays::VarBinViewArray;
use crate::arrays::VarBinViewVTable;
use crate::arrays::varbin_scalar;
use crate::vtable::OperationsVTable;

impl OperationsVTable<VarBinViewVTable> for VarBinViewVTable {
    fn scalar_at(array: &VarBinViewArray, index: usize) -> Scalar {
        varbin_scalar(array.bytes_at(index), array.dtype())
    }
}
