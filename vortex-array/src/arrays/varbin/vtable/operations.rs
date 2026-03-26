// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ExecutionCtx;
use crate::arrays::VarBin;
use crate::arrays::VarBinArray;
use crate::arrays::varbin::varbin_scalar;
use crate::scalar::Scalar;
use crate::vtable::OperationsVTable;

impl OperationsVTable<VarBin> for VarBin {
    fn scalar_at(
        array: &VarBinArray,
        index: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        Ok(varbin_scalar(array.bytes_at(index), array.dtype()))
    }
}
