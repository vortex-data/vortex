// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ExecutionCtx;
use crate::arrays::Constant;
use crate::arrays::constant::vtable::ConstantArray;
use crate::scalar::Scalar;
use crate::vtable::OperationsVTable;

impl OperationsVTable<Constant> for Constant {
    fn scalar_at(
        array: &ConstantArray,
        _index: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        Ok(array.scalar.clone())
    }
}
