// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ExecutionCtx;
use crate::arrays::Constant;
use crate::scalar::Scalar;
use crate::vtable::ArrayView;
use crate::vtable::OperationsVTable;

impl OperationsVTable<Constant> for Constant {
    fn scalar_at(
        array: ArrayView<'_, Constant>,
        _index: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        Ok(array.scalar.clone())
    }
}
