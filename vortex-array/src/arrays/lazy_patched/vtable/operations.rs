// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::array::OperationsVTable;
use crate::arrays::lazy_patched::LazyPatched;
use crate::scalar::Scalar;

impl OperationsVTable<LazyPatched> for LazyPatched {
    fn scalar_at(
        array: ArrayView<'_, LazyPatched>,
        index: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        Ok(if let Some(scalar) = array.patches().get_patched(index)? {
            scalar
        } else {
            array.inner().scalar_at(index)?
        })
    }
}
