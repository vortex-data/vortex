// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::array::OperationsVTable;
use crate::arrays::ListView;
use crate::arrays::listview::ListViewArrayExt;
use crate::scalar::Scalar;
use crate::scalar::ScalarValue;

impl OperationsVTable<ListView> for ListView {
    fn scalar_at(
        array: ArrayView<'_, ListView>,
        index: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        // By the preconditions we know that the list scalar is not null.
        let list = array.list_elements_at(index)?;
        let scalar_value = ScalarValue::Array(list);
        Scalar::try_new(array.dtype().clone(), Some(scalar_value))
    }
}
