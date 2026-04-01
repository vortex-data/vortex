// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_error::VortexResult;

use crate::ExecutionCtx;
use crate::arrays::ListView;
use crate::arrays::listview::vtable::ListViewArray;
use crate::scalar::Scalar;
use crate::vtable::OperationsVTable;

impl OperationsVTable<ListView> for ListView {
    fn scalar_at(
        array: &ListViewArray,
        index: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        // By the preconditions we know that the list scalar is not null.
        let list = array.list_elements_at(index)?;
        let children: Vec<Scalar> = (0..list.len())
            .map(|i| list.scalar_at(i))
            .collect::<VortexResult<_>>()?;

        Ok(Scalar::list(
            Arc::new(list.dtype().clone()),
            children,
            array.dtype.nullability(),
        ))
    }
}
