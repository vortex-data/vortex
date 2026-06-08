// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_error::VortexResult;

use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::array::OperationsVTable;
use crate::arrays::ListView;
use crate::arrays::listview::ListViewArrayExt;
use crate::scalar::Scalar;

impl OperationsVTable<ListView> for ListView {
    fn scalar_at(
        array: ArrayView<'_, ListView>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        // By the preconditions we know that the list scalar is not null.
        // Canonicalize the element slice once so the per-element `execute_scalar` calls below do
        // not re-decode the underlying encoding for every element.
        let list = array
            .list_elements_at(index)?
            .execute::<Canonical>(ctx)?
            .into_array();
        let children: Vec<Scalar> = (0..list.len())
            .map(|i| list.execute_scalar(i, ctx))
            .collect::<VortexResult<_>>()?;

        Ok(Scalar::list(
            Arc::new(list.dtype().clone()),
            children,
            array.dtype().nullability(),
        ))
    }
}
