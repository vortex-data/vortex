// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_error::VortexResult;

use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::array::OperationsVTable;
use crate::arrays::List;
use crate::arrays::list::ListArrayExt;
use crate::scalar::Scalar;

impl OperationsVTable<List> for List {
    fn scalar_at(
        array: ArrayView<'_, List>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        // By the preconditions we know that the list scalar is not null.
        // Canonicalize the element slice once so the per-element `execute_scalar` calls below do
        // not re-decode the underlying encoding for every element.
        let elems = array
            .list_elements_at(index)?
            .execute::<Canonical>(ctx)?
            .into_array();
        let scalars: Vec<Scalar> = (0..elems.len())
            .map(|i| elems.execute_scalar(i, ctx))
            .collect::<VortexResult<_>>()?;

        Ok(Scalar::list(
            Arc::new(elems.dtype().clone()),
            scalars,
            array.dtype().nullability(),
        ))
    }
}
