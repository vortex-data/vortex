// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_error::VortexResult;

use crate::ExecutionCtx;
use crate::arrays::List;
use crate::arrays::list::vtable::ListArray;
use crate::scalar::Scalar;
use crate::vtable::OperationsVTable;

impl OperationsVTable<List> for List {
    fn scalar_at(array: &ListArray, index: usize, _ctx: &mut ExecutionCtx) -> VortexResult<Scalar> {
        // By the preconditions we know that the list scalar is not null.
        let elems = array.list_elements_at(index)?;
        let scalars: Vec<Scalar> = (0..elems.len())
            .map(|i| elems.scalar_at(i))
            .collect::<VortexResult<_>>()?;

        Ok(Scalar::list(
            Arc::new(elems.dtype().clone()),
            scalars,
            array.dtype().nullability(),
        ))
    }
}
