// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::arrays::ListArray;
use crate::arrays::ListVTable;
use crate::vtable::OperationsVTable;

impl OperationsVTable<ListVTable> for ListVTable {
    fn scalar_at(array: &ListArray, index: usize) -> VortexResult<Scalar> {
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
