// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::array::OperationsVTable;
use crate::arrays::List;
use crate::arrays::list::ListArrayExt;
use crate::scalar::Scalar;

impl OperationsVTable<List> for List {
    type ProbeState = ();

    fn scalar_at(
        array: ArrayView<'_, List>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        // By the preconditions we know that the list scalar is not null.
        let elems = array.list_elements_at(index, ctx)?;
        Scalar::list_from_array(elems, array.dtype().nullability(), ctx)
    }
}
