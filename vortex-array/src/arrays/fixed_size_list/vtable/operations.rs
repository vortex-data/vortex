// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::array::OperationsVTable;
use crate::arrays::FixedSizeList;
use crate::arrays::fixed_size_list::FixedSizeListArrayExt;
use crate::scalar::Scalar;

impl OperationsVTable<FixedSizeList> for FixedSizeList {
    type ProbeState = ();

    fn scalar_at(
        array: ArrayView<'_, FixedSizeList>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        // By the preconditions we know that the list scalar is not null.
        let list = array.fixed_size_list_elements_at(index)?;
        debug_assert_eq!(list.len(), array.list_size() as usize);
        Scalar::fixed_size_list_from_array(list, array.dtype().nullability(), ctx)
    }
}
