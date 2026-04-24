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
    fn scalar_at(
        array: ArrayView<'_, FixedSizeList>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        // By the preconditions we know that the list scalar is not null.
        let list = array.fixed_size_list_elements_at(index)?;
        let children_elements: Vec<Scalar> = (0..list.len())
            .map(|i| list.execute_scalar(i, ctx))
            .collect::<VortexResult<_>>()?;

        debug_assert_eq!(children_elements.len(), array.list_size() as usize);

        Ok(Scalar::fixed_size_list(
            list.dtype().clone(),
            children_elements,
            array.dtype().nullability(),
        ))
    }
}
