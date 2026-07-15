// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::array::OperationsVTable;
use crate::arrays::Union;
use crate::arrays::union::UnionArrayExt;
use crate::scalar::Scalar;

impl OperationsVTable<Union> for Union {
    fn scalar_at(
        array: ArrayView<'_, Union>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let type_id = array
            .type_ids()
            .execute_scalar(index, ctx)?
            .as_primitive()
            .typed_value::<i8>()
            .ok_or_else(|| vortex_err!("UnionArray type ID at index {index} is null"))?;
        let child_index = array
            .variants()
            .tag_to_child_index(type_id)
            .ok_or_else(|| vortex_err!("Unknown UnionArray type ID {type_id}"))?;
        let child = array
            .child(child_index)
            .ok_or_else(|| vortex_err!("UnionArray is missing child {child_index}"))?;
        let child_scalar = child.execute_scalar(index, ctx)?;

        Scalar::union(array.variants().clone(), type_id, child_scalar)
    }
}
