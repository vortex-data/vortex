// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;

use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::array::OperationsVTable;
use crate::arrays::Union;
use crate::arrays::union::UnionArrayExt;
use crate::arrays::union::UnionArraySlotsExt;
use crate::scalar::Scalar;

impl OperationsVTable<Union> for Union {
    fn scalar_at(
        array: ArrayView<'_, Union>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let type_id_scalar = array.type_ids().execute_scalar(index, ctx)?;
        let Some(type_id) = type_id_scalar.as_primitive().typed_value::<u8>() else {
            return Ok(Scalar::null(array.dtype().clone()));
        };

        let Some(child_index) = array.variants().tag_to_child_index(type_id) else {
            vortex_panic!("Unknown UnionArray type ID {type_id}")
        };
        let child = array
            .child(child_index)
            .ok_or_else(|| vortex_err!("UnionArray is missing child {child_index}"))?;
        let child_scalar = child.execute_scalar(index, ctx)?;

        Scalar::union(
            array.variants().clone(),
            type_id,
            child_scalar,
            array.dtype().nullability(),
        )
    }
}
