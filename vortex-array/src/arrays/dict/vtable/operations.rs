// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use super::Dict;
use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::array::OperationsVTable;
use crate::arrays::dict::DictArraySlotsExt;
use crate::scalar::Scalar;

impl OperationsVTable<Dict> for Dict {
    fn scalar_at(
        array: ArrayView<'_, Dict>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let Some(dict_index) = array
            .codes()
            .execute_scalar(index, ctx)?
            .as_primitive()
            .as_::<usize>()
        else {
            return Ok(Scalar::null(array.dtype().clone()));
        };

        Ok(array
            .values()
            .execute_scalar(dict_index, ctx)?
            .cast(array.dtype())
            .vortex_expect("Array dtype will only differ by nullability"))
    }
}
