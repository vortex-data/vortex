// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use super::Dict;
use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::array::OperationsVTable;
use crate::scalar::Scalar;

impl OperationsVTable<Dict> for Dict {
    fn scalar_at(
        array: ArrayView<'_, Dict>,
        index: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let Some(dict_index) = array
            .codes()
            .scalar_at(index)?
            .as_primitive()
            .as_::<usize>()
        else {
            return Ok(Scalar::null(array.dtype().clone()));
        };

        Ok(array
            .values()
            .scalar_at(dict_index)?
            .cast(array.dtype())
            .vortex_expect("Array dtype will only differ by nullability"))
    }
}
