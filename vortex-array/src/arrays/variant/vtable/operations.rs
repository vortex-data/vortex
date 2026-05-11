// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use super::merge_typed_scalar_as_variant;
use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::array::OperationsVTable;
use crate::arrays::Variant;
use crate::arrays::variant::VariantArrayExt;
use crate::scalar::Scalar;

impl OperationsVTable<Variant> for Variant {
    fn scalar_at(
        array: ArrayView<'_, Variant>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let fallback = array.core_storage().execute_scalar(index, ctx)?;
        if fallback.is_null() {
            return Ok(fallback);
        }

        let Some(shredded) = array.shredded() else {
            return Ok(fallback);
        };

        let typed = shredded.execute_scalar(index, ctx)?;
        merge_typed_scalar_as_variant(typed, Some(fallback), array.dtype())
    }
}
