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
        let core_storage = array.core_storage();
        if core_storage.is_invalid(index, ctx)? {
            return Ok(Scalar::null(array.dtype().clone()));
        }

        let Some(shredded) = array.shredded() else {
            return core_storage.execute_scalar(index, ctx);
        };

        let typed = shredded.execute_scalar(index, ctx)?;
        // If the shredded value is null OR we shredded an object we want to merge back together.
        let fallback = (typed.is_null() || typed.dtype().is_struct())
            .then(|| core_storage.execute_scalar(index, ctx))
            .transpose()?;
        merge_typed_scalar_as_variant(typed, fallback, array.dtype())
    }
}
