// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::array::OperationsVTable;
use crate::arrays::Struct;
use crate::arrays::struct_::StructArrayExt;
use crate::scalar::Scalar;
use crate::scalar::ScalarValue;

impl OperationsVTable<Struct> for Struct {
    fn scalar_at(
        array: ArrayView<'_, Struct>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let field_values = array
            .iter_unmasked_fields()
            .map(|field| field.execute_scalar(index, ctx).map(Scalar::into_value))
            .collect::<VortexResult<Vec<_>>>()?;
        // SAFETY: The vtable guarantees index is in-bounds and non-null before this is called.
        // Each field's scalar_at returns a value with the field's own dtype.
        Ok(unsafe {
            Scalar::new_unchecked(
                array.dtype().clone(),
                Some(ScalarValue::Tuple(field_values)),
            )
        })
    }
}
