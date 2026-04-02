// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::array::OperationsVTable;
use crate::arrays::Struct;
use crate::scalar::Scalar;

impl OperationsVTable<Struct> for Struct {
    fn scalar_at(
        array: ArrayView<'_, Struct>,
        index: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let field_scalars: VortexResult<Vec<Scalar>> = array
            .iter_unmasked_fields()
            .map(|field| field.scalar_at(index))
            .collect();
        // SAFETY: The vtable guarantees index is in-bounds and non-null before this is called.
        // Each field's scalar_at returns a scalar with the field's own dtype.
        Ok(unsafe { Scalar::struct_unchecked(array.dtype().clone(), field_scalars?) })
    }
}
