// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::array::OperationsVTable;
use crate::array::ProbeState;
use crate::arrays::Struct;
use crate::arrays::struct_::StructArrayExt;
use crate::arrays::struct_::StructSlots;
use crate::scalar::Scalar;
use crate::scalar::ScalarValue;

impl OperationsVTable<Struct> for Struct {
    type ProbeState = ();

    fn probe_scalar(
        state: &mut ProbeState<'_, Struct>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let array = state.array();
        let nfields = array.iter_unmasked_fields().len();
        let mut field_values = Vec::with_capacity(nfields);
        for field in 0..nfields {
            let slot = StructSlots::FIELDS_OFFSET + field;
            let value = state
                .slot(slot)?
                .ok_or_else(|| vortex_err!("Struct field slot {slot} is absent"))?
                .execute_scalar(index, ctx)?;
            field_values.push(value.into_value());
        }
        // SAFETY: The vtable guarantees index is in-bounds and non-null before this is called.
        // Each field read returns a value with the field's own dtype.
        Ok(unsafe {
            Scalar::new_unchecked(
                array.dtype().clone(),
                Some(ScalarValue::Tuple(field_values)),
            )
        })
    }

    fn scalar_at(
        array: ArrayView<'_, Struct>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        Self::probe_scalar(&mut ProbeState::once(array), index, ctx)
    }
}
