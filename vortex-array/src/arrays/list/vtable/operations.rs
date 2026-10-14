// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::array::OperationsVTable;
use crate::array::ProbeState;
use crate::arrays::List;
use crate::arrays::list::ListSlots;
use crate::dtype::DType;
use crate::scalar::Scalar;

impl OperationsVTable<List> for List {
    type ProbeState = ();

    fn probe_scalar(
        state: &mut ProbeState<'_, List>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let array = state.array();
        if !state.is_valid(index, ctx)? {
            return Ok(Scalar::null(array.dtype().clone()));
        }

        let DType::List(element_dtype, nullability) = array.dtype() else {
            unreachable!("List arrays have a list dtype")
        };
        let start = offset_at(state, index, ctx)?;
        let end = offset_at(state, index + 1, ctx)?;
        let mut elements = state
            .slot(ListSlots::ELEMENTS)?
            .ok_or_else(|| vortex_err!("List elements slot is missing"))?;
        let scalars = (start..end)
            .map(|i| elements.execute_scalar(i, ctx))
            .collect::<VortexResult<Vec<_>>>()?;

        Ok(Scalar::list(element_dtype.clone(), scalars, *nullability))
    }

    fn scalar_at(
        array: ArrayView<'_, List>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        Self::probe_scalar(&mut ProbeState::once(array), index, ctx)
    }
}

fn offset_at(
    state: &mut ProbeState<'_, List>,
    index: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<usize> {
    state
        .slot(ListSlots::OFFSETS)?
        .ok_or_else(|| vortex_err!("List offsets slot is missing"))?
        .execute_scalar(index, ctx)?
        .as_primitive()
        .as_::<usize>()
        .ok_or_else(|| vortex_err!("offset value does not fit in usize"))
}
