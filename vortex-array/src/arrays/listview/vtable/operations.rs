// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::array::OperationsVTable;
use crate::array::ProbeState;
use crate::arrays::ListView;
use crate::arrays::listview::ListViewSlots;
use crate::dtype::DType;
use crate::scalar::Scalar;

impl OperationsVTable<ListView> for ListView {
    type ProbeState = ();

    fn probe_scalar(
        state: &mut ProbeState<'_, ListView>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let array = state.array();
        if !state.is_valid(index, ctx)? {
            return Ok(Scalar::null(array.dtype().clone()));
        }

        let DType::List(element_dtype, nullability) = array.dtype() else {
            unreachable!("ListView arrays have a list dtype")
        };
        let start = position_at(state, ListViewSlots::OFFSETS, index, ctx)?;
        let size = position_at(state, ListViewSlots::SIZES, index, ctx)?;
        let mut elements = state
            .slot(ListViewSlots::ELEMENTS)?
            .ok_or_else(|| vortex_err!("ListView elements slot is missing"))?;
        let children = (start..start + size)
            .map(|i| elements.execute_scalar(i, ctx))
            .collect::<VortexResult<Vec<_>>>()?;

        Ok(Scalar::list(
            Arc::clone(element_dtype),
            children,
            *nullability,
        ))
    }

    fn scalar_at(
        array: ArrayView<'_, ListView>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        Self::probe_scalar(&mut ProbeState::once(array), index, ctx)
    }
}

fn position_at(
    state: &mut ProbeState<'_, ListView>,
    slot: usize,
    index: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<usize> {
    Ok(state
        .slot(slot)?
        .ok_or_else(|| vortex_err!("ListView position slot {slot} is missing"))?
        .execute_scalar(index, ctx)?
        .as_primitive()
        .as_::<usize>()
        .vortex_expect("offset or size must fit in usize"))
}
