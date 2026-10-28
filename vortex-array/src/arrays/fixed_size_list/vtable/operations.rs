// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::array::OperationsVTable;
use crate::array::ProbeState;
use crate::arrays::FixedSizeList;
use crate::arrays::fixed_size_list::FixedSizeListSlots;
use crate::dtype::DType;
use crate::scalar::Scalar;

impl OperationsVTable<FixedSizeList> for FixedSizeList {
    type ProbeState = ();

    fn probe_scalar(
        state: &mut ProbeState<'_, FixedSizeList>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let array = state.array();
        if !state.is_valid(index, ctx)? {
            return Ok(Scalar::null(array.dtype().clone()));
        }

        let DType::FixedSizeList(element_dtype, list_size, nullability) = array.dtype() else {
            unreachable!("FixedSizeList arrays have a fixed-size list dtype")
        };
        let start = *list_size as usize * index;
        let end = start + *list_size as usize;
        let mut elements = state
            .slot(FixedSizeListSlots::ELEMENTS)?
            .ok_or_else(|| vortex_err!("FixedSizeList elements slot is missing"))?;
        let children_elements = (start..end)
            .map(|i| elements.execute_scalar(i, ctx))
            .collect::<VortexResult<Vec<_>>>()?;

        Ok(Scalar::fixed_size_list(
            Arc::clone(element_dtype),
            children_elements,
            *nullability,
        ))
    }

    fn scalar_at(
        array: ArrayView<'_, FixedSizeList>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        Self::probe_scalar(&mut ProbeState::once(array), index, ctx)
    }
}
