// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! RunEnd probes retain their ends and values child probes across lookups.

use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::ExecutionCtx;
use crate::ProbeState;
use crate::arrays::runend::RunEnd;
use crate::arrays::runend::RunEndArrayExt;
use crate::arrays::runend::RunEndSlots;
use crate::match_each_unsigned_integer_ptype;
use crate::scalar::Scalar;
use crate::search_sorted::SearchSorted;
use crate::search_sorted::SearchSortedPrimitiveArray;
use crate::search_sorted::SearchSortedSide;

pub(crate) fn scalar_at(
    state: &mut ProbeState<'_, RunEnd>,
    index: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Scalar> {
    let array = state.array();
    let logical_index = array
        .offset()
        .checked_add(index)
        .ok_or_else(|| vortex_err!("RunEnd logical index overflow"))?;
    // The run is that of the first end strictly greater than the logical index. The search
    // reads the ends through this state's ends probe, so a retained read keeps the child's
    // preparation within and between searches, and a one-off read shares one probe across the
    // search.
    let ends = state
        .slot(RunEndSlots::ENDS)?
        .ok_or_else(|| vortex_err!("RunEnd ends slot is missing"))?;
    let len = ends.array().len();
    let run = match_each_unsigned_integer_ptype!(ends.array().dtype().as_ptype(), |T| {
        SearchSortedPrimitiveArray::<T>::with_probe(ends, ctx)
            .search_sorted(&logical_index, SearchSortedSide::Right)?
            .to_ends_index(len)
    });
    state
        .slot(RunEndSlots::VALUES)?
        .ok_or_else(|| vortex_err!("RunEnd values slot is missing"))?
        .execute_scalar(run, ctx)
}

#[cfg(test)]
mod tests;
