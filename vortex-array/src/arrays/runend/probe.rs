// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! RunEnd probes retain their decoded ends and their values child probe across lookups.

use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::ExecutionCtx;
use crate::ProbeState;
use crate::arrays::runend::RunEnd;
use crate::arrays::runend::RunEndArrayExt;
use crate::arrays::runend::RunEndArraySlotsExt;
use crate::arrays::runend::RunEndSlots;
use crate::match_each_unsigned_integer_ptype;
use crate::scalar::Scalar;
use crate::search_sorted::SearchSorted;
use crate::search_sorted::SearchSortedPrimitiveArray;
use crate::search_sorted::SearchSortedSide;
use crate::search_sorted::decode_sorted_positions;

/// The run ends a repeated probe keeps, decoded on its first read so that later reads binary
/// search a plain buffer instead of reading the ends a value at a time.
#[derive(Debug, Default)]
pub struct RunEndProbeState {
    ends: Option<Buffer<u64>>,
}

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
    let (retained, mut children) = state.split();
    // The physical index of the run is that of the first end strictly greater than the logical
    // index.
    let run = match retained {
        Some(retained) => {
            if retained.ends.is_none() {
                retained.ends = Some(decode_sorted_positions(array.ends(), ctx)?);
            }
            let ends = retained.ends.as_ref().vortex_expect("decoded above");
            let logical_index = u64::try_from(logical_index)?;
            ends.partition_point(|&end| end <= logical_index)
        }
        None => {
            // A one-off read searches through one ends probe rather than decoding every end.
            let ends = children
                .slot(RunEndSlots::ENDS)?
                .ok_or_else(|| vortex_err!("RunEnd ends slot is missing"))?;
            let len = ends.array().len();
            match_each_unsigned_integer_ptype!(ends.array().dtype().as_ptype(), |T| {
                SearchSortedPrimitiveArray::<T>::with_probe(ends, ctx)
                    .search_sorted(&logical_index, SearchSortedSide::Right)?
                    .to_ends_index(len)
            })
        }
    };
    children
        .slot(RunEndSlots::VALUES)?
        .ok_or_else(|| vortex_err!("RunEnd values slot is missing"))?
        .execute_scalar(run, ctx)
}

#[cfg(test)]
mod tests;
