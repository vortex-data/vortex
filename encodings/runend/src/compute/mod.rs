// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compute kernels for run-end encoded arrays.
//!
//! Aggregate kernels work on the visible runs of each logical slice. The shared range helper keeps
//! retained runs outside that slice from contributing to their results.

mod cast;
mod compare;
mod fill_null;
pub(crate) mod filter;
pub(crate) mod is_constant;
pub(crate) mod is_sorted;
pub(crate) mod min_max;
pub(crate) mod sum;
pub(crate) mod take;
pub(crate) mod take_from;

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_error::VortexResult;

use crate::RunEnd;
use crate::array::RunEndArrayExt;
use crate::array::RunEndArraySlotsExt;

// Slices can retain leading or trailing runs outside their logical range.
fn logical_values(array: ArrayView<'_, RunEnd>, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
    if array.is_empty() {
        return array.values().slice(0..0);
    }

    let start = array.find_physical_index(0, ctx)?;
    let end = array.find_slice_end_index(array.len(), ctx)?;

    array.values().slice(start..end)
}

#[cfg(test)]
mod tests;
