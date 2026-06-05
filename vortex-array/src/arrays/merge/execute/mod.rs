// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Execution logic for [`Merge`](super::Merge), dispatched on the branch value type.
//!
//! All branches share a value type (validated in [`Merge::check`](super::Merge::check)), so the
//! physical merge kernel is chosen from the first branch. The selector type is an orthogonal
//! concern handled within each kernel. Only boolean branches are implemented today (see [`bool`]).

mod bool;

use vortex_error::VortexResult;
use vortex_error::vortex_panic;

use super::Merge;
use super::MergeArrayExt;
use crate::array::Array;
use crate::executor::ExecutionCtx;
use crate::executor::ExecutionResult;

/// Executes a [`MergeArray`](super::MergeArray) by dispatching on the branch value type.
pub(super) fn execute(
    array: Array<Merge>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ExecutionResult> {
    if array.branch(0).dtype().is_boolean() {
        bool::execute(array, ctx)
    } else {
        let branch_dtype = array.branch(0).dtype().clone();
        vortex_panic!(
            "merge execution is only implemented for boolean branches; branch dtype {} is not yet \
             supported",
            branch_dtype
        )
    }
}
