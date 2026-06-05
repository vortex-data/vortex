// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Execution logic for [`Merge`](super::Merge), dispatched on the selector type.
//!
//! Only the boolean-selector case is implemented today (see [`bool`]); integer selectors
//! construct but panic here until the N-branch path is wired up.

mod bool;

use vortex_error::VortexResult;
use vortex_error::vortex_panic;

use super::Merge;
use super::MergeArrayExt;
use crate::array::Array;
use crate::executor::ExecutionCtx;
use crate::executor::ExecutionResult;

/// Executes a [`MergeArray`](super::MergeArray) by dispatching on the selector type.
pub(super) fn execute(
    array: Array<Merge>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ExecutionResult> {
    if array.selector().dtype().is_boolean() {
        bool::execute(array, ctx)
    } else {
        let selector_dtype = array.selector().dtype().clone();
        vortex_panic!(
            "merge execution is only implemented for boolean selectors; selector dtype {} is \
             not yet supported",
            selector_dtype
        )
    }
}
