// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Execution logic for [`Interleave`](super::Interleave), dispatched on the value type.
//!
//! All values share a type (validated in [`Interleave::check`](super::Interleave::check)), so the
//! physical gather kernel is chosen from the first value. The selector types are an orthogonal
//! concern handled within each kernel. Only boolean values are implemented today (see [`bool`]).

mod bool;

use vortex_error::VortexResult;
use vortex_error::vortex_panic;

use super::Interleave;
use super::InterleaveArrayExt;
use crate::array::Array;
use crate::executor::ExecutionCtx;
use crate::executor::ExecutionResult;

/// Executes an [`InterleaveArray`](super::InterleaveArray) by dispatching on the value type.
pub(super) fn execute(
    array: Array<Interleave>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ExecutionResult> {
    if array.value(0).dtype().is_boolean() {
        bool::execute(array, ctx)
    } else {
        let value_dtype = array.value(0).dtype().clone();
        vortex_panic!(
            "interleave execution is only implemented for boolean values; value dtype {} is not \
             yet supported",
            value_dtype
        )
    }
}
