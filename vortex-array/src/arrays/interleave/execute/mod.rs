// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Execution logic for [`Interleave`], dispatched on the value type.
//!
//! All values share a type (validated in [`Interleave::check`]), so the
//! physical gather kernel is chosen from the array's dtype. The selector types are an orthogonal
//! concern handled within each kernel.
//!
//! [`Interleave::check`]: super::Interleave::check

mod bool;
mod primitive;

use vortex_error::VortexResult;
use vortex_error::vortex_panic;

use super::Interleave;
use crate::array::Array;
use crate::executor::ExecutionCtx;
use crate::executor::ExecutionResult;

/// Executes an [`InterleaveArray`](super::InterleaveArray) by dispatching on the value type.
pub(super) fn execute(
    array: Array<Interleave>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ExecutionResult> {
    if array.dtype().is_boolean() {
        bool::execute(array, ctx)
    } else if array.dtype().is_primitive() {
        primitive::execute(array, ctx)
    } else {
        vortex_panic!(
            "interleave execution is not implemented for value dtype {}",
            array.dtype()
        )
    }
}
