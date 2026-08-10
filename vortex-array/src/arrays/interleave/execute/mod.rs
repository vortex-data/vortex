// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Execution logic for [`Interleave`], dispatched on the value type.
//!
//! All values share a type (validated in [`Interleave::check`]), so the
//! physical gather kernel is chosen from the first value. The selector types are an orthogonal
//! concern handled within each kernel.
//!
//! [`Interleave::check`]: super::Interleave::check

mod bool;
mod primitive;

use num_traits::AsPrimitive;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
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

/// Validate selector lengths and bounds, returning the common output length.
///
/// On success, `branches.len() == rows.len() == len`; for every `i < len`,
/// `branches[i] < num_values` and `rows[i] < value_len(branches[i])`.
fn validate_selectors<A, R, F>(
    num_values: usize,
    value_len: F,
    branches: &[A],
    rows: &[R],
) -> VortexResult<usize>
where
    A: AsPrimitive<usize>,
    R: AsPrimitive<usize>,
    F: Fn(usize) -> usize,
{
    let len = branches.len();
    vortex_ensure!(
        rows.len() == len,
        "interleave selectors differ in length: array_indices {len}, row_indices {}",
        rows.len()
    );

    for i in 0..len {
        let branch = branches[i].as_();
        vortex_ensure!(branch < num_values, "interleave array index out of bounds");
        vortex_ensure!(
            rows[i].as_() < value_len(branch),
            "interleave row index out of bounds"
        );
    }

    Ok(len)
}
