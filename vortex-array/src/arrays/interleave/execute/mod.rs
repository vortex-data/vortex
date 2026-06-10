// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Execution logic for [`Interleave`](super::Interleave), dispatched on the value type.
//!
//! All values share a type (validated in [`Interleave::check`](super::Interleave::check)), so the
//! physical gather kernel is chosen from the first value. The selector types are an orthogonal
//! concern handled within each kernel. Kernels exist for boolean ([`bool`]), decimal ([`decimal`]),
//! utf8/binary ([`varbinview`]), list ([`listview`]), and struct ([`struct_`]) values.

mod bool;
mod decimal;
mod listview;
mod struct_;
mod varbinview;

use num_traits::AsPrimitive;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;

use super::Interleave;
use super::InterleaveArrayExt;
use crate::array::Array;
use crate::dtype::DType;
use crate::executor::ExecutionCtx;
use crate::executor::ExecutionResult;

/// Executes an [`InterleaveArray`](super::InterleaveArray) by dispatching on the value type.
pub(super) fn execute(
    array: Array<Interleave>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ExecutionResult> {
    let value_dtype = array.value(0).dtype().clone();
    match value_dtype {
        DType::Bool(_) => bool::execute(array, ctx),
        DType::Decimal(..) => decimal::execute(array, ctx),
        DType::Utf8(_) | DType::Binary(_) => varbinview::execute(array, ctx),
        DType::List(..) => listview::execute(array, ctx),
        DType::Struct(..) => struct_::execute(array, ctx),
        other => {
            vortex_panic!("interleave execution is not yet implemented for value dtype {other}")
        }
    }
}

/// Validates the per-row selector bounds: `branches[i] < value_lens.len()` and
/// `rows[i] < value_lens[branches[i]]` for every `i`.
///
/// Checking once up front (returning an error rather than panicking) lets the kernels' gather
/// loops index without per-row branches.
fn check_selector_bounds<A: AsPrimitive<usize>, R: AsPrimitive<usize>>(
    branches: &[A],
    rows: &[R],
    value_lens: &[usize],
) -> VortexResult<()> {
    for (branch, row) in branches.iter().zip(rows) {
        let branch = branch.as_();
        vortex_ensure!(
            branch < value_lens.len(),
            "interleave array index out of bounds"
        );
        vortex_ensure!(
            row.as_() < value_lens[branch],
            "interleave row index out of bounds"
        );
    }
    Ok(())
}
