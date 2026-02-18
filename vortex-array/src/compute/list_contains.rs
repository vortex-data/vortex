// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::Array;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::expr::EmptyOptions;
use crate::expr::ExecutionArgs;
use crate::expr::ListContains as ListContainsVTable;

/// Compute a `Bool`-typed array the same length as `array` where elements is `true` if the list
/// item contains the `value`, `false` otherwise.
///
/// **Deprecated**: Use the expr-based `list_contains` from [`crate::expr`] instead.
#[deprecated(note = "Use the expr-based list_contains kernel pattern instead")]
pub fn list_contains(array: &dyn Array, value: &dyn Array) -> VortexResult<ArrayRef> {
    use crate::expr::VTable as _;
    let mut ctx = ExecutionCtx::new(VortexSession::empty());
    ListContainsVTable.execute(
        &EmptyOptions,
        ExecutionArgs {
            inputs: vec![array.to_array(), value.to_array()],
            row_count: array.len(),
            ctx: &mut ctx,
        },
    )
}
