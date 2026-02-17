// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::Array;
use crate::ArrayRef;
use crate::LEGACY_SESSION;
use crate::VortexSessionExecute;
use crate::expr;
pub use crate::expr::BetweenOptions;
pub use crate::expr::StrictComparison;

/// Compute between (a <= x <= b).
///
/// This is an optimized implementation that is equivalent to `(a <= x) AND (x <= b)`.
///
/// The `BetweenOptions` defines if the lower or upper bounds are strict (exclusive) or non-strict
/// (inclusive).
#[deprecated(note = "Use ArrayBuiltins::between instead")]
pub fn between(
    arr: &dyn Array,
    lower: &dyn Array,
    upper: &dyn Array,
    options: &BetweenOptions,
) -> VortexResult<ArrayRef> {
    expr::between_canonical(
        arr,
        lower,
        upper,
        options,
        &mut LEGACY_SESSION.create_execution_ctx(),
    )
}
