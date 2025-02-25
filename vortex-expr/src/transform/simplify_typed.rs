use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::ExprRef;
use crate::transform::remove_merge::remove_merge;
use crate::transform::remove_select::remove_select;
use crate::transform::simplify::simplify;

/// Unlike `simplify`, this function simplifies an expression under the assumption that scope is
/// a known DType. Simplified is applied first and then additional rules.
///
/// NOTE: After typed simplification, returned expressions is "bound" to the scope DType.
///     Applying the returned expression to a different DType may produce wrong results.
pub fn simplify_typed(e: ExprRef, scope_dt: &DType) -> VortexResult<ExprRef> {
    let e = simplify(e)?;

    let e = remove_select(e, scope_dt)?;
    let e = remove_merge(e, scope_dt)?;

    Ok(e)
}
