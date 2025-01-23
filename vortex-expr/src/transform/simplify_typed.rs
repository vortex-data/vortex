use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::transform::remove_select::remove_select;
use crate::transform::simplify::simplify;
use crate::ExprRef;

/// This pass simplifies an expression under the assumption that ident()/scope as a fixed DType.
/// There is another pass `simplify` that simplifies an expression without any assumptions.
/// This pass also applies simplify.
pub fn simplify_typed(e: ExprRef, scope_dt: &DType) -> VortexResult<ExprRef> {
    let e = simplify(e)?;
    remove_select(e, scope_dt)
}
