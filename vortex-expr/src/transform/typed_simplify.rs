use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::transform::remove_select::remove_select;
use crate::transform::simplify::simplify;
use crate::ExprRef;

pub fn typed_simplify(e: ExprRef, scope_dt: DType) -> VortexResult<ExprRef> {
    let e = simplify(e)?;
    remove_select(e, scope_dt)
}
