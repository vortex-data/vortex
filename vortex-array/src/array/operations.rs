use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::ArrayRef;

/// Implementation trait for common or required array operations.
///
/// There is no good answer for what should be an extensible compute function, vs a built-in
/// operation. At the moment, we promote a compute function to a built-in operation if it is
/// called sufficiently often that the compute dispatch overhead is non-trivial.
pub trait ArrayOperationsImpl {
    // TODO(ngates): add _is_constant here
}
