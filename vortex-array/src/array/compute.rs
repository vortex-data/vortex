use crate::arcref::ArcRef;
use crate::compute::{ComputeFn, Kernel};

/// A trait used to register static kernels for known compute functions.
/// Dynamic kernels must be returned via the `_find_kernel` method.
pub trait ArrayComputeImpl {
    const FILTER: Option<ArcRef<dyn Kernel>> = None;

    /// Fallback implementation to lookup compute kernels at runtime.
    fn _find_kernel(&self, _compute_fn: &dyn ComputeFn) -> Option<ArcRef<dyn Kernel>> {
        None
    }
}
