use crate::compute::{ComputeFn, KernelRef};

/// A trait used to register static kernels for known compute functions.
/// Dynamic kernels must be returned via the `_find_kernel` method.
pub trait ArrayComputeImpl {
    const FILTER: Option<KernelRef> = None;

    /// Fallback implementation to lookup compute kernels at runtime.
    fn _find_kernel(&self, _compute_fn: &dyn ComputeFn) -> Option<KernelRef> {
        None
    }
}
