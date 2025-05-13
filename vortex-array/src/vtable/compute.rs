use vortex_error::VortexResult;

use crate::compute::{ComputeFn, InvocationArgs, Output};
use crate::vtable::{NotSupported, VTable};

pub trait ComputeVTable<V: VTable> {
    /// Dynamically invokes the given compute function on the array.
    ///
    /// Returns `None` if the compute function is not supported by this array, otherwise attempts
    /// to invoke the function.
    ///
    /// This can be useful to support compute functions based on some property of the function,
    /// without knowing at compile-time what that function is. For example, any elementwise
    /// function can be pushed-down over a dictionary array and evaluated only on the unique values.
    fn invoke(
        array: &V::Array,
        compute_fn: &ComputeFn,
        args: &InvocationArgs,
    ) -> VortexResult<Option<Output>>;
}

impl<V: VTable> ComputeVTable<V> for NotSupported {
    fn invoke(
        _array: &V::Array,
        _compute_fn: &ComputeFn,
        _args: &InvocationArgs,
    ) -> VortexResult<Option<Output>> {
        Ok(None)
    }
}
