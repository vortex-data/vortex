use vortex_error::{vortex_bail, vortex_err, vortex_panic, VortexResult};
use vortex_scalar::Scalar;

use crate::{ArrayDType, ArrayData};

pub trait ScalarAtFn {
    fn scalar_at(&self, index: usize) -> VortexResult<Scalar>;

    fn scalar_at_unchecked(&self, index: usize) -> Scalar;
}

pub fn scalar_at(array: impl AsRef<ArrayData>, index: usize) -> VortexResult<Scalar> {
    let array = array.as_ref();
    if index >= array.len() {
        vortex_bail!(OutOfBounds: index, 0, array.len());
    }

    if !array.with_dyn(|a| a.is_valid(index)) {
        return Ok(Scalar::null(array.dtype().clone()));
    }

    array.with_dyn(|a| {
        a.scalar_at()
            .map(|t| t.scalar_at(index))
            .unwrap_or_else(|| Err(vortex_err!(NotImplemented: "scalar_at", array.encoding().id())))
    })
}

/// Returns a [`Scalar`] value without checking for validity or array bounds. Might panic *OR* return an invalid value if used incorrectly.
pub fn scalar_at_unchecked(array: impl AsRef<ArrayData>, index: usize) -> Scalar {
    let array = array.as_ref();
    array
        .with_dyn(|a| a.scalar_at().map(|s| s.scalar_at_unchecked(index)))
        .unwrap_or_else(|| vortex_panic!(NotImplemented: "scalar_at", array.encoding().id()))
}
