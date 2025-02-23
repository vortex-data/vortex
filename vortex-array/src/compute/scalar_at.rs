use vortex_error::{vortex_bail, vortex_err, VortexError, VortexExpect, VortexResult};
use vortex_scalar::Scalar;

use crate::encoding::Encoding;
use crate::{Array, ArrayRef};

/// Implementation of scalar_at for an encoding.
///
/// SAFETY: the index is guaranteed to be within the bounds of the [ArrayRef].
pub trait ScalarAtFn<A> {
    fn scalar_at(&self, array: A, index: usize) -> VortexResult<Scalar>;
}

impl<E: Encoding> ScalarAtFn<&dyn Array> for E
where
    E: for<'a> ScalarAtFn<&'a E::Array>,
{
    fn scalar_at(&self, array: &dyn Array, index: usize) -> VortexResult<Scalar> {
        let array_ref = array
            .as_any()
            .downcast_ref::<E::Array>()
            .vortex_expect("Failed to downcast array");
        let vtable = array.vtable();

        ScalarAtFn::scalar_at(self, array_ref, index)
    }
}

pub fn scalar_at(array: &dyn Array, index: usize) -> VortexResult<Scalar> {
    if index >= array.len() {
        vortex_bail!(OutOfBounds: index, 0, array.len());
    }

    if !array.is_valid(index)? {
        return Ok(Scalar::null(array.dtype().clone()));
    }

    let scalar = array
        .vtable()
        .scalar_at_fn()
        .map(|f| f.scalar_at(array, index))
        .unwrap_or_else(|| Err(vortex_err!(NotImplemented: "scalar_at", array.encoding())))?;

    debug_assert_eq!(
        scalar.dtype(),
        array.dtype(),
        "ScalarAt dtype mismatch {}",
        array.encoding()
    );

    Ok(scalar)
}
