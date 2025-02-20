use vortex_error::{vortex_bail, vortex_err, VortexError, VortexResult};
use vortex_scalar::Scalar;

use crate::encoding::Encoding;
use crate::{Array, ArrayRef};

/// Implementation of scalar_at for an encoding.
///
/// SAFETY: the index is guaranteed to be within the bounds of the [ArrayRef].
pub trait ScalarAtFn<A: ?Sized> {
    fn scalar_at(&self, array: &A, index: usize) -> VortexResult<Scalar>;
}

impl<E: Encoding> ScalarAtFn<dyn Array> for E
where
    E: ScalarAtFn<E::Array>,
    for<'a> &'a E::Array: TryFrom<&'a dyn Array, Error = VortexError>,
{
    fn scalar_at(&self, array: &dyn Array, index: usize) -> VortexResult<Scalar> {
        let (array_ref, encoding) = array.try_downcast_ref::<E>()?;
        ScalarAtFn::scalar_at(encoding, array_ref, index)
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
