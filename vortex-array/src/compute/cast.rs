use vortex_dtype::DType;
use vortex_error::{vortex_bail, VortexExpect, VortexResult};

use crate::encoding::Encoding;
use crate::{Array, ArrayRef, IntoArray};

pub trait CastFn<A> {
    fn cast(&self, array: A, dtype: &DType) -> VortexResult<ArrayRef>;
}

impl<E: Encoding> CastFn<&dyn Array> for E
where
    E: for<'a> CastFn<&'a E::Array>,
{
    fn cast(&self, array: &dyn Array, dtype: &DType) -> VortexResult<ArrayRef> {
        let array_ref = array
            .as_any()
            .downcast_ref::<E::Array>()
            .vortex_expect("Failed to downcast array");
        CastFn::cast(self, array_ref, dtype)
    }
}

/// Attempt to cast an array to a desired DType.
///
/// Some array support the ability to narrow or upcast.
pub fn try_cast(array: &dyn Array, dtype: &DType) -> VortexResult<ArrayRef> {
    if array.dtype() == dtype {
        return Ok(array.to_array());
    }

    let casted = try_cast_impl(array, dtype)?;

    debug_assert_eq!(
        casted.len(),
        array.len(),
        "Cast length mismatch {}",
        array.encoding()
    );
    debug_assert_eq!(
        casted.dtype(),
        dtype,
        "Cast dtype mismatch {}",
        array.encoding()
    );

    Ok(casted)
}

fn try_cast_impl(array: &dyn Array, dtype: &DType) -> VortexResult<ArrayRef> {
    // TODO(ngates): check for null_count if dtype is non-nullable
    if let Some(f) = array.vtable().cast_fn() {
        return f.cast(array, dtype);
    }

    // Otherwise, we fall back to the canonical implementations.
    log::debug!(
        "Falling back to canonical cast for encoding {} and dtype {} to {}",
        array.encoding(),
        array.dtype(),
        dtype
    );
    let canonicalized = array.to_canonical()?.into_array();
    if let Some(f) = canonicalized.vtable().cast_fn() {
        return f.cast(&canonicalized, dtype);
    }

    vortex_bail!(
        "No compute kernel to cast array from {} to {}",
        array.dtype(),
        dtype
    )
}
