use vortex_error::{vortex_bail, vortex_err, VortexError, VortexExpect, VortexResult};
use vortex_scalar::Scalar;

use crate::encoding::Encoding;
use crate::{Array, ArrayRef, IntoArray};

/// Implementation of fill_null for an encoding.
///
/// SAFETY: the fill value is guaranteed to be non-null.
pub trait FillNullFn<A> {
    fn fill_null(&self, array: A, fill_value: Scalar) -> VortexResult<ArrayRef>;
}

impl<E: Encoding> FillNullFn<&dyn Array> for E
where
    E: for<'a> FillNullFn<&'a E::Array>,
{
    fn fill_null(&self, array: &dyn Array, fill_value: Scalar) -> VortexResult<ArrayRef> {
        let array_ref = array
            .as_any()
            .downcast_ref::<E::Array>()
            .vortex_expect("Failed to downcast array");
        let vtable = array.vtable();
        let encoding = vtable
            .as_any()
            .downcast_ref::<E>()
            .vortex_expect("Failed to downcast encoding");
        FillNullFn::fill_null(encoding, array_ref, fill_value)
    }
}

pub fn fill_null(array: &dyn Array, fill_value: Scalar) -> VortexResult<ArrayRef> {
    if !array.dtype().is_nullable() {
        return Ok(array.to_array());
    }

    if fill_value.is_null() {
        vortex_bail!("Cannot fill_null with a null value")
    }

    if !array.dtype().eq_ignore_nullability(fill_value.dtype()) {
        vortex_bail!(MismatchedTypes: array.dtype(), fill_value.dtype())
    }

    let fill_value_nullability = fill_value.dtype().nullability();
    let filled = fill_null_impl(array, fill_value)?;

    debug_assert_eq!(
        filled.len(),
        array.len(),
        "FillNull length mismatch {}",
        array.encoding()
    );
    debug_assert_eq!(
        filled.dtype(),
        &array.dtype().with_nullability(fill_value_nullability),
        "FillNull dtype mismatch {}",
        array.encoding()
    );

    Ok(filled)
}

fn fill_null_impl(array: &dyn Array, fill_value: Scalar) -> VortexResult<ArrayRef> {
    if let Some(fill_null_fn) = array.vtable().fill_null_fn() {
        return fill_null_fn.fill_null(array, fill_value);
    }

    log::debug!("FillNullFn not implemented for {}", array.encoding());
    let canonical_arr = array.to_canonical()?.into_array();
    if let Some(fill_null_fn) = canonical_arr.vtable().fill_null_fn() {
        return fill_null_fn.fill_null(&canonical_arr, fill_value);
    }

    vortex_bail!(
        "fill null not implemented for canonical encoding {}, fallback from {}",
        canonical_arr.encoding(),
        array.encoding()
    )
}
