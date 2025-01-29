use vortex_error::{vortex_bail, VortexError, VortexResult};
use vortex_scalar::Scalar;

use crate::encoding::Encoding;
use crate::{ArrayData, IntoArrayData, IntoCanonical};

/// Implementation of fill_null for an encoding.
///
/// SAFETY: the fill value is guaranteed to be non-null.
pub trait FillNullFn<Array> {
    fn fill_null(&self, array: &Array, fill_value: Scalar) -> VortexResult<ArrayData>;
}

impl<E: Encoding> FillNullFn<ArrayData> for E
where
    E: FillNullFn<E::Array>,
    for<'a> &'a E::Array: TryFrom<&'a ArrayData, Error = VortexError>,
{
    fn fill_null(&self, array: &ArrayData, fill_value: Scalar) -> VortexResult<ArrayData> {
        let (array_ref, encoding) = array.try_downcast_ref::<E>()?;
        FillNullFn::fill_null(encoding, array_ref, fill_value)
    }
}

pub fn fill_null(array: impl AsRef<ArrayData>, fill_value: Scalar) -> VortexResult<ArrayData> {
    let array = array.as_ref();
    if !array.dtype().is_nullable() {
        return Ok(array.clone());
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
        array.encoding().id()
    );
    debug_assert_eq!(
        filled.dtype(),
        &array.dtype().with_nullability(fill_value_nullability),
        "FillNull dtype mismatch {}",
        array.encoding().id()
    );

    Ok(filled)
}

fn fill_null_impl(array: &ArrayData, fill_value: Scalar) -> VortexResult<ArrayData> {
    if let Some(fill_null_fn) = array.encoding().fill_null_fn() {
        return fill_null_fn.fill_null(array, fill_value);
    }

    log::debug!("FillNullFn not implemented for {}", array.encoding().id());
    let canonical_arr = array.clone().into_canonical()?.into_array();
    if let Some(fill_null_fn) = canonical_arr.encoding().fill_null_fn() {
        return fill_null_fn.fill_null(&canonical_arr, fill_value);
    }

    vortex_bail!(
        "fill null not implemented for canonical encoding {}, fallback from {}",
        canonical_arr.encoding().id(),
        array.encoding().id()
    )
}
