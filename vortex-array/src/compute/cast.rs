use vortex_dtype::DType;
use vortex_error::{vortex_bail, vortex_err, VortexError, VortexResult};

use crate::encoding::Encoding;
use crate::{ArrayDType, ArrayData, IntoArrayData, IntoCanonical};

pub trait CastFn<Array> {
    fn cast(&self, array: &Array, dtype: &DType) -> VortexResult<ArrayData>;
}

impl<E: Encoding> CastFn<ArrayData> for E
where
    E: CastFn<E::Array>,
    for<'a> &'a E::Array: TryFrom<&'a ArrayData, Error = VortexError>,
{
    fn cast(&self, array: &ArrayData, dtype: &DType) -> VortexResult<ArrayData> {
        let array_ref = <&E::Array>::try_from(array)?;
        let encoding = array
            .encoding()
            .as_any()
            .downcast_ref::<E>()
            .ok_or_else(|| vortex_err!("Mismatched encoding"))?;
        CastFn::cast(encoding, array_ref, dtype)
    }
}

/// Attempt to cast an array to a desired DType.
///
/// Some array support the ability to narrow or upcast.
pub fn try_cast(array: impl AsRef<ArrayData>, dtype: &DType) -> VortexResult<ArrayData> {
    let array = array.as_ref();
    if array.dtype() == dtype {
        return Ok(array.clone());
    }

    let casted = try_cast_impl(array, dtype)?;

    debug_assert_eq!(
        casted.len(),
        array.len(),
        "Cast length mismatch {}",
        array.encoding().id()
    );
    debug_assert_eq!(
        casted.dtype(),
        dtype,
        "Cast dtype mismatch {}",
        array.encoding().id()
    );

    Ok(casted)
}

fn try_cast_impl(array: &ArrayData, dtype: &DType) -> VortexResult<ArrayData> {
    // TODO(ngates): check for null_count if dtype is non-nullable
    if let Some(f) = array.encoding().cast_fn() {
        return f.cast(array, dtype);
    }

    // Otherwise, we fall back to the canonical implementations.
    log::debug!(
        "Falling back to canonical cast for encoding {} and dtype {} to {}",
        array.encoding().id(),
        array.dtype(),
        dtype
    );
    let canonicalized = array.clone().into_canonical()?.into_array();
    if let Some(f) = canonicalized.encoding().cast_fn() {
        return f.cast(&canonicalized, dtype);
    }

    vortex_bail!(
        "No compute kernel to cast array from {} to {}",
        array.dtype(),
        dtype
    )
}
