use vortex_dtype::DType;
use vortex_error::{vortex_err, VortexError, VortexResult};

use crate::encoding::Encoding;
use crate::{ArrayDType, ArrayData};

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

    // TODO(ngates): check for null_count if dtype is non-nullable
    array
        .encoding()
        .cast_fn()
        .map(|f| f.cast(array, dtype))
        .unwrap_or_else(|| Err(vortex_err!(NotImplemented: "cast", array.encoding().id())))
}
