use vortex_dtype::DType;
use vortex_error::{vortex_bail, vortex_err, VortexError, VortexResult};

use crate::encoding::Encoding;
use crate::{ArrayDType, ArrayData, IntoArrayData, IntoArrayVariant};

pub trait InvertFn<Array> {
    /// Logically invert a boolean array. Converts true -> false, false -> true, null -> null.
    fn invert(&self, array: &Array) -> VortexResult<ArrayData>;
}

impl<E: Encoding> InvertFn<ArrayData> for E
where
    E: InvertFn<E::Array>,
    for<'a> &'a E::Array: TryFrom<&'a ArrayData, Error = VortexError>,
{
    fn invert(&self, array: &ArrayData) -> VortexResult<ArrayData> {
        let array_ref = <&E::Array>::try_from(array)?;
        let encoding = array
            .encoding()
            .as_any()
            .downcast_ref::<E>()
            .ok_or_else(|| vortex_err!("Mismatched encoding"))?;
        InvertFn::invert(encoding, array_ref)
    }
}

/// Logically invert a boolean array.
pub fn invert(array: &ArrayData) -> VortexResult<ArrayData> {
    if !matches!(array.dtype(), DType::Bool(..)) {
        vortex_bail!("Expected boolean array, got {}", array.dtype());
    }

    if let Some(f) = array.encoding().invert_fn() {
        let inverted = f.invert(array)?;

        debug_assert_eq!(
            inverted.len(),
            array.len(),
            "Invert length mismatch {}",
            array.encoding().id()
        );
        debug_assert_eq!(
            inverted.dtype(),
            array.dtype(),
            "Invert dtype mismatch {}",
            array.encoding().id()
        );

        return Ok(inverted);
    }

    // Otherwise, we canonicalize into a boolean array and invert.
    log::debug!(
        "No invert implementation found for encoding {}",
        array.encoding().id(),
    );
    invert(&array.clone().into_bool()?.into_array())
}
