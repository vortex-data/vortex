use vortex_dtype::DType;
use vortex_error::{vortex_bail, VortexError, VortexResult};

use crate::encoding::Encoding;
use crate::{Array, IntoArray, IntoArrayVariant};

pub trait InvertFn<A> {
    /// Logically invert a boolean array. Converts true -> false, false -> true, null -> null.
    fn invert(&self, array: &A) -> VortexResult<Array>;
}

impl<E: Encoding> InvertFn<Array> for E
where
    E: InvertFn<E::Array>,
    for<'a> &'a E::Array: TryFrom<&'a Array, Error = VortexError>,
{
    fn invert(&self, array: &Array) -> VortexResult<Array> {
        let (array_ref, encoding) = array.try_downcast_ref::<E>()?;
        InvertFn::invert(encoding, array_ref)
    }
}

/// Logically invert a boolean array.
pub fn invert(array: &Array) -> VortexResult<Array> {
    if !matches!(array.dtype(), DType::Bool(..)) {
        vortex_bail!("Expected boolean array, got {}", array.dtype());
    }

    if let Some(f) = array.vtable().invert_fn() {
        let inverted = f.invert(array)?;

        debug_assert_eq!(
            inverted.len(),
            array.len(),
            "Invert length mismatch {}",
            array.encoding()
        );
        debug_assert_eq!(
            inverted.dtype(),
            array.dtype(),
            "Invert dtype mismatch {}",
            array.encoding()
        );

        return Ok(inverted);
    }

    // Otherwise, we canonicalize into a boolean array and invert.
    log::debug!(
        "No invert implementation found for encoding {}",
        array.encoding(),
    );
    invert(&array.clone().into_bool()?.into_array())
}
