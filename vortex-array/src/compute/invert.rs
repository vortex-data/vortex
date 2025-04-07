use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult, vortex_bail};

use crate::encoding::Encoding;
use crate::{Array, ArrayRef, ToCanonical};

pub trait InvertFn<A> {
    /// Logically invert a boolean array. Converts true -> false, false -> true, null -> null.
    fn invert(&self, array: A) -> VortexResult<ArrayRef>;
}

impl<E: Encoding> InvertFn<&dyn Array> for E
where
    E: for<'a> InvertFn<&'a E::Array>,
{
    fn invert(&self, array: &dyn Array) -> VortexResult<ArrayRef> {
        let array_ref = array
            .as_any()
            .downcast_ref::<E::Array>()
            .vortex_expect("Failed to downcast array");
        InvertFn::invert(self, array_ref)
    }
}

/// Logically invert a boolean array.
pub fn invert(array: &dyn Array) -> VortexResult<ArrayRef> {
    if !matches!(array.dtype(), DType::Bool(..)) {
        vortex_bail!("Expected boolean array, got {}", array.dtype());
    }

    if let Some(f) = array.vtable().invert_fn() {
        let inverted = f.invert(array)?;

        assert_eq!(
            inverted.len(),
            array.len(),
            "Invert length mismatch {}",
            array.encoding()
        );
        assert_eq!(
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
    invert(&array.to_bool()?.into_array())
}
