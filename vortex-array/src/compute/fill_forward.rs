use vortex_error::{vortex_err, VortexExpect, VortexResult};

use crate::encoding::Encoding;
use crate::{Array, ArrayRef};

/// Trait for filling forward on an array, i.e., replacing nulls with the last non-null value.
///
/// If the array is non-nullable, it is returned as-is.
/// If the array is entirely nulls, the fill forward operation returns an array of the same length, filled with the default value of the array's type.
/// The DType of the returned array is the same as the input array; the Validity of the returned array is always either NonNullable or AllValid.
pub trait FillForwardFn<A> {
    fn fill_forward(&self, array: A) -> VortexResult<ArrayRef>;
}

impl<E: Encoding> FillForwardFn<&dyn Array> for E
where
    E: for<'a> FillForwardFn<&'a E::Array>,
{
    fn fill_forward(&self, array: &dyn Array) -> VortexResult<ArrayRef> {
        let array_ref = array
            .as_any()
            .downcast_ref::<E::Array>()
            .vortex_expect("Failed to downcast array");
        FillForwardFn::fill_forward(self, array_ref)
    }
}

pub fn fill_forward(array: &dyn Array) -> VortexResult<ArrayRef> {
    if !array.dtype().is_nullable() {
        return Ok(array.to_array());
    }

    let filled = array
        .vtable()
        .fill_forward_fn()
        .map(|f| f.fill_forward(array))
        .unwrap_or_else(|| {
            Err(vortex_err!(
                NotImplemented: "fill_forward",
                array.encoding()
            ))
        })?;

    debug_assert_eq!(
        filled.len(),
        array.len(),
        "FillForward length mismatch {}",
        array.encoding()
    );
    debug_assert_eq!(
        filled.dtype(),
        array.dtype(),
        "FillForward dtype mismatch {}",
        array.encoding()
    );

    Ok(filled)
}
