use vortex_error::{vortex_err, VortexError, VortexResult};

use crate::encoding::Encoding;
use crate::{ArrayDType, ArrayData};

/// Trait for filling forward on an array, i.e., replacing nulls with the last non-null value.
///
/// If the array is non-nullable, it is returned as-is.
/// If the array is entirely nulls, the fill forward operation returns an array of the same length, filled with the default value of the array's type.
/// The DType of the returned array is the same as the input array; the Validity of the returned array is always either NonNullable or AllValid.
pub trait FillForwardFn<Array> {
    fn fill_forward(&self, array: &Array) -> VortexResult<ArrayData>;
}

impl<E: Encoding> FillForwardFn<ArrayData> for E
where
    E: FillForwardFn<E::Array>,
    for<'a> &'a E::Array: TryFrom<&'a ArrayData, Error = VortexError>,
{
    fn fill_forward(&self, array: &ArrayData) -> VortexResult<ArrayData> {
        let array_ref = <&E::Array>::try_from(array)?;
        let encoding = array
            .encoding()
            .as_any()
            .downcast_ref::<E>()
            .ok_or_else(|| vortex_err!("Mismatched encoding"))?;
        FillForwardFn::fill_forward(encoding, array_ref)
    }
}

pub fn fill_forward(array: impl AsRef<ArrayData>) -> VortexResult<ArrayData> {
    let array = array.as_ref();
    if !array.dtype().is_nullable() {
        return Ok(array.clone());
    }
    array
        .encoding()
        .fill_forward_fn()
        .map(|f| f.fill_forward(array))
        .unwrap_or_else(|| {
            Err(vortex_err!(
                NotImplemented: "fill_forward",
                array.encoding().id()
            ))
        })
}
