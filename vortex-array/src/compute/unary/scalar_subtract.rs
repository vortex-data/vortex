use vortex_dtype::DType;
use vortex_error::{vortex_err, VortexError, VortexResult};
use vortex_scalar::Scalar;

use crate::encoding::Encoding;
use crate::{ArrayDType, ArrayData, IntoArrayVariant};

pub trait SubtractScalarFn<Array> {
    fn subtract_scalar(&self, array: &Array, to_subtract: &Scalar) -> VortexResult<ArrayData>;
}

impl<E: Encoding> SubtractScalarFn<ArrayData> for E
where
    E: SubtractScalarFn<E::Array>,
    for<'a> &'a E::Array: TryFrom<&'a ArrayData, Error = VortexError>,
{
    fn subtract_scalar(&self, array: &ArrayData, to_subtract: &Scalar) -> VortexResult<ArrayData> {
        let array_ref = <&E::Array>::try_from(array)?;
        let encoding = array
            .encoding()
            .as_any()
            .downcast_ref::<E>()
            .ok_or_else(|| vortex_err!("Mismatched encoding"))?;
        SubtractScalarFn::subtract_scalar(encoding, array_ref, to_subtract)
    }
}

pub fn subtract_scalar(
    array: impl AsRef<ArrayData>,
    to_subtract: &Scalar,
) -> VortexResult<ArrayData> {
    let array = array.as_ref();

    if let Some(f) = array.encoding().subtract_scalar_fn() {
        return f.subtract_scalar(array, to_subtract);
    }

    // if subtraction is not implemented for the given array type, but the array has a numeric
    // DType, we can flatten the array and apply subtraction to the flattened primitive array
    match array.dtype() {
        DType::Primitive(..) => subtract_scalar(array.clone().into_primitive()?, to_subtract),
        _ => Err(vortex_err!(
            NotImplemented: "scalar_subtract",
            array.encoding().id()
        )),
    }
}
