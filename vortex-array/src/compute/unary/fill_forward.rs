use vortex_error::{vortex_err, VortexResult};

use crate::{ArrayDType, ArrayData};

/// Trait for filling forward on an array, i.e., replacing nulls with the last non-null value.
///
/// If the array is non-nullable, it is returned as-is.
/// If the array is entirely nulls, the fill forward operation returns an array of the same length, filled with the default value of the array's type.
/// The DType of the returned array is the same as the input array; the Validity of the returned array is always either NonNullable or AllValid.
pub trait FillForwardFn {
    fn fill_forward(&self) -> VortexResult<ArrayData>;
}

pub fn fill_forward(array: impl AsRef<ArrayData>) -> VortexResult<ArrayData> {
    let array = array.as_ref();
    if !array.dtype().is_nullable() {
        return Ok(array.clone());
    }

    array.with_dyn(|a| {
        a.fill_forward()
            .map(|t| t.fill_forward())
            .unwrap_or_else(|| {
                Err(vortex_err!(
                    NotImplemented: "fill_forward",
                    array.encoding().id()
                ))
            })
    })
}
