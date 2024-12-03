use vortex_error::{vortex_err, VortexError, VortexResult};

use crate::encoding::Encoding;
use crate::ArrayData;

pub trait ListFn<Array> {
    fn sum(&self, array: &Array) -> VortexResult<ArrayData>;
    fn mean(&self, array: &Array) -> VortexResult<ArrayData>;
}

impl<E: Encoding> ListFn<ArrayData> for E
where
    E: ListFn<E::Array>,
    for<'a> &'a E::Array: TryFrom<&'a ArrayData, Error = VortexError>,
{
    fn sum(&self, array: &ArrayData) -> VortexResult<ArrayData> {
        let array_ref = <&E::Array>::try_from(array)?;
        let encoding = array
            .encoding()
            .as_any()
            .downcast_ref::<E>()
            .ok_or_else(|| vortex_err!("Mismatched encoding"))?;
        ListFn::sum(encoding, array_ref)
    }

    fn mean(&self, array: &ArrayData) -> VortexResult<ArrayData> {
        let array_ref = <&E::Array>::try_from(array)?;
        let encoding = array
            .encoding()
            .as_any()
            .downcast_ref::<E>()
            .ok_or_else(|| vortex_err!("Mismatched encoding"))?;
        ListFn::mean(encoding, array_ref)
    }
}

/// Return the sum of each element in the list array.
pub fn list_sum(array: impl AsRef<ArrayData>) -> VortexResult<ArrayData> {
    let array = array.as_ref();

    if let Some(f) = array.encoding().list_fn() {
        return f.sum(array);
    }

    Err(vortex_err!(
        NotImplemented: "list_sum",
        array.encoding().id()
    ))
}

/// Return the mean of each element in the list array.
#[allow(dead_code)]
pub fn list_mean(array: impl AsRef<ArrayData>) -> VortexResult<ArrayData> {
    let array = array.as_ref();

    if let Some(f) = array.encoding().list_fn() {
        return f.mean(array);
    }

    Err(vortex_err!(
        NotImplemented: "list_mean",
        array.encoding().id()
    ))
}
