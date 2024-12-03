use vortex_error::{vortex_err, VortexError, VortexResult};
use crate::ArrayData;
use crate::encoding::Encoding;

pub trait ListMeanFn<Array> {
    fn list_mean(&self, array: &Array) -> VortexResult<ArrayData>;
}

impl<E: Encoding> ListMeanFn<ArrayData> for E
where
    E: ListMeanFn<E::Array>,
    for<'a> &'a E::Array: TryFrom<&'a ArrayData, Error = VortexError>,
{
    fn list_mean(&self, array: &ArrayData) -> VortexResult<ArrayData> {
        let array_ref = <&E::Array>::try_from(array)?;
        let encoding = array
            .encoding()
            .as_any()
            .downcast_ref::<E>()
            .ok_or_else(|| vortex_err!("Mismatched encoding"))?;
        ListMeanFn::list_mean(encoding, array_ref)
    }
}

/// Return the mean of each element in the list array.
pub fn list_mean(array: impl AsRef<ArrayData>) -> VortexResult<ArrayData> {
    let array = array.as_ref();

    if let Some(f) = array.encoding().list_mean_fn() {
        return f.list_mean(array);
    }

    Err(vortex_err!(
        NotImplemented: "list_mean",
        array.encoding().id()
    ))
}