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

#[cfg(test)]
mod tests {
    use crate::array::{ListArray, PrimitiveArray};
    use crate::compute::list_mean;
    use crate::validity::Validity;
    use crate::{IntoArrayData, IntoArrayVariant};

    #[test]
    fn test_list_mean() {
        let elements = PrimitiveArray::from(vec![1i32, 2, 3, 4, 5]);
        let offsets = PrimitiveArray::from(vec![0, 2, 4, 5]);
        let validity = Validity::AllValid;

        let list =
            ListArray::try_new(elements.into_array(), offsets.into_array(), validity).unwrap();

        let mean = list_mean(&list).unwrap();
        assert_eq!(mean.into_primitive().unwrap().maybe_null_slice::<f64>(), &[1.5, 3.5, 5.0]);
    }
}
