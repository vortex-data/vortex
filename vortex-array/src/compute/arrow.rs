use arrow_array::{Array, ArrayRef};
use arrow_cast::cast;
use arrow_schema::DataType;
use vortex_dtype::DType;
use vortex_error::{vortex_bail, vortex_err, VortexError, VortexExpect, VortexResult};

use crate::encoding::Encoding;
use crate::stats::ArrayStatistics;
use crate::{ArrayDType, ArrayData, Canonical, IntoArrayData, IntoCanonical};

/// Trait for Arrow conversion compute function.
pub trait ToArrowFn<Array> {
    /// Convert the array to an Arrow array of the given type.
    ///
    /// Implementation can return None if the conversion cannot be specialized by this encoding.
    /// In this case, the default conversion via `into_canonical` will be used.
    fn to_arrow(&self, array: &Array, data_type: &DataType) -> VortexResult<Option<ArrayRef>>;
}

impl<E: Encoding> ToArrowFn<ArrayData> for E
where
    E: ToArrowFn<E::Array>,
    for<'a> &'a E::Array: TryFrom<&'a ArrayData, Error = VortexError>,
{
    fn to_arrow(&self, array: &ArrayData, data_type: &DataType) -> VortexResult<Option<ArrayRef>> {
        let (array_ref, encoding) = array.try_downcast_ref::<E>()?;
        ToArrowFn::to_arrow(encoding, array_ref, data_type)
    }
}

/// Convert the array to an Arrow array of the given type.
pub fn to_arrow<A: AsRef<ArrayData>>(array: A, data_type: &DataType) -> VortexResult<ArrayRef> {
    let array = array.as_ref();

    if let Some(result) = array
        .encoding()
        .to_arrow_fn()
        .and_then(|f| f.to_arrow(array, data_type).transpose())
        .transpose()?
    {
        assert_eq!(
            result.data_type(),
            data_type,
            "ToArrowFn returned wrong data type"
        );
        return Ok(result);
    }

    // Fall back to canonicalizing and then converting.
    let array = array.clone().into_canonical()?.into_array();
    array
        .encoding()
        .to_arrow_fn()
        .vortex_expect("Canonical encodings must implement ToArrowFn")
        .to_arrow(&array, data_type)?
        .ok_or_else(|| {
            vortex_err!(
                "Failed to convert array {} to Arrow {}",
                array.encoding().id(),
                data_type
            )
        })
}
