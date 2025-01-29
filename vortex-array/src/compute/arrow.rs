use arrow_array::{Array, ArrayRef};
use arrow_cast::cast;
use arrow_schema::DataType;
use vortex_dtype::DType;
use vortex_error::{vortex_bail, vortex_err, VortexError, VortexExpect, VortexResult};

use crate::encoding::Encoding;
use crate::stats::ArrayStatistics;
use crate::{ArrayDType, ArrayData, Canonical, IntoArrayData, IntoCanonical};

/// A result type that allows returning the original [`ArrayData`] in the case of unsupported
/// conversion.
pub type ArrowResult = Result<ArrayRef, ArrayData>;

/// Trait for Arrow conversion compute function.
pub trait ToArrowFn<Array> {
    /// Convert the array to an Arrow array of the given type.
    ///
    /// Implementation can return None if the conversion cannot be specialized by this encoding.
    /// In this case, the default conversion via `into_canonical` will be used.
    fn to_arrow(&self, array: Array, data_type: &DataType) -> VortexResult<ArrowResult>;
}

impl<E: Encoding> ToArrowFn<ArrayData> for E
where
    E: ToArrowFn<E::Array>,
    E::Array: TryFrom<ArrayData, Error = VortexError>,
{
    fn to_arrow(&self, array: ArrayData, data_type: &DataType) -> VortexResult<ArrowResult> {
        let encoding = array
            .encoding()
            .as_any()
            .downcast_ref::<E>()
            .vortex_expect("Failed to downcast encoding");
        let array = E::Array::try_from(array)?;
        ToArrowFn::to_arrow(encoding, array, data_type)
    }
}

/// Convert the array to an Arrow array of the given type.
pub fn to_arrow(array: ArrayData, data_type: &DataType) -> VortexResult<ArrayRef> {
    let array = if let Some(f) = array.encoding().to_arrow_fn() {
        // Attempt to invoke the conversion function, returning the owned ArrayData if it was
        // unsuccessful.
        match f.to_arrow(array, data_type)? {
            Ok(arrow_array) => return Ok(arrow_array),
            Err(array) => array,
        }
    } else {
        array
    };

    // Fall back to canonicalizing and then converting.
    let array = array.into_canonical()?.into_array();
    array
        .encoding()
        .to_arrow_fn()
        .vortex_expect("Canonical encodings must implement this function")
        .to_arrow(array, data_type)?
        .map_err(|array| {
            vortex_err!(
                "Failed to convert array {} to Arrow {}",
                array.encoding().id(),
                data_type
            )
        })
}
