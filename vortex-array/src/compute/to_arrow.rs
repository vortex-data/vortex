use arrow_array::{Array as ArrowArray, ArrayRef};
use arrow_schema::DataType;
use vortex_error::{vortex_err, VortexError, VortexExpect, VortexResult};

use crate::arrow::infer_data_type;
use crate::builders::builder_with_capacity;
use crate::encoding::Encoding;
use crate::Array;

/// Trait for Arrow conversion compute function.
pub trait ToArrowFn<A> {
    /// Return the preferred Arrow [`DataType`] of the encoding, or None of the canonical
    /// [`DataType`] for the array's Vortex [`vortex_dtype::DType`] should be used.
    fn preferred_arrow_data_type(&self, _array: &A) -> VortexResult<Option<DataType>> {
        Ok(None)
    }

    /// Convert the array to an Arrow array of the given type.
    ///
    /// Implementation can return None if the conversion cannot be specialized by this encoding.
    /// In this case, the default conversion via `into_canonical` will be used.
    fn to_arrow(&self, array: &A, data_type: &DataType) -> VortexResult<Option<ArrayRef>>;
}

impl<E: Encoding> ToArrowFn<Array> for E
where
    E: ToArrowFn<E::Array>,
    for<'a> &'a E::Array: TryFrom<&'a Array, Error = VortexError>,
{
    fn preferred_arrow_data_type(&self, array: &Array) -> VortexResult<Option<DataType>> {
        let (array_ref, encoding) = array.try_downcast_ref::<E>()?;
        ToArrowFn::preferred_arrow_data_type(encoding, array_ref)
    }

    fn to_arrow(&self, array: &Array, data_type: &DataType) -> VortexResult<Option<ArrayRef>> {
        let (array_ref, encoding) = array.try_downcast_ref::<E>()?;
        ToArrowFn::to_arrow(encoding, array_ref, data_type)
    }
}

/// Return the preferred Arrow [`DataType`] of the array.
pub fn preferred_arrow_data_type<A: AsRef<Array>>(array: A) -> VortexResult<DataType> {
    let array = array.as_ref();

    if let Some(result) = array
        .vtable()
        .to_arrow_fn()
        .and_then(|f| f.preferred_arrow_data_type(array).transpose())
        .transpose()?
    {
        return Ok(result);
    }

    // Otherwise, we use the default.
    infer_data_type(array.dtype())
}

/// Convert the array to an Arrow array of the given type.
pub fn to_arrow<A: AsRef<Array>>(array: A, data_type: &DataType) -> VortexResult<ArrayRef> {
    let array = array.as_ref();

    if let Some(result) = array
        .vtable()
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
    let mut builder = builder_with_capacity(array.dtype(), array.len());
    builder.extend_from_array(array.clone())?;
    let array = builder.finish()?;
    array
        .vtable()
        .to_arrow_fn()
        .vortex_expect("Canonical encodings must implement ToArrowFn")
        .to_arrow(&array, data_type)?
        .ok_or_else(|| {
            vortex_err!(
                "Failed to convert array {} to Arrow {}",
                array.encoding(),
                data_type
            )
        })
}

#[cfg(test)]
mod tests {

    use std::sync::Arc;

    use arrow_array::types::Int32Type;
    use arrow_array::{ArrayRef, PrimitiveArray, StringViewArray, StructArray};
    use arrow_buffer::NullBuffer;

    use crate::arrow::infer_data_type;
    use crate::compute::to_arrow;
    use crate::{array, IntoArray};

    #[test]
    fn test_to_arrow() {
        let array = array::StructArray::from_fields(
            vec![
                (
                    "a",
                    array::PrimitiveArray::from_option_iter(vec![Some(1), None, Some(2)])
                        .into_array(),
                ),
                (
                    "b",
                    array::VarBinViewArray::from_iter_str(vec!["a", "b", "c"]).into_array(),
                ),
            ]
            .as_slice(),
        )
        .unwrap();

        let arrow_array: ArrayRef = Arc::new(
            StructArray::try_from(vec![
                (
                    "a",
                    Arc::new(PrimitiveArray::<Int32Type>::from_iter_values_with_nulls(
                        vec![1, 0, 2],
                        Some(NullBuffer::from(vec![true, false, true])),
                    )) as ArrayRef,
                ),
                (
                    "b",
                    Arc::new(StringViewArray::from(vec![Some("a"), Some("b"), Some("c")])),
                ),
            ])
            .unwrap(),
        );

        assert_eq!(
            &to_arrow(&array, &infer_data_type(array.dtype()).unwrap()).unwrap(),
            &arrow_array
        );
    }
}
