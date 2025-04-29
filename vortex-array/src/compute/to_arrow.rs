use arrow_array::{Array as ArrowArray, ArrayRef as ArrowArrayRef};
use arrow_schema::DataType;
use vortex_error::{VortexExpect, VortexResult, vortex_err};

use crate::encoding::Encoding;
use crate::{Array, IntoArray};

/// Trait for Arrow conversion compute function.
pub trait ToArrowFn<A> {
    /// Return the preferred Arrow [`DataType`] of the encoding, or None of the canonical
    /// [`DataType`] for the array's Vortex [`vortex_dtype::DType`] should be used.
    fn preferred_arrow_data_type(&self, _array: A) -> VortexResult<Option<DataType>> {
        Ok(None)
    }

    /// Convert the array to an Arrow array of the given type.
    ///
    /// Implementation can return None if the conversion cannot be specialized by this encoding.
    /// In this case, the default conversion via `to_canonical` will be used.
    fn to_arrow(&self, array: A, data_type: &DataType) -> VortexResult<Option<ArrowArrayRef>>;
}

impl<E: Encoding> ToArrowFn<&dyn Array> for E
where
    E: for<'a> ToArrowFn<&'a E::Array>,
{
    fn preferred_arrow_data_type(&self, array: &dyn Array) -> VortexResult<Option<DataType>> {
        let array_ref = array
            .as_any()
            .downcast_ref::<E::Array>()
            .vortex_expect("Failed to downcast array");
        ToArrowFn::preferred_arrow_data_type(self, array_ref)
    }

    fn to_arrow(
        &self,
        array: &dyn Array,
        data_type: &DataType,
    ) -> VortexResult<Option<ArrowArrayRef>> {
        let array_ref = array
            .as_any()
            .downcast_ref::<E::Array>()
            .vortex_expect("Failed to downcast array");
        ToArrowFn::to_arrow(self, array_ref, data_type)
    }
}

/// Return the preferred Arrow [`DataType`] of the array.
pub fn preferred_arrow_data_type(array: &dyn Array) -> VortexResult<DataType> {
    if let Some(result) = array
        .vtable()
        .to_arrow_fn()
        .and_then(|f| f.preferred_arrow_data_type(array).transpose())
        .transpose()?
    {
        return Ok(result);
    }

    // Otherwise, we use the default.
    array.dtype().to_arrow_dtype()
}

pub fn to_arrow_preferred(array: &dyn Array) -> VortexResult<ArrowArrayRef> {
    let data_type = preferred_arrow_data_type(array)?;
    to_arrow(array, &data_type)
}

/// Convert the array to an Arrow array of the given type.
pub fn to_arrow(array: &dyn Array, data_type: &DataType) -> VortexResult<ArrowArrayRef> {
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
    let canonical_array = array.to_canonical()?.into_array();
    let arrow_array = canonical_array
        .vtable()
        .to_arrow_fn()
        .vortex_expect("Canonical encodings must implement ToArrowFn")
        .to_arrow(&canonical_array, data_type)?
        .ok_or_else(|| {
            vortex_err!(
                "Failed to convert array {} to Arrow {}",
                canonical_array.encoding(),
                data_type
            )
        })?;

    assert_eq!(array.len(), arrow_array.len());

    Ok(arrow_array)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::types::Int32Type;
    use arrow_array::{ArrayRef, PrimitiveArray, StringViewArray, StructArray};
    use arrow_buffer::NullBuffer;

    use crate::array::Array;
    use crate::arrays;
    use crate::compute::to_arrow;

    #[test]
    fn test_to_arrow() {
        let array = arrays::StructArray::from_fields(
            vec![
                (
                    "a",
                    arrays::PrimitiveArray::from_option_iter(vec![Some(1), None, Some(2)])
                        .into_array(),
                ),
                (
                    "b",
                    arrays::VarBinViewArray::from_iter_str(vec!["a", "b", "c"]).into_array(),
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
            &to_arrow(&array, &array.dtype().to_arrow_dtype().unwrap()).unwrap(),
            &arrow_array
        );
    }
}
