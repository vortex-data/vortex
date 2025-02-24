use arrow_array::{new_null_array, ArrayRef as ArrowArrayRef};
use arrow_schema::DataType;
use vortex_dtype::{match_each_integer_ptype, DType};
use vortex_error::{vortex_bail, VortexResult};
use vortex_mask::Mask;
use vortex_scalar::Scalar;

use crate::arrays::null::NullArray;
use crate::arrays::NullEncoding;
use crate::compute::{MaskFn, MinMaxFn, MinMaxResult, ScalarAtFn, SliceFn, TakeFn, ToArrowFn};
use crate::variants::PrimitiveArrayTrait;
use crate::vtable::ComputeVTable;
use crate::{Array, ArrayRef, ToCanonical};

impl ComputeVTable for NullEncoding {
    fn mask_fn(&self) -> Option<&dyn MaskFn<&dyn Array>> {
        Some(self)
    }

    fn scalar_at_fn(&self) -> Option<&dyn ScalarAtFn<&dyn Array>> {
        Some(self)
    }

    fn slice_fn(&self) -> Option<&dyn SliceFn<&dyn Array>> {
        Some(self)
    }

    fn take_fn(&self) -> Option<&dyn TakeFn<&dyn Array>> {
        Some(self)
    }

    fn to_arrow_fn(&self) -> Option<&dyn ToArrowFn<&dyn Array>> {
        Some(self)
    }

    fn min_max_fn(&self) -> Option<&dyn MinMaxFn<&dyn Array>> {
        Some(self)
    }
}

impl MaskFn<&NullArray> for NullEncoding {
    fn mask(&self, array: &NullArray, _mask: Mask) -> VortexResult<ArrayRef> {
        Ok(array.to_array().into_array())
    }
}

impl SliceFn<&NullArray> for NullEncoding {
    fn slice(&self, _array: &NullArray, start: usize, stop: usize) -> VortexResult<ArrayRef> {
        Ok(NullArray::new(stop - start).into_array())
    }
}

impl ScalarAtFn<&NullArray> for NullEncoding {
    fn scalar_at(&self, _array: &NullArray, _index: usize) -> VortexResult<Scalar> {
        Ok(Scalar::null(DType::Null))
    }
}

impl TakeFn<&NullArray> for NullEncoding {
    fn take(&self, array: &NullArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        let indices = indices.to_primitive()?;

        // Enforce all indices are valid
        match_each_integer_ptype!(indices.ptype(), |$T| {
            for index in indices.as_slice::<$T>() {
                if !((*index as usize) < array.len()) {
                    vortex_bail!(OutOfBounds: *index as usize, 0, array.len());
                }
            }
        });

        Ok(NullArray::new(indices.len()).into_array())
    }
}

impl ToArrowFn<&NullArray> for NullEncoding {
    fn to_arrow(
        &self,
        array: &NullArray,
        data_type: &DataType,
    ) -> VortexResult<Option<ArrowArrayRef>> {
        if data_type != &DataType::Null {
            vortex_bail!("Unsupported data type: {data_type}");
        }
        Ok(Some(new_null_array(data_type, array.len())))
    }
}

impl MinMaxFn<&NullArray> for NullEncoding {
    fn min_max(&self, _array: &NullArray) -> VortexResult<Option<MinMaxResult>> {
        Ok(None)
    }
}

#[cfg(test)]
mod test {
    use vortex_buffer::buffer;
    use vortex_dtype::DType;
    use vortex_mask::Mask;

    use crate::array::Array;
    use crate::arrays::null::NullArray;
    use crate::compute::{scalar_at, slice, take};
    use crate::{ArrayExt, IntoArray};

    #[test]
    fn test_slice_nulls() {
        let nulls = NullArray::new(10);
        let sliced = slice(&nulls, 0, 4).unwrap().as_::<NullArray>().clone();

        assert_eq!(sliced.len(), 4);
        assert!(matches!(sliced.validity_mask().unwrap(), Mask::AllFalse(4)));
    }

    #[test]
    fn test_take_nulls() {
        let nulls = NullArray::new(10);
        let taken = take(&nulls, &buffer![0u64, 2, 4, 6, 8].into_array())
            .unwrap()
            .as_::<NullArray>()
            .clone();

        assert_eq!(taken.len(), 5);
        assert!(matches!(taken.validity_mask().unwrap(), Mask::AllFalse(5)));
    }

    #[test]
    fn test_scalar_at_nulls() {
        let nulls = NullArray::new(10);

        let scalar = scalar_at(&nulls, 0).unwrap();
        assert!(scalar.is_null());
        assert_eq!(scalar.dtype().clone(), DType::Null);
    }
}
