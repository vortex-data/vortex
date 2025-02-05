use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::array::{BoolArray, BoolEncoding};
use crate::compute::CastFn;
use crate::{Array, IntoArray};

impl CastFn<BoolArray> for BoolEncoding {
    fn cast(&self, array: &BoolArray, dtype: &DType) -> VortexResult<Array> {
        assert!(matches!(dtype, DType::Bool(_)));

        // If the types are the same, return the array,
        // otherwise set the array nullability as the dtype nullability.
        if array.dtype() != dtype {
            Ok(BoolArray::new(array.boolean_buffer(), dtype.nullability()).into_array())
        } else {
            Ok(array.clone().into_array())
        }
    }
}
