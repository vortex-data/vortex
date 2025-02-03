use vortex_dtype::DType;
use vortex_error::{vortex_bail, VortexResult};

use crate::array::{BoolArray, BoolEncoding};
use crate::compute::CastFn;
use crate::{Array, IntoArray};

impl CastFn<BoolArray> for BoolEncoding {
    fn cast(&self, array: &BoolArray, dtype: &DType) -> VortexResult<Array> {
        let DType::Bool(new_nullability) = dtype else {
            vortex_bail!("cannot cast from {} to {}", array.dtype(), dtype);
        };

        let new_validity = array.validity().cast_nullability(*new_nullability)?;
        BoolArray::try_new(array.boolean_buffer(), new_validity).map(IntoArray::into_array)
    }
}
