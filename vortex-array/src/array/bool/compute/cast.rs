use vortex_dtype::DType;
use vortex_error::{vortex_bail, VortexResult};

use crate::array::{BoolArray, BoolEncoding};
use crate::compute::CastFn;
use crate::{ArrayData, IntoArrayData};

impl CastFn<BoolArray> for BoolEncoding {
    fn cast(&self, array: &BoolArray, dtype: &DType) -> VortexResult<ArrayData> {
        let DType::Bool(new_nullability) = dtype else {
            vortex_bail!(MismatchedTypes: "bool type", dtype);
        };

        BoolArray::try_new(
            array.boolean_buffer(),
            array.validity().with_nullability(*new_nullability)?,
        )
        .map(IntoArrayData::into_array)
    }
}
