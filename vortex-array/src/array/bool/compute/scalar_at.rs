use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::array::{BoolArray, BoolEncoding};
use crate::compute::unary::ScalarAtFn;
use crate::ArrayDType;

impl ScalarAtFn<BoolArray> for BoolEncoding {
    fn scalar_at(&self, array: &BoolArray, index: usize) -> VortexResult<Scalar> {
        Ok(Scalar::bool(
            array.boolean_buffer().value(index),
            array.dtype().nullability(),
        ))
    }
}
