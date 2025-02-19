use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::arrays::{BoolArray, BoolEncoding};
use crate::compute::ScalarAtFn;

impl ScalarAtFn<BoolArray> for BoolEncoding {
    fn scalar_at(&self, array: &BoolArray, index: usize) -> VortexResult<Scalar> {
        Ok(Scalar::bool(
            array.boolean_buffer().value(index),
            array.dtype().nullability(),
        ))
    }
}
