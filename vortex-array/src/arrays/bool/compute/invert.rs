use std::ops::Not;

use vortex_error::VortexResult;

use crate::arrays::{BoolArray, BoolEncoding};
use crate::compute::InvertFn;
use crate::{Array, ArrayRef};

impl InvertFn<&BoolArray> for BoolEncoding {
    fn invert(&self, array: &BoolArray) -> VortexResult<ArrayRef> {
        Ok(BoolArray::new(array.boolean_buffer().not(), array.validity().clone()).into_array())
    }
}
