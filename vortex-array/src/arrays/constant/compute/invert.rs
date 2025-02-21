use vortex_error::VortexResult;

use crate::arrays::{ConstantArray, ConstantEncoding};
use crate::compute::InvertFn;
use crate::{Array, ArrayRef, IntoArray};

impl InvertFn<&ConstantArray> for ConstantEncoding {
    fn invert(&self, array: &ConstantArray) -> VortexResult<ArrayRef> {
        match array.scalar().as_bool().value() {
            None => Ok(array.to_array().into_array()),
            Some(b) => Ok(ConstantArray::new(!b, array.len()).into_array()),
        }
    }
}
