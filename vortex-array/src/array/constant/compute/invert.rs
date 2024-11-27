use vortex_error::VortexResult;

use crate::array::{ConstantArray, ConstantEncoding};
use crate::compute::InvertFn;
use crate::{ArrayData, ArrayLen, IntoArrayData, ToArrayData};

impl InvertFn<ConstantArray> for ConstantEncoding {
    fn invert(&self, array: &ConstantArray) -> VortexResult<ArrayData> {
        match array.scalar().as_bool().value() {
            None => Ok(array.to_array()),
            Some(b) => Ok(ConstantArray::new(!b, array.len()).into_array()),
        }
    }
}
