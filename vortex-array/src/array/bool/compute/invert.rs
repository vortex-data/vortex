use std::ops::Not;

use vortex_error::VortexResult;

use crate::array::{BoolArray, BoolEncoding};
use crate::compute::InvertFn;
use crate::{ArrayData, IntoArrayData};

impl InvertFn<BoolArray> for BoolEncoding {
    fn invert(&self, array: &BoolArray) -> VortexResult<ArrayData> {
        Ok(BoolArray::try_new(array.boolean_buffer().not(), array.validity())?.into_array())
    }
}
