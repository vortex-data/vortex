use std::ops::Not;

use vortex_error::VortexResult;

use crate::arrays::{BoolArray, BoolEncoding};
use crate::compute::{InvertKernel, InvertKernelAdapter};
use crate::{Array, ArrayRef, register_kernel};

impl InvertKernel for BoolEncoding {
    fn invert(&self, array: &BoolArray) -> VortexResult<ArrayRef> {
        Ok(BoolArray::new(array.boolean_buffer().not(), array.validity().clone()).into_array())
    }
}

register_kernel!(InvertKernelAdapter(BoolEncoding).lift());
