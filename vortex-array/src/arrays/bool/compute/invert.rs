use std::ops::Not;

use vortex_error::VortexResult;

use crate::arrays::{BoolArray, BoolEncoding, BoolVTable};
use crate::compute::{InvertKernel, InvertKernelAdapter};
use crate::vtable::ValidityChild;
use crate::{Array, ArrayRef, register_kernel};

impl InvertKernel for BoolVTable {
    fn invert(&self, array: &BoolArray) -> VortexResult<ArrayRef> {
        Ok(BoolArray::new(array.boolean_buffer().not(), array.validity().clone()).into_array())
    }
}

register_kernel!(InvertKernelAdapter(BoolVTable).lift());
