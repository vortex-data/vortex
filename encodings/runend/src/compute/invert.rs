use vortex_array::compute::{invert, InvertFn};
use vortex_array::{ArrayRef, IntoArray};
use vortex_error::VortexResult;

use crate::{RunEndArray, RunEndEncoding};

impl InvertFn<RunEndArray> for RunEndEncoding {
    fn invert(&self, array: &RunEndArray) -> VortexResult<ArrayRef> {
        RunEndArray::with_offset_and_length(
            array.ends(),
            invert(&array.values())?,
            array.len(),
            array.offset(),
        )
        .map(|a| a.into_array())
    }
}
