use vortex_array::compute::{invert, InvertFn};
use vortex_array::{ArrayData, ArrayLen, IntoArrayData};
use vortex_error::VortexResult;

use crate::{RunEndArray, RunEndEncoding};

impl InvertFn<RunEndArray> for RunEndEncoding {
    fn invert(&self, array: &RunEndArray) -> VortexResult<ArrayData> {
        RunEndArray::with_offset_and_length(
            array.ends(),
            invert(&array.values())?,
            array.validity(),
            array.len(),
            array.offset(),
        )
        .map(|a| a.into_array())
    }
}
