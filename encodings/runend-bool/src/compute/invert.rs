use vortex_array::compute::InvertFn;
use vortex_array::{ArrayData, ArrayLen, IntoArrayData};
use vortex_error::VortexResult;

use crate::{RunEndBoolArray, RunEndBoolEncoding};

impl InvertFn<RunEndBoolArray> for RunEndBoolEncoding {
    fn invert(&self, array: &RunEndBoolArray) -> VortexResult<ArrayData> {
        RunEndBoolArray::with_offset_and_size(
            array.ends(),
            // We only need to invert the starting bool
            !array.start(),
            array.validity(),
            array.len(),
            array.offset(),
        )
        .map(|a| a.into_array())
    }
}
