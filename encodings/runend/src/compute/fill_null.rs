use vortex_array::compute::{fill_null, FillNullFn};
use vortex_array::{ArrayData, ArrayLen, IntoArrayData};
use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::{RunEndArray, RunEndEncoding};

impl FillNullFn<RunEndArray> for RunEndEncoding {
    fn fill_null(&self, array: &RunEndArray, fill_value: Scalar) -> VortexResult<ArrayData> {
        Ok(RunEndArray::with_offset_and_length(
            array.ends(),
            fill_null(array.values(), fill_value)?,
            array.len(),
            array.offset(),
        )?
        .into_array())
    }
}
