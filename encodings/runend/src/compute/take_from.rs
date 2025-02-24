use vortex_array::compute::{TakeFromFn, take};
use vortex_array::{Array, ArrayRef};
use vortex_error::VortexResult;

use crate::{RunEndArray, RunEndEncoding};

impl TakeFromFn<&RunEndArray> for RunEndEncoding {
    fn take_from(&self, indices: &RunEndArray, array: &dyn Array) -> VortexResult<ArrayRef> {
        // Order the values to prepare for runend decoding.
        let shuffled = take(array, indices.values())?;
        RunEndArray::try_new(indices.ends().clone(), shuffled).map(|a| a.into_array())
    }
}
