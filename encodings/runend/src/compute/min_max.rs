use vortex_array::compute::{MinMaxFn, MinMaxResult, min_max};
use vortex_error::VortexResult;

use crate::{RunEndArray, RunEndEncoding};

impl MinMaxFn<&RunEndArray> for RunEndEncoding {
    fn min_max(&self, array: &RunEndArray) -> VortexResult<Option<MinMaxResult>> {
        min_max(array.values())
    }
}
