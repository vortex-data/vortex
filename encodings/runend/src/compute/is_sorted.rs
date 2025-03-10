use vortex_array::Array;
use vortex_array::compute::{IsSortedFn, is_sorted, is_strict_sorted};

use crate::{RunEndArray, RunEndEncoding};

impl IsSortedFn<&RunEndArray> for RunEndEncoding {
    fn is_sorted(&self, array: &RunEndArray) -> vortex_error::VortexResult<bool> {
        is_sorted(array.values())
    }

    fn is_strict_sorted(&self, array: &RunEndArray) -> vortex_error::VortexResult<bool> {
        is_strict_sorted(array.to_canonical()?.as_ref())
    }
}
