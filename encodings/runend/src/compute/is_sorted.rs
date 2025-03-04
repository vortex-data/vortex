use vortex_array::Array;
use vortex_array::compute::{IsSortedFn, is_sorted, is_strict_sorted};

use crate::{RunEndArray, RunEndEncoding};

impl IsSortedFn<&RunEndArray> for RunEndEncoding {
    fn is_sorted(&self, array: &RunEndArray, strict: bool) -> vortex_error::VortexResult<bool> {
        if strict {
            // For now - we just fall back to the underlying implementation here.
            is_strict_sorted(array.to_canonical()?.as_ref())
        } else {
            is_sorted(array.values())
        }
    }
}
