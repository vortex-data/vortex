use vortex_error::VortexResult;

use crate::Array;
use crate::arrays::{ChunkedArray, ChunkedEncoding};
use crate::compute::{IsSortedFn, is_sorted, is_strict_sorted, min_max};

impl IsSortedFn<&ChunkedArray> for ChunkedEncoding {
    fn is_sorted(&self, array: &ChunkedArray) -> VortexResult<bool> {
        is_sorted_impl(array, false, is_sorted)
    }

    fn is_strict_sorted(&self, array: &ChunkedArray) -> VortexResult<bool> {
        is_sorted_impl(array, true, is_strict_sorted)
    }
}

fn is_sorted_impl(
    array: &ChunkedArray,
    strict: bool,
    reentry_fn: impl Fn(&dyn Array) -> VortexResult<bool>,
) -> VortexResult<bool> {
    let mut chunks_min_max = Vec::default();

    for chunk in array.chunks() {
        if !reentry_fn(chunk)? {
            return Ok(false);
        }

        let min_max_vals = min_max(chunk)?;
        chunks_min_max.push(min_max_vals);
    }

    let min_max_sorted = chunks_min_max.iter().flatten().is_sorted_by(|a, b| {
        if strict {
            a.max < b.min
        } else {
            a.max <= b.min
        }
    });

    Ok(min_max_sorted)
}
