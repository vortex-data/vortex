use vortex_error::VortexResult;

use crate::arrays::{ChunkedArray, ChunkedEncoding};
use crate::compute::{
    IsSortedKernel, IsSortedKernelAdapter, is_sorted, is_strict_sorted, scalar_at,
};
use crate::{Array, register_kernel};

impl IsSortedKernel for ChunkedEncoding {
    fn is_sorted(&self, array: &ChunkedArray) -> VortexResult<bool> {
        is_sorted_impl(array, false, is_sorted)
    }

    fn is_strict_sorted(&self, array: &ChunkedArray) -> VortexResult<bool> {
        is_sorted_impl(array, true, is_strict_sorted)
    }
}

register_kernel!(IsSortedKernelAdapter(ChunkedEncoding).lift());

fn is_sorted_impl(
    array: &ChunkedArray,
    strict: bool,
    reentry_fn: impl Fn(&dyn Array) -> VortexResult<bool>,
) -> VortexResult<bool> {
    let mut first_last = Vec::default();

    for chunk in array.chunks() {
        if chunk.is_empty() {
            continue;
        }

        let first = scalar_at(chunk, 0)?;
        let last = scalar_at(chunk, chunk.len() - 1)?;

        first_last.push((first, last));
    }

    let chunk_sorted = first_last
        .iter()
        .is_sorted_by(|a, b| if strict { a.1 < b.0 } else { a.1 <= b.0 });

    if !chunk_sorted {
        return Ok(false);
    }

    for chunk in array.chunks() {
        if !reentry_fn(chunk)? {
            return Ok(false);
        }
    }

    Ok(true)
}
