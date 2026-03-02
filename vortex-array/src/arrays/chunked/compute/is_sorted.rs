// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::Array;
use crate::ArrayRef;
use crate::arrays::ChunkedArray;
use crate::arrays::ChunkedVTable;
use crate::compute::IsSortedKernel;
use crate::compute::IsSortedKernelAdapter;
use crate::compute::is_sorted;
use crate::compute::is_strict_sorted;
use crate::register_kernel;

impl IsSortedKernel for ChunkedVTable {
    fn is_sorted(&self, array: &ChunkedArray) -> VortexResult<Option<bool>> {
        is_sorted_impl(array, false, is_sorted)
    }

    fn is_strict_sorted(&self, array: &ChunkedArray) -> VortexResult<Option<bool>> {
        is_sorted_impl(array, true, is_strict_sorted)
    }
}

register_kernel!(IsSortedKernelAdapter(ChunkedVTable).lift());

fn is_sorted_impl(
    array: &ChunkedArray,
    strict: bool,
    reentry_fn: impl Fn(&ArrayRef) -> VortexResult<Option<bool>>,
) -> VortexResult<Option<bool>> {
    let mut first_last = Vec::default();

    for chunk in array.chunks() {
        if chunk.is_empty() {
            continue;
        }

        let first = chunk.scalar_at(0)?;
        let last = chunk.scalar_at(chunk.len() - 1)?;

        first_last.push((first, last));
    }

    let chunk_sorted = first_last
        .iter()
        .is_sorted_by(|a, b| if strict { a.1 < b.0 } else { a.1 <= b.0 });

    if !chunk_sorted {
        return Ok(Some(false));
    }

    for chunk in array.chunks() {
        match reentry_fn(chunk)? {
            None => {
                return Ok(None);
            }
            Some(v) => {
                if !v {
                    return Ok(Some(false));
                }
            }
        }
    }

    Ok(Some(true))
}
