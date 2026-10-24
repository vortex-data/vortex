// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ExecutionCtx;
use vortex_array::arrays::FixedSizeListArray;
use vortex_error::VortexResult;

use super::IsSortedIteratorExt;
use crate::fixed_size_bytes::FixedSizeBytes;

/// Check whether a fixed-size list of bytes is sorted by byte, with nulls first.
pub(super) fn check_fixed_size_bytes_sorted(
    array: &FixedSizeListArray,
    strict: bool,
    ctx: &mut ExecutionCtx,
) -> VortexResult<bool> {
    let mask = array
        .as_ref()
        .validity()?
        .execute_mask(array.len(), ctx)?
        .to_bit_buffer();
    let bytes = FixedSizeBytes::try_new(array, ctx)?;
    let iter = bytes
        .values()
        .zip(mask.iter())
        .map(|(value, valid)| valid.then_some(value));
    Ok(if strict {
        iter.is_strict_sorted()
    } else {
        iter.is_sorted()
    })
}
