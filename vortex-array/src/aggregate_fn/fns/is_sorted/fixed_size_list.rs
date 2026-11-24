// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BitBuffer;
use vortex_error::VortexResult;

use super::IsSortedIteratorExt;
use crate::ExecutionCtx;
use crate::aggregate_fn::fns::fixed_size_bytes::FixedSizeBytes;
use crate::arrays::FixedSizeBinaryArray;
use crate::arrays::FixedSizeListArray;

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
    Ok(bytes_sorted(&bytes, &mask, strict))
}

/// Check whether a fixed-size binary array is sorted by byte, with nulls first.
pub(super) fn check_fixed_size_binary_sorted(
    array: &FixedSizeBinaryArray,
    strict: bool,
    ctx: &mut ExecutionCtx,
) -> VortexResult<bool> {
    let mask = array
        .validity()?
        .execute_mask(array.len(), ctx)?
        .to_bit_buffer();
    let bytes = FixedSizeBytes::from_fixed_size_binary(array);
    Ok(bytes_sorted(&bytes, &mask, strict))
}

/// Check whether the values of `bytes` whose `mask` bit is set are sorted, with nulls first.
fn bytes_sorted(bytes: &FixedSizeBytes, mask: &BitBuffer, strict: bool) -> bool {
    let iter = bytes
        .values()
        .zip(mask.iter())
        .map(|(value, valid)| valid.then_some(value));
    if strict {
        iter.is_strict_sorted()
    } else {
        iter.is_sorted()
    }
}
