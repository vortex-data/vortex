// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Weighted reduction of the valid runs that intersect a logical range.
//!
//! All-valid inputs traverse the end and value slices directly. Partially valid inputs reuse
//! cached valid run indices or iterate the validity bitmap without building indices. Both paths
//! clip the boundary runs to the requested range.
//!
//! Ranges use positions in the unsliced array and must be covered by strictly increasing run ends.
//! Ends, values, and validity describe the same number of runs. Each reduction returns
//! `(sum, is_empty)`, where a `None` sum records overflow. Valid NaNs make the input non-empty even
//! when skipped.
//!
//! Signed arithmetic widens the product, and floating-point arithmetic uses fused multiply-add,
//! so a run can cancel a preceding sum even when its product alone exceeds the result type.

use std::ops::Range;

use num_traits::AsPrimitive;
use num_traits::ToPrimitive;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::UnsignedPType;
use vortex_error::VortexExpect;
use vortex_mask::Mask;
use vortex_mask::MaskValues;

/// Return `None` if the run's product or the accumulated sum exceeds `u64`.
pub(super) fn add_unsigned_run<T: AsPrimitive<u64>>(sum: u64, value: T, len: usize) -> Option<u64> {
    value
        .as_()
        .checked_mul(len as u64)
        .and_then(|product| sum.checked_add(product))
}

/// Widen the product and addition to `i128` before checking whether the result fits in `i64`.
pub(super) fn add_signed_run<T: AsPrimitive<i64>>(sum: i64, value: T, len: usize) -> Option<i64> {
    let product = i128::from(value.as_()) * len as i128;

    i64::try_from(i128::from(sum) + product).ok()
}

/// Add a run with fused multiplication, leaving the sum unchanged for skipped NaNs.
///
/// Always returns `Some`, including for infinite or NaN results, to share the integer callback type.
pub(super) fn add_float_run<T: NativePType>(
    sum: f64,
    value: T,
    len: usize,
    skip_nans: bool,
) -> Option<f64> {
    if skip_nans && value.is_nan() {
        return Some(sum);
    }

    let value = ToPrimitive::to_f64(&value).vortex_expect("Float values fit in f64");

    // Fuse the operations so a finite sum can cancel a product that exceeds f64::MAX.
    Some(value.mul_add(len as f64, sum))
}

/// Sum an independent range without materializing validity indices.
pub(super) fn sum_range<E: UnsignedPType, T: NativePType, A: NativePType>(
    ends: &[E],
    values: &[T],
    validity: &Mask,
    range: Range<usize>,
    add_run: impl Fn(A, T, usize) -> Option<A>,
) -> (Option<A>, bool) {
    if range.is_empty() {
        return (Some(A::default()), true);
    }

    match validity {
        Mask::AllTrue(_) => sum_all_valid(ends, values, range, add_run),
        Mask::AllFalse(_) => (Some(A::default()), true),
        Mask::Values(validity) => sum_partially_valid(ends, values, validity, range, add_run),
    }
}

/// Sum an all-valid range directly from the end and value slices.
///
/// The caller must supply a non-empty range.
fn sum_all_valid<E: UnsignedPType, T: NativePType, A: NativePType>(
    ends: &[E],
    values: &[T],
    range: Range<usize>,
    add_run: impl Fn(A, T, usize) -> Option<A>,
) -> (Option<A>, bool) {
    let mut sum = A::default();
    let first = ends.partition_point(|end| end.as_() <= range.start);
    let mut start = range.start;

    for (&end, &value) in ends[first..].iter().zip(&values[first..]) {
        let end = end.as_();
        if end >= range.end {
            return (add_run(sum, value, range.end - start), false);
        }

        let Some(next) = add_run(sum, value, end - start) else {
            return (None, false);
        };

        sum = next;
        start = end;
    }

    (Some(sum), false)
}

/// Sum a non-empty range using cached indices or a bounded validity bitmap.
fn sum_partially_valid<E: UnsignedPType, T: NativePType, A: NativePType>(
    ends: &[E],
    values: &[T],
    validity: &MaskValues,
    range: Range<usize>,
    add_run: impl Fn(A, T, usize) -> Option<A>,
) -> (Option<A>, bool) {
    if let Some(indices) = validity.cached_indices() {
        let start = indices.partition_point(|&index| ends[index].as_() <= range.start);

        return sum_indexed_runs(
            ends,
            values,
            indices[start..].iter().copied(),
            range,
            add_run,
        );
    }

    let first = ends.partition_point(|end| end.as_() <= range.start);
    let last = ends.partition_point(|end| end.as_() < range.end);

    // Bound the bitmap so finding the next valid run cannot scan beyond the array's slice.
    let bits = validity.bit_buffer().slice(first..=last);
    let indices = bits.set_indices().map(|index| first + index);

    sum_indexed_runs(ends, values, indices, range, add_run)
}

/// Sum valid runs in increasing order, clipping them to the range.
///
/// Every supplied run must end after the range's start.
fn sum_indexed_runs<E: UnsignedPType, T: NativePType, A: NativePType>(
    ends: &[E],
    values: &[T],
    indices: impl Iterator<Item = usize>,
    range: Range<usize>,
    add_run: impl Fn(A, T, usize) -> Option<A>,
) -> (Option<A>, bool) {
    let mut sum = A::default();
    let mut is_empty = true;

    for index in indices {
        let end = ends[index].as_();
        let run_start = if index == 0 { 0 } else { ends[index - 1].as_() };
        let start = run_start.max(range.start);
        if start >= range.end {
            break;
        }

        let overlap_len = end.min(range.end) - start;
        is_empty = false;
        let Some(next) = add_run(sum, values[index], overlap_len) else {
            return (None, false);
        };

        sum = next;

        if end >= range.end {
            break;
        }
    }

    (Some(sum), is_empty)
}
