// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Pushdown of `compare`/`between` against a constant for a *known-sorted* delta array.
//!
//! A delta array decodes each value as a per-lane prefix sum, so there is no algebraic
//! rewrite that discharges a comparison the way Frame-of-Reference shifts a constant. The
//! one regime where it collapses cheaply is when the decoded values are non-decreasing
//! (e.g. unsigned, non-strictly increasing data): the result of any comparison against a
//! constant is then a single contiguous index range, located by binary search over
//! `scalar_at`. Each probe decodes one 1,024-element chunk, so finding both boundaries is
//! `O(chunk * log n)` instead of the `O(n)` full decode.
//!
//! This is gated on the *cached, exact* `IsSorted` statistic. Unsigned deltas alone do not
//! guarantee monotonicity, because deltas are stored with `wrapping_sub`, so a decreasing
//! step wraps to a large unsigned value. `IsSorted` is the sound signal, and it is never
//! computed on the fly here (that would be `O(n)`, defeating the purpose).

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::expr::stats::Stat;
use vortex_array::expr::stats::StatsProviderExt;
use vortex_array::validity::Validity;
use vortex_buffer::BitBufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::Delta;

/// Returns `true` only when the array is *known* (cached and exact) to be non-decreasing.
///
/// Never triggers an `O(n)` computation: if the statistic is absent or inexact we report
/// `false` and let the caller fall back to the default decode-then-compare path.
pub(crate) fn is_known_sorted(array: ArrayView<'_, Delta>) -> bool {
    array.statistics().with_typed_stats_set(|stats| {
        stats
            .get_as::<bool>(Stat::IsSorted)
            .as_exact()
            .unwrap_or(false)
    })
}

/// First index `i` in `[0, len)` whose decoded value is `>= target`, assuming the array is
/// non-decreasing. Returns `len` if no such element exists.
pub(crate) fn lower_bound<T: NativePType + PartialOrd>(
    arr: &ArrayRef,
    len: usize,
    target: T,
    ctx: &mut ExecutionCtx,
) -> VortexResult<usize> {
    partition_point(arr, len, ctx, |v: T| v >= target)
}

/// First index `i` in `[0, len)` whose decoded value is `> target`, assuming the array is
/// non-decreasing. Returns `len` if no such element exists.
pub(crate) fn upper_bound<T: NativePType + PartialOrd>(
    arr: &ArrayRef,
    len: usize,
    target: T,
    ctx: &mut ExecutionCtx,
) -> VortexResult<usize> {
    partition_point(arr, len, ctx, |v: T| v > target)
}

/// Binary search for the first index whose decoded value satisfies `pred`, assuming `pred`
/// is monotone (`false` then `true`) over the non-decreasing array.
fn partition_point<T, F>(
    arr: &ArrayRef,
    len: usize,
    ctx: &mut ExecutionCtx,
    mut pred: F,
) -> VortexResult<usize>
where
    T: NativePType + PartialOrd,
    F: FnMut(T) -> bool,
{
    let (mut lo, mut hi) = (0usize, len);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let value = arr
            .execute_scalar(mid, ctx)?
            .as_primitive()
            .typed_value::<T>()
            .vortex_expect("sorted-delta pushdown requires non-null values");
        if pred(value) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    Ok(lo)
}

/// Build a boolean array that is `true` exactly on `[start, end)` (or its complement when
/// `invert` is set), with the requested nullability (always-valid when nullable, since the
/// gated array has no null values and null bounds are handled before reaching here).
pub(crate) fn bool_range(
    len: usize,
    start: usize,
    end: usize,
    invert: bool,
    nullability: Nullability,
) -> ArrayRef {
    let start = start.min(len);
    let end = end.clamp(start, len);
    let mut bits = if invert {
        BitBufferMut::new_set(len)
    } else {
        BitBufferMut::new_unset(len)
    };
    if start < end {
        bits.fill_range(start, end, !invert);
    }
    let validity = match nullability {
        Nullability::NonNullable => Validity::NonNullable,
        Nullability::Nullable => Validity::AllValid,
    };
    BoolArray::new(bits.freeze(), validity).into_array()
}
