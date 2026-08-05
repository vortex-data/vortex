// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Checked-lane execution for numeric kernels, driven by the shared
//! `vortex-compute` lane kernels.

use std::ops::BitOrAssign;

use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_compute::lane_kernels::IndexedSource;
use vortex_compute::lane_kernels::IndexedSourceExt;
use vortex_mask::AllOr;
use vortex_mask::Mask;

/// Evidence that a lane failed, anything other than [`Default`] meaning failure.
///
/// `bool` is the ordinary choice; [`map_checked_into`] explains why the wider members exist and
/// asserts the width bound that membership here does **not** imply.
///
/// [`map_checked_into`]: IndexedSourceExt::map_checked_into
pub(super) trait Failure: Copy + Default + PartialEq + BitOrAssign {}

impl Failure for bool {}
impl Failure for u8 {}
impl Failure for u16 {}
impl Failure for u32 {}
impl Failure for u64 {}

/// Apply the fallible `apply` over every lane of `source`, returning
/// `Err(first_failing_valid_lane)` only when it returns `None` on a _valid_ lane.
///
/// `apply` also runs on invalid lanes, whose failures are masked out and whose values are
/// unspecified, so it must be total: no panics or side effects on any stored lane value.
///
/// This drives the one-pass early-exit kernels, which abort at the end of the enclosing 64-lane
/// chunk. Use it when per-lane failure handling is cheap relative to the operation, as in integer
/// division. Prefer [`checked_apply_lanes`] when the failure check itself vectorizes.
///
/// `#[inline]`: this must inline into the caller that builds the closure, so that a captured
/// constant operand flattens into a register rather than living behind a pointer the loop reloads
/// on every lane, which blocks vectorization.
#[inline]
pub(super) fn checked_lanes<S, T, Apply>(
    source: S,
    valid_rows: &Mask,
    apply: Apply,
) -> Result<Buffer<T>, usize>
where
    S: IndexedSource,
    T: Copy + Default,
    Apply: FnMut(S::Item) -> Option<T>,
{
    let len = source.len();
    debug_assert_eq!(len, valid_rows.len());

    let valid_bits = match valid_rows.bit_buffer() {
        AllOr::All => None,
        AllOr::None => return Ok(Buffer::zeroed(len)),
        AllOr::Some(valid_bits) => Some(valid_bits),
    };

    let mut values = BufferMut::<T>::with_capacity(len);

    let out = &mut values.spare_capacity_mut()[..len];
    match valid_bits {
        None => source.try_map_into(out, apply)?,
        Some(valid_bits) => source.try_map_masked_into(valid_bits, out, apply)?,
    }

    // SAFETY: the kernels initialize every lane in `out`.
    unsafe { values.set_len(len) };

    Ok(values.freeze())
}

/// Apply the split value/failure `apply` over every lane of `source`, returning
/// `Err(first_failing_valid_lane)` only when it flags a _valid_ lane.
///
/// The hot pass writes every value unconditionally and OR-reduces one piece of evidence, leaving
/// the loop free of per-lane selects and per-chunk exit branches. Only if a lane flagged does a
/// cold second pass re-run `apply` through the early-exit kernels, which drop null-lane failures
/// and attribute the first valid one.
///
/// Like [`checked_lanes`], `apply` runs on invalid lanes and must be total.
///
/// `#[inline]`: see [`checked_lanes`].
#[inline]
pub(super) fn checked_apply_lanes<S, T, Fail, Apply>(
    source: S,
    valid_rows: &Mask,
    mut apply: Apply,
) -> Result<Buffer<T>, usize>
where
    S: IndexedSource + Copy,
    T: Copy + Default,
    Fail: Failure,
    Apply: FnMut(S::Item) -> (T, Fail),
{
    let len = source.len();
    debug_assert_eq!(len, valid_rows.len());

    let valid_bits = match valid_rows.bit_buffer() {
        AllOr::All => None,
        AllOr::None => return Ok(Buffer::zeroed(len)),
        AllOr::Some(valid_bits) => Some(valid_bits),
    };

    let mut values = BufferMut::<T>::with_capacity(len);

    let out = &mut values.spare_capacity_mut()[..len];
    if source.map_checked_into(out, &mut apply) != Fail::default() {
        let mut checked = |item: S::Item| {
            let (value, failure) = apply(item);
            (failure == Fail::default()).then_some(value)
        };
        match valid_bits {
            None => source.try_map_into(out, &mut checked)?,
            Some(valid_bits) => source.try_map_masked_into(valid_bits, out, &mut checked)?,
        }
    }

    // SAFETY: the kernels initialize every lane in `out`.
    unsafe { values.set_len(len) };

    Ok(values.freeze())
}
