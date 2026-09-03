// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Checked-lane execution for the decimal kernels, driven by the shared
//! `vortex-compute` lane kernels.
//!
//! The primitive widths do not come through here: they are computed one row at a time by
//! [`row`](super::row), which writes a value for every row and reduces failure evidence without
//! scanning the finished output.

use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_compute::lane_kernels::IndexedSource;
use vortex_compute::lane_kernels::IndexedSourceExt;
use vortex_mask::AllOr;
use vortex_mask::Mask;

/// Apply the fallible `apply` over every lane of `source`, returning
/// `Err(first_failing_valid_lane)` only when it returns `None` on a valid lane.
///
/// `apply` also runs on invalid lanes, whose failures are masked out and whose values are
/// unspecified, so it must be total: no panics or side effects on any stored lane value.
///
/// This drives the one-pass early-exit kernels: failures abort at the end of the enclosing
/// 64-lane chunk. It suits an operation whose per-lane failure handling is cheap relative to the
/// operation itself, which is what the decimal kernels and their per-lane casts are.
///
/// Keep this wrapper inlineable so captured constants can become loop invariants in the caller.
/// The lane kernels retain their own inlining decisions.
#[inline]
pub(super) fn checked_lanes<S, T, Apply>(
    source: S,
    valid_rows: &Mask,
    apply: Apply,
) -> Result<Buffer<T>, usize>
where
    S: IndexedSource,
    T: Copy + Default,
    Apply: Fn(S::Item) -> Option<T>,
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
