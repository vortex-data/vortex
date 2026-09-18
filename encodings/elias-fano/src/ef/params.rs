// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The layout's geometry, as pure arithmetic over `(span, n)`. See the [module docs](super) for the
//! layout these size.
//!
//! An encoder stores the results and [`validate_layout`](super::validate_layout) re-derives them,
//! refusing a layout that disagrees.

use super::Error;

/// One zero-sample is stored per `1 << LOG_SAMPLING0` unset bits of the upper array.
///
/// The upper array is roughly 50% dense, so 512 zeros span about 512 bits — eight words, a window
/// short enough for [`select_zero_range`](super::select_zero_range) to walk without vectorising.
/// [`LOG_SAMPLING1`] is sized the same way.
pub const LOG_SAMPLING0: usize = 9;

/// One one-sample is stored per `1 << LOG_SAMPLING1` set bits of the upper array.
pub const LOG_SAMPLING1: usize = 8;

/// The widest low part we will store.
///
/// `l == 64` would leave no high part at all. Only a single element spanning the whole `u64` range
/// reaches the clamp.
pub const MAX_LOWER_WIDTH: u8 = 63;

/// The number of low bits to give each element, written `l` in the literature.
///
/// `l = floor(log2(universe / n))` balances the halves: low parts cost `l` bits each and the upper
/// array costs about `n + universe / 2^l` bits, so the total lands near `n * (l + 2)`.
pub fn lower_width(span: u64, n: usize) -> u8 {
    debug_assert!(n > 0, "lower_width is undefined for an empty sequence");

    // The universe is `span + 1` values, which is 2^64 when the span fills a u64 — hence u128.
    let universe = u128::from(span) + 1;
    let n = u128::from(n as u64);
    if universe <= n {
        // More elements than distinct values: the sequence is dense, or has many duplicates.
        // Every bit is better spent on the upper array, which stays O(n) either way.
        return 0;
    }
    let width = (universe / n).ilog2();
    u8::try_from(width).unwrap_or(u8::MAX).min(MAX_LOWER_WIDTH)
}

/// The length in bits of the upper array, written `H` in the literature.
///
/// One set bit per element, one unset bit per high-part bucket boundary, and `+ 2` for the sentinel
/// and a trailing guard zero, so the largest selectable zero rank `span >> lower_width` is always
/// present. Bounded at roughly `3n`, since `lower_width` keeps `(span + 1) >> lower_width < 2n`.
pub fn upper_len(span: u64, n: usize, lower_width: u8) -> Result<u64, Error> {
    let buckets = span >> lower_width;
    let upper_len = (n as u64)
        .checked_add(buckets)
        .and_then(|v| v.checked_add(2))
        .ok_or(Error::UpperLenOverflow {
            n,
            span,
            lower_width,
        })?;
    if usize::try_from(upper_len).is_err() {
        return Err(Error::UpperLenTooLarge { upper_len });
    }
    Ok(upper_len)
}

/// The number of unset bits in an upper array of `upper_len` bits holding `n` elements.
///
/// No read path calls this. It states the identity [`num_samples0`] must agree with, which
/// [`validate_layout`](super::validate_layout) asserts.
#[inline]
pub fn num_zeros(upper_len: u64, n: usize) -> u64 {
    upper_len - n as u64
}

/// The number of zero-samples the layout calls for.
///
/// The unset bits are the sentinel, one terminator per bucket, and the guard zero, so the universe
/// alone fixes this count whatever `n` is. A reader therefore splits the shared samples buffer into
/// its two tables without the seam being stored.
#[inline]
pub fn num_samples0(span: u64, lower_width: u8) -> u64 {
    // Saturating because `lower_width` arrives from metadata: a corrupt zero against a full-width
    // span would otherwise overflow here rather than at the buffer-length check that catches it.
    ((span >> lower_width).saturating_add(1)) >> LOG_SAMPLING0
}

/// The number of one-samples the layout calls for.
///
/// Rank 0 is never sampled — the first set bit is where a reader starts anyway — so the samples are
/// counted over ranks `1..n`.
#[inline]
pub fn num_samples1(n: usize) -> u64 {
    (n as u64).saturating_sub(1) >> LOG_SAMPLING1
}

/// The mask keeping the low `lower_width` bits of an element.
#[inline]
pub fn lower_mask(lower_width: u8) -> u64 {
    if lower_width == 0 {
        0
    } else {
        u64::MAX >> (64 - u32::from(lower_width))
    }
}
