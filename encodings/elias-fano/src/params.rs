// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The bit layout's geometry, as pure arithmetic over `(span, n)`. The encoder writes the results
//! into the metadata and `validate_parts` re-derives them, refusing an array that disagrees.
//!
//! Each element splits into an `l`-bit low part, bit-packed in a child slot, and a high part
//! `element >> l` set as one bit at position `high + index + 1`. The `+ index` keeps positions
//! distinct when elements share a high part, so reading element `i` is a `select1` and the inverse
//! is `high = position - index - 1`. The `+ 1` sentinel aligns the unset bits with the high parts,
//! giving `rank1(select0(h)) == select0(h) - h`, so one `select0` counts the elements below a high
//! part with no rank directory stored.

use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

/// One zero-sample is stored per `1 << LOG_SAMPLING0` unset bits of the upper array.
///
/// The upper array is roughly 50% dense, so 512 zeros span about 512 bits — one or two 64-byte
/// chunks, the window [`BitBuffer::select_zero_range`](vortex_buffer::BitBuffer::select_zero_range)
/// is fastest over. [`LOG_SAMPLING1`] is sized the same way.
pub(crate) const LOG_SAMPLING0: usize = 9;

/// One one-sample is stored per `1 << LOG_SAMPLING1` set bits of the upper array.
pub(crate) const LOG_SAMPLING1: usize = 8;

/// How far ahead `next_geq` walks from where the cursor sits before reseating through the
/// zero-sample table.
///
/// Bounded in high-part buckets, which the cursor can measure up front, and in elements, which is
/// what it pays: a bucket holding a run of duplicates is arbitrarily many elements deep, so a gap
/// of a few buckets is not a gap of a few steps.
pub(crate) const LINEAR_SCAN_THRESHOLD: u64 = 8;

/// The widest low part we will store.
///
/// `l == 64` would leave no high part and would ask the bit-packed child for its own full width,
/// which FastLanes does not do. Only a single element spanning the whole `u64` range reaches it.
pub(crate) const MAX_LOWER_WIDTH: u8 = 63;

/// The number of low bits to give each element, written `l` in the literature.
///
/// `l = floor(log2(universe / n))` balances the halves: low parts cost `l` bits each and the upper
/// array costs about `n + universe / 2^l` bits, so the total lands near `n * (l + 2)`.
pub(crate) fn lower_width(span: u64, n: usize) -> u8 {
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
pub(crate) fn upper_len(span: u64, n: usize, lower_width: u8) -> VortexResult<u64> {
    let buckets = span >> lower_width;
    let upper_len = (n as u64)
        .checked_add(buckets)
        .and_then(|v| v.checked_add(2))
        .ok_or_else(|| {
            vortex_error::vortex_err!(
                "Elias-Fano upper array overflows: n {n}, span {span}, lower_width {lower_width}"
            )
        })?;
    vortex_ensure!(
        usize::try_from(upper_len).is_ok(),
        "Elias-Fano upper array of {upper_len} bits does not fit in memory"
    );
    Ok(upper_len)
}

/// The number of unset bits in an upper array of `upper_len` bits holding `n` elements.
///
/// No query path calls this; it states the identity [`num_samples0`]'s derivation has to agree
/// with, and `validate_parts` asserts the two against each other.
#[inline]
pub(crate) fn num_zeros(upper_len: u64, n: usize) -> u64 {
    upper_len - n as u64
}

/// The number of zero-samples the layout calls for.
///
/// The unset bits are the sentinel, one terminator per bucket, and the guard zero, so the universe
/// alone fixes this count whatever `n` is. That is what lets a reader split the shared samples
/// buffer into its two tables without the seam being written into the metadata.
#[inline]
pub(crate) fn num_samples0(span: u64, lower_width: u8) -> u64 {
    // Saturating because `lower_width` arrives from metadata: a corrupt zero against a full-width
    // span would otherwise overflow here rather than at the buffer-length check that catches it.
    ((span >> lower_width).saturating_add(1)) >> LOG_SAMPLING0
}

/// The size of the low-bits slot when `lower_width == 0`.
///
/// That slot is `ConstantArray::new(0u64, n)`, whose single buffer is the scalar encoded as
/// protobuf: one tag byte and one varint. Pinned by
/// `tests::test_encoded_bit_size_matches_encoder`, which fails if that encoding ever changes.
const CONSTANT_LOWER_BYTES: u64 = 2;

/// The exact size in bits of the buffers an Elias-Fano encoding of `n` values spanning `span`
/// occupies, without building one.
///
/// This is what the compressor prices a candidate column at. It is exact rather than asymptotic:
/// the upper array rounded up to whole bytes, both sample tables, and the FastLanes-padded
/// low-bits child. It equals `8 * array.nbytes()` for the array [`crate::elias_fano_encode`]
/// produces from the same `(span, n)`, which `tests::test_encoded_bit_size_matches_encoder` pins.
///
/// The literature's `n * (log2(u / n) + 2)` is the asymptotic form of the same quantity. This
/// includes the constant overheads a real column of a few thousand values actually pays, which is
/// what a cost model has to compare against bit-packing.
pub fn encoded_bit_size(span: u64, n: usize) -> VortexResult<u64> {
    if n == 0 {
        // The degenerate array `compress::empty` builds: a two-bit upper array, and nothing else.
        return Ok(8);
    }

    let lower_width = lower_width(span, n);
    let upper_len = upper_len(span, n, lower_width)?;

    // The upper array is byte-padded on disk.
    let upper_bytes = upper_len.div_ceil(8);

    // One sample per `1 << LOG_SAMPLING0` unset bits, and one per `1 << LOG_SAMPLING1` set bits
    // above the first: `UpperBuilder::push` never samples rank 0, so the one-samples are counted
    // over ranks `1..n`.
    let samples = num_samples0(span, lower_width) + ((n as u64 - 1) >> LOG_SAMPLING1);
    let sample_bytes = samples * size_of::<u64>() as u64;

    // A zero-width low part is a `ConstantArray` of `0u64`, whose only buffer is the scalar as
    // protobuf. Otherwise FastLanes packs whole 1024-element blocks, padding the tail; see
    // `BitPackedArray`'s buffer length.
    let lower_bytes = if lower_width == 0 {
        CONSTANT_LOWER_BYTES
    } else {
        (n.div_ceil(1024) as u64) * 128 * u64::from(lower_width)
    };

    Ok((upper_bytes + sample_bytes + lower_bytes) * 8)
}

#[inline]
pub(crate) fn lower_mask(lower_width: u8) -> u64 {
    if lower_width == 0 {
        0
    } else {
        u64::MAX >> (64 - u32::from(lower_width))
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    // Four elements over a universe of 18: 2 low bits, 4 buckets.
    #[case(17, 4, 2, 10)]
    // A dense run 0..n: universe == n, so no low bits, and the upper array is 2n + 1.
    #[case(999, 1000, 0, 2001)]
    // Sparse: 1000 elements over a 2^20 universe wants 10 low bits, leaving 1024 buckets.
    #[case((1 << 20) - 1, 1000, 10, 1000 + 1023 + 2)]
    // All-equal input. Span 0 means one bucket and no low bits.
    #[case(0, 100, 0, 102)]
    // A single element at the very top of the u64 range: the clamp fires.
    #[case(u64::MAX, 1, 63, 4)]
    fn test_geometry(
        #[case] span: u64,
        #[case] n: usize,
        #[case] expected_width: u8,
        #[case] expected_upper_len: u64,
    ) -> VortexResult<()> {
        let width = lower_width(span, n);
        assert_eq!(width, expected_width, "lower_width");
        assert_eq!(upper_len(span, n, width)?, expected_upper_len, "upper_len");
        Ok(())
    }

    /// The upper array must always be long enough to hold the position that the highest element
    /// claims, and to leave at least one zero rank above the highest one a query can name.
    #[rstest]
    #[case(0, 1)]
    #[case(1, 1)]
    #[case(u64::MAX, 1)]
    #[case(u64::MAX, 1024)]
    #[case(1_000_000, 100_000)]
    #[case(7, 8)]
    #[case(255, 256)]
    fn test_upper_len_leaves_room(#[case] span: u64, #[case] n: usize) -> VortexResult<()> {
        let width = lower_width(span, n);
        let upper_len = upper_len(span, n, width)?;

        // The last element sits at `(span >> width) + (n - 1) + 1`, which must be in bounds.
        let last_position = (span >> width) + n as u64;
        assert!(last_position < upper_len, "last position {last_position}");

        // A reseat may name any zero rank up to the maximum element's high part.
        let max_zero_rank = span >> width;
        assert!(max_zero_rank < num_zeros(upper_len, n), "max zero rank");
        Ok(())
    }

    #[test]
    fn test_lower_mask() {
        assert_eq!(lower_mask(0), 0);
        assert_eq!(lower_mask(1), 1);
        assert_eq!(lower_mask(8), 0xFF);
        assert_eq!(lower_mask(63), u64::MAX >> 1);
    }
}
