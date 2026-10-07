// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Walks a mask one `u64` word at a time, for kernels that process lanes in blocks of up to 64.

use vortex_buffer::BitBufferView;

/// Writes out the walk over the words of a [`BitBufferView`], binding `$word`, `$start` and
/// `$len` for each: `$word` holds the mask bits for lanes `$start..$start + $len` in its low `$len`
/// bits, and its other bits are unset.
///
/// The words come from [`BitBufferView::unaligned_chunks`], which reads the 8-byte aligned body as
/// a plain `&[u64]` with no per-word reshifting. Any misalignment is isolated in a shorter first
/// and last word, which run `$partial`. Every other word runs `$full`, where `$len` is the
/// constant 64.
///
/// The bodies are written into each walk rather than passed as closures. A closure called for the
/// first, full and last words, or nested in another closure, kept large callers from being
/// inlined: the ListView zip ran 6% slower and FoR's encoding 17% slower. A single call site
/// instead makes `$len` a runtime value for full words, which nearly doubled the time of the
/// filter's SIMD compress.
macro_rules! walk_mask_words {
    (
        $mask:expr,
        |$word:ident, $start:ident, $len:ident| full: $full:block partial: $partial:block
    ) => {{
        let mask: BitBufferView<'_> = $mask;
        let unaligned = mask.unaligned_chunks();
        let lead = unaligned.lead_padding();
        let mut next = 0;

        if let Some(prefix) = unaligned.prefix() {
            let ($word, $start, $len) = (prefix >> lead, next, (64 - lead).min(mask.len()));
            $partial
            next += $len;
        }

        for &$word in unaligned.chunks() {
            let ($start, $len): (usize, usize) = (next, 64);
            $full
            next += 64;
        }

        if let Some(suffix) = unaligned.suffix() {
            let ($word, $start, $len) = (suffix, next, mask.len() - next);
            $partial
            next += $len;
        }

        debug_assert_eq!(next, mask.len());
    }};
}

/// Invokes `f` with each `(word, start, len)` of `mask`, where `word` holds the mask bits for
/// lanes `start..start + len` in its low `len` bits and its other bits are unset.
///
/// `mask` is a [`BitBuffer`](vortex_buffer::BitBuffer) or a [`BitBufferView`], whose slices
/// cost nothing. Every word covers 64 lanes except a shorter first and last word where the mask
/// is not 8-byte aligned.
#[allow(clippy::inline_always)]
#[inline(always)]
pub fn for_each_mask_word<'a>(
    mask: impl Into<BitBufferView<'a>>,
    mut f: impl FnMut(u64, usize, usize),
) {
    walk_mask_words!(mask.into(), |word, start, len|
        full: { f(word, start, len); }
        partial: { f(word, start, len); }
    );
}

/// Like [`for_each_mask_word`], stopping at and returning the first error from `f`.
#[allow(clippy::inline_always)]
#[inline(always)]
pub fn try_for_each_mask_word<'a, E>(
    mask: impl Into<BitBufferView<'a>>,
    mut f: impl FnMut(u64, usize, usize) -> Result<(), E>,
) -> Result<(), E> {
    walk_mask_words!(mask.into(), |word, start, len|
        full: { f(word, start, len)?; }
        partial: { f(word, start, len)?; }
    );
    Ok(())
}

/// Invokes `f(index, value, valid)` for each of `values`, where `valid` is mask bit `index`.
///
/// Every value is visited, valid or not, so callers can combine `valid` with the value without
/// branching. The mask is read a word at a time, as in [`for_each_mask_word`], and each full word
/// is a fixed 64-value loop that unrolls and vectorizes. Each bit is read from its byte of the word
/// rather than by shifting the whole `u64`, which keeps the vectorized loop in 8-bit lanes.
///
/// # Panics
///
/// Panics if `values` and `mask` have different lengths.
#[allow(clippy::inline_always)]
#[inline(always)]
pub fn for_each_masked_value<'a, T: Copy>(
    values: &[T],
    mask: impl Into<BitBufferView<'a>>,
    mut f: impl FnMut(usize, T, bool),
) {
    /// Visits the values of a partial first or last word, which at most two words per mask are.
    #[inline]
    fn partial<T: Copy>(values: &[T], word: u64, start: usize, f: &mut impl FnMut(usize, T, bool)) {
        let bytes = word.to_le_bytes();
        for (j, &value) in values.iter().enumerate() {
            f(start + j, value, (bytes[j / 8] >> (j % 8)) & 1 == 1);
        }
    }

    let mask = mask.into();
    assert_eq!(
        values.len(),
        mask.len(),
        "values and mask must have the same length"
    );
    walk_mask_words!(mask, |word, start, len|
        full: {
            let bytes = word.to_le_bytes();
            let block = &values[start..start + len];
            for j in 0..64 {
                f(start + j, block[j], (bytes[j / 8] >> (j % 8)) & 1 == 1);
            }
        }
        partial: { partial(&values[start..start + len], word, start, &mut f); }
    );
}

/// A `u64` with the low `len` bits set.
#[inline]
pub fn low_bits_mask(len: usize) -> u64 {
    debug_assert!(len <= 64);
    if len == 64 {
        u64::MAX
    } else {
        (1u64 << len) - 1
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::BitBuffer;

    use super::for_each_mask_word;
    use super::for_each_masked_value;
    use super::low_bits_mask;
    use super::try_for_each_mask_word;

    #[rstest]
    fn words_cover_every_bit(
        #[values(0, 1, 63, 64, 65, 200, 1000)] len: usize,
        #[values(0, 3, 8, 61)] offset: usize,
    ) {
        let bits = BitBuffer::from_iter((0..len + offset).map(|i| i % 3 != 0 && i % 7 != 0));
        let mask = bits.slice(offset..offset + len);

        let mut next = 0;
        let mut collected = Vec::with_capacity(len);
        for_each_mask_word(&mask, |word, start, word_len| {
            assert_eq!(start, next);
            assert!((1..=64).contains(&word_len));
            assert_eq!(
                word & !low_bits_mask(word_len),
                0,
                "bits past the word are unset"
            );
            collected.extend((0..word_len).map(|i| word >> i & 1 == 1));
            next += word_len;
        });

        assert_eq!(next, len);
        assert_eq!(collected, mask.iter().collect::<Vec<_>>());
    }

    #[rstest]
    fn masked_values_match_bits(
        #[values(0, 1, 65, 1000)] len: usize,
        #[values(0, 5)] offset: usize,
    ) {
        let bits = BitBuffer::from_iter((0..len + offset).map(|i| i % 3 != 0));
        let mask = bits.slice(offset..offset + len);
        let values: Vec<usize> = (0..len).collect();

        let mut visited = Vec::with_capacity(len);
        for_each_masked_value(&values, &mask, |index, value, valid| {
            assert_eq!(index, value);
            visited.push(valid);
        });
        assert_eq!(visited, mask.iter().collect::<Vec<_>>());
    }

    #[test]
    fn try_stops_at_first_error() {
        let mask = BitBuffer::new_set(300);
        let mut starts = Vec::new();
        let result = try_for_each_mask_word(&mask, |_, start, _| {
            starts.push(start);
            if start >= 64 { Err(start) } else { Ok(()) }
        });
        let (&last, before) = starts.split_last().unwrap();
        assert_eq!(result, Err(last));
        assert!(before.iter().all(|&start| start < 64));
    }
}
