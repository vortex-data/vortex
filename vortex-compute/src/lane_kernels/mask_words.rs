// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Walks a mask one `u64` word at a time, for kernels that process lanes in blocks of up to 64.

use vortex_buffer::BitBuffer;

/// Invokes `f` with each `(word, start, len)` of `mask`, where `word` holds the mask bits for
/// lanes `start..start + len` in its low `len` bits and its other bits are unset.
///
/// The words come from [`BitBuffer::unaligned_chunks`], which reads the 8-byte aligned body as a
/// plain `&[u64]` with no per-word reshifting. Any misalignment is isolated in a shorter first
/// and last word, so every other word covers 64 lanes.
// This does not delegate to `try_for_each_mask_word`: wrapping `f` in a second closure kept large
// callers from being inlined, and the ListView zip ran 6% slower.
#[allow(clippy::inline_always)]
#[inline(always)]
pub fn for_each_mask_word(mask: &BitBuffer, mut f: impl FnMut(u64, usize, usize)) {
    let unaligned = mask.unaligned_chunks();
    let lead = unaligned.lead_padding();
    let mut start = 0;

    if let Some(prefix) = unaligned.prefix() {
        let len = (64 - lead).min(mask.len());
        f(prefix >> lead, start, len);
        start += len;
    }

    for &word in unaligned.chunks() {
        f(word, start, 64);
        start += 64;
    }

    if let Some(suffix) = unaligned.suffix() {
        let len = mask.len() - start;
        f(suffix, start, len);
        start += len;
    }

    debug_assert_eq!(start, mask.len());
}

/// Like [`for_each_mask_word`], stopping at and returning the first error from `f`.
#[allow(clippy::inline_always)]
#[inline(always)]
pub fn try_for_each_mask_word<E>(
    mask: &BitBuffer,
    mut f: impl FnMut(u64, usize, usize) -> Result<(), E>,
) -> Result<(), E> {
    let unaligned = mask.unaligned_chunks();
    let lead = unaligned.lead_padding();
    let mut start = 0;

    if let Some(prefix) = unaligned.prefix() {
        let len = (64 - lead).min(mask.len());
        f(prefix >> lead, start, len)?;
        start += len;
    }

    for &word in unaligned.chunks() {
        f(word, start, 64)?;
        start += 64;
    }

    if let Some(suffix) = unaligned.suffix() {
        let len = mask.len() - start;
        f(suffix, start, len)?;
        start += len;
    }

    debug_assert_eq!(start, mask.len());
    Ok(())
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
