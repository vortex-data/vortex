// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Walks a mask one `u64` word at a time, for kernels that process lanes in blocks of up to 64.

use vortex_buffer::BitBufferView;

/// The words of a mask, as `(word, start, len)`: `word` holds the mask bits for lanes
/// `start..start + len` in its low `len` bits, and its other bits are unset.
///
/// The words come from [`BitBufferView::unaligned_chunks`], which reads the 8-byte aligned body
/// as a plain `&[u64]` with no per-word reshifting. Any misalignment is isolated in a shorter
/// first and last word, so every other word covers 64 lanes.
pub struct MaskWords<'a> {
    prefix: Option<(u64, usize)>,
    body: std::slice::Iter<'a, u64>,
    suffix: Option<(u64, usize)>,
    start: usize,
}

impl<'a> MaskWords<'a> {
    /// The words of `mask`, a [`BitBuffer`](vortex_buffer::BitBuffer) or a [`BitBufferView`],
    /// whose slices cost nothing.
    #[inline]
    pub fn new(mask: impl Into<BitBufferView<'a>>) -> Self {
        let mask = mask.into();
        let unaligned = mask.unaligned_chunks();
        let lead = unaligned.lead_padding();
        let prefix_len = unaligned
            .prefix()
            .map_or(0, |_| (64 - lead).min(mask.len()));
        let suffix_len = mask.len() - prefix_len - 64 * unaligned.chunks().len();
        Self {
            prefix: unaligned
                .prefix()
                .map(|prefix| (prefix >> lead, prefix_len)),
            body: unaligned.chunks().iter(),
            suffix: unaligned.suffix().map(|suffix| (suffix, suffix_len)),
            start: 0,
        }
    }
}

impl Iterator for MaskWords<'_> {
    type Item = (u64, usize, usize);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let (word, len) = match self.prefix.take() {
            Some(prefix) => prefix,
            None => match self.body.next() {
                Some(&word) => (word, 64),
                None => self.suffix.take()?,
            },
        };
        let start = self.start;
        self.start += len;
        Some((word, start, len))
    }
}

/// Invokes `f` with each `(word, start, len)` of [`MaskWords`].
///
/// `f` is called from a single place, so it inlines like the body of a plain loop.
#[allow(clippy::inline_always)]
#[inline(always)]
pub fn for_each_mask_word<'a>(
    mask: impl Into<BitBufferView<'a>>,
    mut f: impl FnMut(u64, usize, usize),
) {
    for (word, start, len) in MaskWords::new(mask) {
        f(word, start, len);
    }
}

/// Like [`for_each_mask_word`], stopping at and returning the first error from `f`.
#[allow(clippy::inline_always)]
#[inline(always)]
pub fn try_for_each_mask_word<'a, E>(
    mask: impl Into<BitBufferView<'a>>,
    mut f: impl FnMut(u64, usize, usize) -> Result<(), E>,
) -> Result<(), E> {
    for (word, start, len) in MaskWords::new(mask) {
        f(word, start, len)?;
    }
    Ok(())
}

/// Invokes `f(index, value, valid)` for each of `values`, where `valid` is mask bit `index`.
///
/// Every value is visited, valid or not, so callers can combine `valid` with the value without
/// branching. Each full word of [`MaskWords`] is a fixed 64-value loop that unrolls and
/// vectorizes. Each bit is read from its byte of the word rather than by shifting the whole
/// `u64`, which keeps the vectorized loop in 8-bit lanes.
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
    let mask = mask.into();
    assert_eq!(
        values.len(),
        mask.len(),
        "values and mask must have the same length"
    );
    for_each_mask_word(mask, |word, start, len| {
        let bytes = word.to_le_bytes();
        let values = &values[start..start + len];
        if let Ok(block) = <&[T; 64]>::try_from(values) {
            for j in 0..64 {
                f(start + j, block[j], (bytes[j / 8] >> (j % 8)) & 1 == 1);
            }
        } else {
            for (j, &value) in values.iter().enumerate() {
                f(start + j, value, (bytes[j / 8] >> (j % 8)) & 1 == 1);
            }
        }
    });
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
