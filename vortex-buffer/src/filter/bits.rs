// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Walks over the 64-bit words of a selection bitmap.

use crate::BitBuffer;

/// The selection bits that a word walker visits.
#[derive(Clone, Copy)]
pub(super) enum MaskBits<'a> {
    /// All the bits of a bitmap.
    Bitmap(&'a BitBuffer),
    /// The first `len` bits of `words`, with any bits past `len` cleared.
    Words { words: &'a [u64], len: usize },
}

/// Invoke `f` with each `(word, word_start, word_len)` of `bits`, where `word` holds the bits
/// for elements `word_start..word_start + word_len` in its low `word_len` bits.
#[inline]
pub(super) fn for_each_mask_word(bits: MaskBits<'_>, f: impl FnMut(u64, usize, usize)) {
    match bits {
        MaskBits::Bitmap(bits) => for_each_bitmap_word(bits, f),
        MaskBits::Words { words, len } => for_each_slice_word(words, len, f),
    }
}

/// Returns whether `f` holds for every word of `bits`. Stops at the first word for which `f`
/// does not hold.
///
/// Words do not align to element positions, so `f` must only look at properties of a word that
/// do not depend on its position, such as the number of runs of set bits.
pub(super) fn all_mask_words(bits: MaskBits<'_>, mut f: impl FnMut(u64) -> bool) -> bool {
    match bits {
        MaskBits::Bitmap(bits) => bits.unaligned_chunks().iter().all(f),
        MaskBits::Words { words, .. } => words.iter().all(|&word| f(word)),
    }
}

#[inline]
fn for_each_slice_word(words: &[u64], len: usize, mut f: impl FnMut(u64, usize, usize)) {
    debug_assert_eq!(words.len(), len.div_ceil(64));

    for (word_idx, &word) in words.iter().enumerate() {
        let word_start = word_idx * 64;
        f(word, word_start, (len - word_start).min(64));
    }
}

#[inline]
fn for_each_bitmap_word(bits: &BitBuffer, mut f: impl FnMut(u64, usize, usize)) {
    let unaligned = bits.unaligned_chunks();
    let lead = unaligned.lead_padding();
    let mut base = 0;

    if let Some(prefix) = unaligned.prefix() {
        let len = (64 - lead).min(bits.len());
        f(prefix >> lead, base, len);
        base += len;
    }

    for &word in unaligned.chunks() {
        f(word, base, 64);
        base += 64;
    }

    if let Some(suffix) = unaligned.suffix() {
        let len = bits.len() - base;
        f(suffix, base, len);
        base += len;
    }

    debug_assert_eq!(base, bits.len());
}

/// A `u64` with the low `len` bits set.
#[inline]
pub(super) fn low_bits_mask(len: usize) -> u64 {
    debug_assert!(len <= 64);
    if len == 64 {
        u64::MAX
    } else {
        (1u64 << len) - 1
    }
}
