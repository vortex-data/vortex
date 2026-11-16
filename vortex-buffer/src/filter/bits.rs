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

/// Invoke `f` as [`for_each_mask_word`] does, but only for the words that have a set bit.
///
/// A walk that calls out for each word, such as to copy a run, keeps its loop state in memory
/// across the call, and then each empty word costs a store and a load. This walk finds the next
/// word with a set bit with [`next_set_word`], a scan without calls, so that long stretches of
/// empty words cost one compare for each block of words.
#[inline]
pub(super) fn for_each_set_mask_word(bits: MaskBits<'_>, mut f: impl FnMut(u64, usize, usize)) {
    match bits {
        MaskBits::Bitmap(bits) => {
            let unaligned = bits.unaligned_chunks();
            let lead = unaligned.lead_padding();
            let mut base = 0;

            if let Some(prefix) = unaligned.prefix() {
                let len = (64 - lead).min(bits.len());
                let word = prefix >> lead;
                if word != 0 {
                    f(word, base, len);
                }
                base += len;
            }

            let chunks = unaligned.chunks();
            for_each_set_word(chunks, |word, index| f(word, base + index * 64, 64));
            base += chunks.len() * 64;

            if let Some(suffix) = unaligned.suffix()
                && suffix != 0
            {
                f(suffix, base, bits.len() - base);
            }
        }
        MaskBits::Words { words, len } => {
            debug_assert_eq!(words.len(), len.div_ceil(64));

            for_each_set_word(words, |word, index| {
                let word_start = index * 64;
                f(word, word_start, (len - word_start).min(64));
            });
        }
    }
}

/// Invoke `f` with each `(word, index)` of `words` where `word` has a set bit.
#[inline]
fn for_each_set_word(words: &[u64], mut f: impl FnMut(u64, usize)) {
    let mut index = 0;

    while let Some(next) = next_set_word(words, index) {
        f(words[next], next);
        index = next + 1;
    }
}

/// The number of words that [`next_set_word`] checks at a time.
const SCAN_BLOCK_WORDS: usize = 8;

/// Returns the index of the first word of `words` at or after `from` that has a set bit.
///
/// Blocks of [`SCAN_BLOCK_WORDS`] words are checked with one compare, so that a long stretch of
/// empty words costs one compare for each 64 bytes and not one for each word.
#[inline]
fn next_set_word(words: &[u64], from: usize) -> Option<usize> {
    if words.get(from).is_some_and(|&word| word != 0) {
        return Some(from);
    }

    let mut index = from;

    while let Some(block) = words[index..].first_chunk::<SCAN_BLOCK_WORDS>() {
        if block.iter().fold(0, |any, &word| any | word) != 0 {
            break;
        }
        index += SCAN_BLOCK_WORDS;
    }

    words[index..]
        .iter()
        .position(|&word| word != 0)
        .map(|skip| index + skip)
}

/// Returns whether `f` holds for every word of `bits` that has a set bit. Stops at the first
/// word for which `f` does not hold.
///
/// Words do not align to element positions, so `f` must only look at properties of a word that
/// do not depend on its position, such as the number of runs of set bits. Words with no set bits
/// are skipped as in [`for_each_set_mask_word`].
pub(super) fn all_set_mask_words(bits: MaskBits<'_>, mut f: impl FnMut(u64) -> bool) -> bool {
    match bits {
        MaskBits::Bitmap(bits) => {
            let unaligned = bits.unaligned_chunks();

            unaligned.prefix().is_none_or(|word| word == 0 || f(word))
                && all_set_words(unaligned.chunks(), &mut f)
                && unaligned.suffix().is_none_or(|word| word == 0 || f(word))
        }
        MaskBits::Words { words, .. } => all_set_words(words, &mut f),
    }
}

/// Returns whether `f` holds for every word of `words` that has a set bit.
#[inline]
fn all_set_words(words: &[u64], f: &mut impl FnMut(u64) -> bool) -> bool {
    let mut index = 0;

    while let Some(next) = next_set_word(words, index) {
        if !f(words[next]) {
            return false;
        }
        index = next + 1;
    }

    true
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
