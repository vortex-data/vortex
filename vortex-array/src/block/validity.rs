// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The validity of a program's blocks: read from the source's mask, edited by ops a block at a
//! time, and gathered into the output's.

use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_mask::Mask;

use super::BLOCK_LEN;

/// The words of a block's validity: bit `i` is row `i`, least significant bit first.
const WORDS: usize = BLOCK_LEN / 64;

/// A block's validity as an op has edited it, or `None` while it is the source's.
pub(super) type Window = Option<[u64; WORDS]>;

/// The validity of one block's rows: the source's, as the ops before have changed it.
pub struct BlockValidity<'a> {
    source: &'a Mask,
    /// The block's first row.
    start: usize,
    pub(super) len: usize,
    window: &'a mut Window,
}

impl<'a> BlockValidity<'a> {
    pub(super) fn new(source: &'a Mask, start: usize, len: usize, window: &'a mut Window) -> Self {
        Self {
            source,
            start,
            len,
            window,
        }
    }

    /// Whether `row`, counted from the block's first row, is valid.
    #[inline]
    pub fn is_valid(&self, row: usize) -> bool {
        assert!(row < self.len, "row {row} outside a block of {}", self.len);
        match &*self.window {
            None => self.source.value(self.start + row),
            Some(words) => words[row / 64] & (1 << (row % 64)) != 0,
        }
    }

    /// Make `row`, counted from the block's first row, valid or null.
    #[inline]
    pub fn set(&mut self, row: usize, valid: bool) {
        if self.is_valid(row) != valid {
            self.words_mut()[row / 64] ^= 1 << (row % 64);
        }
    }

    /// The block's validity as words to edit, bit `i` of the result being row `i`. Bits past the
    /// block's rows are ignored.
    #[inline]
    pub fn words_mut(&mut self) -> &mut [u64] {
        let (source, start, len) = (self.source, self.start, self.len);
        let words = self
            .window
            .get_or_insert_with(|| source_words(source, start, len));
        &mut words[..len.div_ceil(64)]
    }
}

/// The validity of rows `start..start + len` of `mask` as block words.
fn source_words(mask: &Mask, start: usize, len: usize) -> [u64; WORDS] {
    let mut words = [0; WORDS];
    match mask {
        Mask::AllTrue(_) => words.fill(u64::MAX),
        Mask::AllFalse(_) => {}
        Mask::Values(values) => {
            let rows = values.bit_buffer().slice(start..start + len);
            let chunks = rows.chunks();
            let mut filled = 0;
            for (word, chunk) in words.iter_mut().zip(chunks.iter()) {
                *word = chunk;
                filled += 1;
            }
            if chunks.remainder_len() > 0 {
                words[filled] = chunks.remainder_bits();
            }
        }
    }
    words
}

/// A program's output validity: the source's mask until an op edits a block, then the mask's
/// words with each edited block written over them.
pub(super) struct OutputValidity {
    words: Option<Vec<u64>>,
}

impl OutputValidity {
    pub(super) fn new() -> Self {
        Self { words: None }
    }

    /// Record the block of `len` rows from `start`, if an op edited it.
    pub(super) fn push(&mut self, source: &Mask, start: usize, len: usize, window: &Window) {
        let Some(block) = window else {
            return;
        };
        let words = self.words.get_or_insert_with(|| mask_words(source));
        write_bits(words, start, block, len);
    }

    /// The output's validity mask.
    pub(super) fn finish(self, source: &Mask) -> Mask {
        match self.words {
            None => source.clone(),
            Some(words) => {
                let bytes = Buffer::<u64>::from(words).into_byte_buffer();
                Mask::from_buffer(BitBuffer::new(bytes, source.len()))
            }
        }
    }
}

/// Every bit of `mask` as words, row `i` at bit `i`.
fn mask_words(mask: &Mask) -> Vec<u64> {
    let len = mask.len().div_ceil(64);
    match mask {
        Mask::AllTrue(_) => vec![u64::MAX; len],
        Mask::AllFalse(_) => vec![0; len],
        Mask::Values(values) => {
            let chunks = values.bit_buffer().chunks();
            let mut words: Vec<u64> = chunks.iter().collect();
            if chunks.remainder_len() > 0 {
                words.push(chunks.remainder_bits());
            }
            words
        }
    }
}

/// Overwrite bits `at..at + len` of `dst` with the low `len` bits of `src`.
fn write_bits(dst: &mut [u64], at: usize, src: &[u64], len: usize) {
    for (i, &word) in src.iter().enumerate().take(len.div_ceil(64)) {
        let take = (len - i * 64).min(64);
        let mask = if take == 64 {
            u64::MAX
        } else {
            (1 << take) - 1
        };
        let (index, shift) = ((at + i * 64) / 64, (at + i * 64) % 64);
        dst[index] = (dst[index] & !(mask << shift)) | ((word & mask) << shift);
        if shift > 0 && shift + take > 64 {
            let spill = 64 - shift;
            dst[index + 1] = (dst[index + 1] & !(mask >> spill)) | ((word & mask) >> spill);
        }
    }
}

#[cfg(test)]
mod tests {
    use hegel::TestCase;
    use hegel::generators as gs;

    use super::write_bits;

    /// Writing a run of bits anywhere matches writing them one at a time, and leaves the other
    /// bits alone.
    #[hegel::test]
    fn write_bits_matches_bitwise(tc: TestCase) {
        let dst: Vec<u64> = tc.draw(gs::vecs(gs::integers::<u64>()).min_size(1).max_size(20));
        let src: Vec<u64> = tc.draw(gs::vecs(gs::integers::<u64>()).min_size(1).max_size(16));
        let len = tc.draw(gs::integers::<usize>().max_value((src.len() * 64).min(dst.len() * 64)));
        let at = tc.draw(gs::integers::<usize>().max_value(dst.len() * 64 - len));
        let bit = |words: &[u64], i: usize| words[i / 64] & (1 << (i % 64)) != 0;
        let mut actual = dst.clone();
        write_bits(&mut actual, at, &src, len);
        for i in 0..dst.len() * 64 {
            let expected = if (at..at + len).contains(&i) {
                bit(&src, i - at)
            } else {
                bit(&dst, i)
            };
            assert_eq!(bit(&actual, i), expected, "bit {i}");
        }
    }
}
