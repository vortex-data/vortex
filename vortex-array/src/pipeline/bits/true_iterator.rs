// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::pipeline::bits::MaskSliceIterator;
use crate::pipeline::{N, N_WORDS};

pub struct TrueSliceIterator {
    len: usize,
    remaining_bits: usize,
    chunk: Box<[usize; N_WORDS]>,
}

impl MaskSliceIterator for TrueSliceIterator {
    fn next_chunk(&mut self) -> Option<&[usize; N_WORDS]> {
        self._next_chunk()
    }

    fn len(&self) -> usize {
        self.len
    }

    fn true_count(&self) -> usize {
        self.len // All bits are true in TrueSliceIterator
    }
}

impl TrueSliceIterator {
    pub fn new(len_bits: usize) -> Self {
        Self {
            len: len_bits,
            remaining_bits: len_bits,
            chunk: Box::new([usize::MAX; N_WORDS]),
        }
    }

    /// Returns next chunk (always exactly N bits as usize words, zero-padded if needed)
    /// Returns a reference to avoid array copy
    fn _next_chunk(&mut self) -> Option<&[usize; N_WORDS]> {
        if self.remaining_bits == 0 {
            return None;
        }

        let chunk_bits = self.remaining_bits.min(N);
        self.remaining_bits -= chunk_bits;

        // If this is a full chunk (N bits), return all true bits
        if chunk_bits == N {
            return Some(&self.chunk);
        }

        // Handle partial chunk - need to mask off unused bits
        let usize_bits = usize::BITS as usize;

        // Calculate how many complete usize words we need
        let complete_words = chunk_bits / usize_bits;

        // Handle the partial word if there is one
        let remaining_bits_in_partial_word = chunk_bits % usize_bits;
        if remaining_bits_in_partial_word > 0 {
            let mask = (1usize << remaining_bits_in_partial_word) - 1;
            self.chunk[complete_words] = mask;
            // Zero out any remaining words
            self.chunk[complete_words + 1..].fill(0);
        } else {
            // Zero out any remaining words after complete words
            self.chunk[complete_words..].fill(0);
        }

        Some(&self.chunk)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::bits::BitView;

    #[test]
    fn test_exact_multiple_of_n() {
        // Test with exactly 2 * N bits (2048 bits = 2 * 1024)
        let mut iter = TrueSliceIterator::new(2 * N);

        // First chunk should be all true bits
        let chunk1 = iter.next_chunk().unwrap();
        let view1 = BitView::new(chunk1);
        assert_eq!(view1.true_count(), N, "First chunk should have N true bits");

        // Second chunk should be all true bits
        let chunk2 = iter.next_chunk().unwrap();
        let view2 = BitView::new(chunk2);
        assert_eq!(
            view2.true_count(),
            N,
            "Second chunk should have N true bits"
        );

        // No more chunks
        assert!(iter.next_chunk().is_none(), "Should be no more chunks");
    }

    #[test]
    fn test_single_complete_chunk() {
        // Test with exactly N bits (1024 bits)
        let mut iter = TrueSliceIterator::new(N);

        let chunk = iter.next_chunk().unwrap();
        let view = BitView::new(chunk);
        assert_eq!(view.true_count(), N, "Chunk should have N true bits");

        // No more chunks
        assert!(iter.next_chunk().is_none(), "Should be no more chunks");
    }

    #[test]
    fn test_partial_chunk_small() {
        // Test with 100 bits (less than N)
        let test_bits = 100;
        let mut iter = TrueSliceIterator::new(test_bits);

        let chunk = iter.next_chunk().unwrap();
        let view = BitView::new(chunk);
        assert_eq!(
            view.true_count(),
            test_bits,
            "Chunk should have exactly {} true bits",
            test_bits
        );

        // No more chunks
        assert!(iter.next_chunk().is_none(), "Should be no more chunks");
    }

    #[test]
    fn test_partial_chunk_word_boundary() {
        // Test with exactly one usize word (64 bits on 64-bit systems)
        let usize_bits = usize::BITS as usize;
        let mut iter = TrueSliceIterator::new(usize_bits);

        let chunk = iter.next_chunk().unwrap();
        let view = BitView::new(chunk);
        assert_eq!(
            view.true_count(),
            usize_bits,
            "Chunk should have exactly {} true bits",
            usize_bits
        );

        // Verify that only the first word is filled
        assert_eq!(chunk[0], usize::MAX, "First word should be all ones");
        assert!(
            chunk[1..].iter().all(|&w| w == 0),
            "Remaining words should be zero"
        );

        // No more chunks
        assert!(iter.next_chunk().is_none(), "Should be no more chunks");
    }

    #[test]
    fn test_partial_chunk_multiple_words() {
        // Test with 2.5 usize words (e.g., 160 bits on 64-bit systems)
        let usize_bits = usize::BITS as usize;
        let test_bits = usize_bits * 2 + usize_bits / 2; // 2.5 words
        let mut iter = TrueSliceIterator::new(test_bits);

        let chunk = iter.next_chunk().unwrap();
        let view = BitView::new(chunk);
        assert_eq!(
            view.true_count(),
            test_bits,
            "Chunk should have exactly {} true bits",
            test_bits
        );

        // Verify the bit pattern
        assert_eq!(chunk[0], usize::MAX, "First word should be all ones");
        assert_eq!(chunk[1], usize::MAX, "Second word should be all ones");

        // Third word should be half filled (32 bits on 64-bit systems)
        let expected_partial = (1usize << (usize_bits / 2)) - 1;
        assert_eq!(
            chunk[2], expected_partial,
            "Third word should be half filled"
        );

        // Remaining words should be zero
        assert!(
            chunk[3..].iter().all(|&w| w == 0),
            "Remaining words should be zero"
        );

        // No more chunks
        assert!(iter.next_chunk().is_none(), "Should be no more chunks");
    }

    #[test]
    fn test_one_plus_partial_chunk() {
        // Test with N + 100 bits (1124 bits = 1024 + 100)
        let test_bits = N + 100;
        let mut iter = TrueSliceIterator::new(test_bits);

        // First chunk should be complete
        let chunk1 = iter.next_chunk().unwrap();
        let view1 = BitView::new(chunk1);
        assert_eq!(view1.true_count(), N, "First chunk should have N true bits");

        // Second chunk should be partial
        let chunk2 = iter.next_chunk().unwrap();
        let view2 = BitView::new(chunk2);
        assert_eq!(
            view2.true_count(),
            100,
            "Second chunk should have exactly 100 true bits"
        );

        // No more chunks
        assert!(iter.next_chunk().is_none(), "Should be no more chunks");
    }

    #[test]
    fn test_single_bit() {
        // Test with just 1 bit
        let mut iter = TrueSliceIterator::new(1);

        let chunk = iter.next_chunk().unwrap();
        let view = BitView::new(chunk);
        assert_eq!(view.true_count(), 1, "Chunk should have exactly 1 true bit");

        // Verify the bit pattern
        assert_eq!(chunk[0], 1, "First word should have only the first bit set");
        assert!(
            chunk[1..].iter().all(|&w| w == 0),
            "Remaining words should be zero"
        );

        // No more chunks
        assert!(iter.next_chunk().is_none(), "Should be no more chunks");
    }

    #[test]
    fn test_zero_bits() {
        // Test with 0 bits
        let mut iter = TrueSliceIterator::new(0);

        // Should immediately return None
        assert!(
            iter.next_chunk().is_none(),
            "Should have no chunks for 0 bits"
        );
    }
}
