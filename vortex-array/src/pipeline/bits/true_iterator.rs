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
        let view1 = BitView::new(&chunk1);
        assert_eq!(view1.true_count(), N, "First chunk should have N true bits");

        // Second chunk should be all true bits
        let chunk2 = iter.next_chunk().unwrap();
        let view2 = BitView::new(&chunk2);
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
        let view = BitView::new(&chunk);
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
        let view = BitView::new(&chunk);
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
        let view = BitView::new(&chunk);
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
        let view = BitView::new(&chunk);
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
        let view1 = BitView::new(&chunk1);
        assert_eq!(view1.true_count(), N, "First chunk should have N true bits");

        // Second chunk should be partial
        let chunk2 = iter.next_chunk().unwrap();
        let view2 = BitView::new(&chunk2);
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
        let view = BitView::new(&chunk);
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

    #[test]
    fn test_large_input() {
        // Test with a large number of bits (5.7 * N)
        let test_bits = (N * 57) / 10; // 5.7 * N
        let mut iter = TrueSliceIterator::new(test_bits);

        let mut total_true_bits = 0;
        let mut chunk_count = 0;

        while let Some(chunk) = iter.next_chunk() {
            let view = BitView::new(&chunk);
            total_true_bits += view.true_count();
            chunk_count += 1;
        }

        assert_eq!(
            total_true_bits, test_bits,
            "Total true bits should match input"
        );
        assert_eq!(
            chunk_count, 6,
            "Should have exactly 6 chunks (5 complete + 1 partial)"
        );
    }

    #[test]
    fn test_edge_case_n_minus_one() {
        // Test with N-1 bits (1023 bits)
        let test_bits = N - 1;
        let mut iter = TrueSliceIterator::new(test_bits);

        let chunk = iter.next_chunk().unwrap();
        let view = BitView::new(&chunk);
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
    fn test_edge_case_n_plus_one() {
        // Test with N+1 bits (1025 bits)
        let test_bits = N + 1;
        let mut iter = TrueSliceIterator::new(test_bits);

        // First chunk should be complete
        let chunk1 = iter.next_chunk().unwrap();
        let view1 = BitView::new(&chunk1);
        assert_eq!(view1.true_count(), N, "First chunk should have N true bits");

        // Second chunk should have 1 bit
        let chunk2 = iter.next_chunk().unwrap();
        let view2 = BitView::new(&chunk2);
        assert_eq!(
            view2.true_count(),
            1,
            "Second chunk should have exactly 1 true bit"
        );

        // Verify the bit pattern of second chunk
        assert_eq!(
            chunk2[0], 1,
            "First word should have only the first bit set"
        );
        assert!(
            chunk2[1..].iter().all(|&w| w == 0),
            "Remaining words should be zero"
        );

        // No more chunks
        assert!(iter.next_chunk().is_none(), "Should be no more chunks");
    }

    #[test]
    fn test_next_chunk_method() {
        // Test the next_chunk method specifically (returns references)
        let mut iter = TrueSliceIterator::new(N + 100);

        // First chunk should be complete
        let chunk1_ref = iter.next_chunk().unwrap();
        let view1 = BitView::new(chunk1_ref);
        assert_eq!(view1.true_count(), N, "First chunk should have N true bits");

        // Second chunk should be partial
        let chunk2_ref = iter.next_chunk().unwrap();
        let view2 = BitView::new(chunk2_ref);
        assert_eq!(
            view2.true_count(),
            100,
            "Second chunk should have exactly 100 true bits"
        );

        // No more chunks
        assert!(iter.next_chunk().is_none(), "Should be no more chunks");
    }

    #[test]
    fn test_demonstration_usage() {
        // Demonstrate typical usage with different scenarios using next_chunk
        println!("=== TrueSliceIterator Demonstration (next_chunk) ===");

        // Case 1: Multiple complete chunks
        println!("Case 1: 3072 bits (3 complete chunks)");
        let mut iter = TrueSliceIterator::new(3 * N);
        let mut chunk_count = 0;
        while let Some(chunk) = iter.next_chunk() {
            chunk_count += 1;
            let view = BitView::new(chunk);
            println!("  Chunk {}: {} true bits", chunk_count, view.true_count());
        }

        // Case 2: Complete chunks plus partial
        println!("Case 2: 2100 bits (2 complete + 1 partial)");
        let mut iter = TrueSliceIterator::new(2 * N + 52);
        let mut chunk_count = 0;
        while let Some(chunk) = iter.next_chunk() {
            chunk_count += 1;
            let view = BitView::new(chunk);
            println!("  Chunk {}: {} true bits", chunk_count, view.true_count());
        }

        // Case 3: Single partial chunk
        println!("Case 3: 500 bits (1 partial chunk)");
        let mut iter = TrueSliceIterator::new(500);
        while let Some(chunk) = iter.next_chunk() {
            let view = BitView::new(chunk);
            println!("  Chunk: {} true bits", view.true_count());
        }
    }
}
