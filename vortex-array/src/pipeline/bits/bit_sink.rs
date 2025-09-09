// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors


use arrow_buffer::{BooleanBuffer, BooleanBufferBuilder};
use vortex_error::{VortexResult, vortex_bail};

use crate::pipeline::{N, N_WORDS};

/// Trait for writing bits in chunks of N (1024) bits at a time
pub trait BitSink {
    /// Get a mutable slice for writing the next chunk of N bits
    /// Returns a mutable reference to N_WORDS (16) usize values
    fn next_chunk(&mut self) -> Option<&mut [usize; N_WORDS]>;

    /// Commit exactly n bits from the current chunk (where n <= N)
    /// This finalizes the current chunk and prepares for the next one
    fn commit_n(&mut self, n: usize) -> VortexResult<()>;

    /// Finish writing and return the final BooleanBuffer
    fn finish(self) -> VortexResult<Option<BooleanBuffer>>;
}

#[derive(Default)]
pub struct EmptyBitSink;

impl BitSink for EmptyBitSink {
    #[inline]
    fn next_chunk(&mut self) -> Option<&mut [usize; N_WORDS]> {
        None
    }

    #[inline]
    fn commit_n(&mut self, n: usize) -> VortexResult<()> {
        Ok(())
    }

    #[inline]
    fn finish(self) -> VortexResult<Option<BooleanBuffer>> {
        Ok(None)
    }
}

/// Aligned bit sink that pre-allocates all memory upfront
/// Efficient for cases where the total number of bits is known ahead of time
/// Requires committing exactly N bits at a time
// TODO(joe): relax the `N` requirement to allow committing less than N bits, as long as the number divides 64.
pub struct AlignedBitSink {
    /// Pre-allocated buffer to hold all bits
    buffer: Vec<u64>,
    /// Current position in the buffer (in N_WORDS chunks)
    chunk_index: usize,
    /// Total number of bits expected
    total_bits: usize,
}

impl AlignedBitSink {
    /// Create a new aligned bit sink with known total bit capacity
    pub fn new(total_bits: usize) -> Self {
        let total_words = total_bits.div_ceil(u64::BITS as usize);
        // Ensure we have at least N_WORDS capacity for safety
        let buffer_words = total_words.max(N / u64::BITS as usize);

        Self {
            buffer: vec![0u64; buffer_words],
            chunk_index: 0,
            total_bits,
        }
    }
}

impl BitSink for AlignedBitSink {
    #[inline]
    fn next_chunk(&mut self) -> Option<&mut [usize; N_WORDS]> {
        const CHUNK_SIZE: usize = N / u64::BITS as usize;
        let start = self.chunk_index * CHUNK_SIZE;
        let end = start + CHUNK_SIZE;

        // Ensure we don't go out of bounds
        if end > self.buffer.len() {
            None
        } else {
            // Return direct mutable reference to the buffer slice
            Some(unsafe {
                &mut *(self.buffer[start..end].as_mut_ptr() as *mut [usize; CHUNK_SIZE])
            })
        }
    }

    #[inline]
    fn commit_n(&mut self, n: usize) -> VortexResult<()> {
        // AlignedBitSink requires committing exactly N bits
        if n != N {
            if self.chunk_index == 0 && n == self.total_bits {
                self.chunk_index += 1;
                return Ok(());
            }
            vortex_bail!(
                "AlignedBitSink requires committing exactly {} bits, got {}",
                N,
                n
            );
        }

        self.chunk_index += 1;
        Ok(())
    }

    #[inline]
    fn finish(self) -> VortexResult<Option<BooleanBuffer>> {
        Ok(Some(BooleanBuffer::new(
            arrow_buffer::Buffer::from_vec(self.buffer),
            0,
            self.total_bits,
        )))
    }
}

/// Unaligned bit sink that uses a fixed-size working buffer
/// Efficient for streaming writes where total size may not be known
pub struct UnalignedBitSink {
    /// Builder for constructing the final BooleanBuffer
    builder: BooleanBufferBuilder,
    /// Working buffer for current chunk (always N bits)
    working_buffer: [usize; N_WORDS],
}

impl UnalignedBitSink {
    /// Create a new unaligned bit sink with optional capacity hint
    pub fn new(capacity_hint: usize) -> Self {
        let builder = BooleanBufferBuilder::new(capacity_hint);

        Self {
            builder,
            working_buffer: [0; N_WORDS],
        }
    }
}

impl BitSink for UnalignedBitSink {
    #[inline]
    fn next_chunk(&mut self) -> Option<&mut [usize; N_WORDS]> {
        // Clear the working buffer and return it
        self.working_buffer.fill(0);
        Some(&mut self.working_buffer)
    }

    #[inline]
    fn commit_n(&mut self, n: usize) -> VortexResult<()> {
        if n > N {
            vortex_bail!("Cannot commit more than {} bits per chunk, got {}", N, n);
        }

        // Extract individual bits from the working buffer and append to builder
        let bits_per_word = usize::BITS as usize;

        let buf = unsafe { &*(self.working_buffer.as_ptr() as *const [u8; N / 8]) };
        self.builder.append_packed_range(0..n, buf);

        Ok(())
    }

    #[inline]
    fn finish(mut self) -> VortexResult<Option<BooleanBuffer>> {
        Ok(Some(self.builder.finish()))
    }
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexExpect;

    use super::*;

    #[test]
    fn test_aligned_bit_sink_exact_chunks() {
        let total_bits = N * 2; // Exactly 2 chunks
        let mut sink = AlignedBitSink::new(total_bits);

        // First chunk - set all bits
        {
            let chunk = sink.next_chunk().vortex_expect("");
            chunk.fill(usize::MAX);
            sink.commit_n(N).unwrap();
        }

        // Second chunk - set all bits
        {
            let chunk = sink.next_chunk().vortex_expect("");
            chunk.fill(usize::MAX);
            sink.commit_n(N).unwrap();
        }

        let result = sink.finish().unwrap().unwrap();
        assert_eq!(result.len(), total_bits);
        assert_eq!(result.count_set_bits(), total_bits);
    }

    #[test]
    fn test_aligned_bit_sink_requires_exact_n() {
        let total_bits = N * 2;
        let mut sink = AlignedBitSink::new(total_bits);

        // First chunk - should work with exactly N bits
        {
            let chunk = sink.next_chunk().vortex_expect("");
            chunk.fill(usize::MAX);
            assert!(sink.commit_n(N).is_ok());
        }

        // Try to commit with non-N value - should fail
        {
            let chunk = sink.next_chunk().vortex_expect("");
            chunk.fill(usize::MAX);
            assert!(sink.commit_n(100).is_err(), "Should fail when n != N");
            assert!(sink.commit_n(N - 1).is_err(), "Should fail when n != N");
            assert!(sink.commit_n(N + 1).is_err(), "Should fail when n != N");
            // Finally commit with correct N
            assert!(sink.commit_n(N).is_ok());
        }

        let result = sink.finish().unwrap().unwrap();
        assert_eq!(result.len(), total_bits);
        assert_eq!(result.count_set_bits(), total_bits);
    }

    #[test]
    fn test_unaligned_bit_sink_streaming() {
        let mut sink = UnalignedBitSink::new(N * 2);

        // First chunk - alternating pattern
        {
            let chunk = sink.next_chunk().vortex_expect("");
            for word in chunk.iter_mut() {
                *word = 0x5555555555555555; // Alternating 01010101 pattern
            }
            sink.commit_n(N).unwrap();
        }

        // Second partial chunk - first 500 bits all set
        {
            let chunk = sink.next_chunk().vortex_expect("");
            let full_words = 500 / (usize::BITS as usize);
            for i in 0..full_words {
                chunk[i] = usize::MAX;
            }
            let remaining_bits = 500 % (usize::BITS as usize);
            if remaining_bits > 0 && full_words < N_WORDS {
                chunk[full_words] = (1usize << remaining_bits) - 1;
            }
            sink.commit_n(500).unwrap();
        }

        let result = sink.finish().unwrap().unwrap();
        assert_eq!(result.len(), N + 500);

        // Check the alternating pattern in the first N bits
        let expected_true_in_first_chunk = N / 2; // Half the bits should be true
        let first_chunk_trues = (0..N).filter(|&i| result.value(i)).count();
        assert_eq!(first_chunk_trues, expected_true_in_first_chunk);
    }

    #[test]
    fn test_unaligned_bit_sink_small_commits() {
        let mut sink = UnalignedBitSink::new(N);

        // Commit small chunks of different sizes
        for chunk_size in [1, 10, 50, 100, 200] {
            let chunk = sink.next_chunk().vortex_expect("");
            // Set all bits in this chunk
            chunk.fill(usize::MAX);
            sink.commit_n(chunk_size).unwrap();
        }

        let result = sink.finish().unwrap().unwrap();
        let expected_total = 1 + 10 + 50 + 100 + 200;
        assert_eq!(result.len(), expected_total);
        assert_eq!(result.count_set_bits(), expected_total);
    }

    #[test]
    fn test_both_sinks_consistency() {
        // Test that both sinks produce the same result for the same input
        // Use exactly 2*N bits since AlignedBitSink requires N-bit commits
        let total_bits = N * 2;

        // Set up aligned sink
        let mut aligned_sink = AlignedBitSink::new(total_bits);

        // Set up unaligned sink
        let mut unaligned_sink = UnalignedBitSink::new(total_bits);

        // Write the same pattern to both sinks
        let pattern = [0x3333333333333333usize; N_WORDS]; // 00110011 repeating pattern

        // First complete chunk
        {
            let aligned_chunk = aligned_sink.next_chunk().vortex_expect("");
            aligned_chunk.copy_from_slice(&pattern);
            aligned_sink.commit_n(N).unwrap();

            let unaligned_chunk = unaligned_sink.next_chunk().vortex_expect("");
            unaligned_chunk.copy_from_slice(&pattern);
            unaligned_sink.commit_n(N).unwrap();
        }

        // Second complete chunk
        {
            let aligned_chunk = aligned_sink.next_chunk().vortex_expect("");
            aligned_chunk.copy_from_slice(&pattern);
            aligned_sink.commit_n(N).unwrap();

            let unaligned_chunk = unaligned_sink.next_chunk().vortex_expect("");
            unaligned_chunk.copy_from_slice(&pattern);
            unaligned_sink.commit_n(N).unwrap();
        }

        let aligned_result = aligned_sink.finish().unwrap().unwrap();
        let unaligned_result = unaligned_sink.finish().unwrap().unwrap();

        // Both should have the same length and bit count
        assert_eq!(aligned_result.len(), unaligned_result.len());
        assert_eq!(
            aligned_result.count_set_bits(),
            unaligned_result.count_set_bits()
        );

        // Compare bit by bit
        for i in 0..total_bits {
            assert_eq!(
                aligned_result.value(i),
                unaligned_result.value(i),
                "Bit {} differs between aligned and unaligned sinks",
                i
            );
        }
    }

    #[test]
    fn test_error_cases() {
        let mut sink = AlignedBitSink::new(100);

        // Try to commit more than N bits
        let chunk = sink.next_chunk().vortex_expect("");
        chunk.fill(usize::MAX);
        assert!(sink.commit_n(N + 1).is_err());

        // Try to commit less than N bits (AlignedBitSink requires exactly N)
        let mut sink2 = AlignedBitSink::new(N * 2);
        let chunk2 = sink2.next_chunk().vortex_expect("");
        chunk2.fill(usize::MAX);
        assert!(sink2.commit_n(100).is_err()); // Not exactly N
    }
}
