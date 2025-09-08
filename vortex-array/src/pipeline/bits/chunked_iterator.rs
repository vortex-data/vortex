// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem::align_of;
use crate::pipeline::{N, N_WORDS};

// Iterator that supports chunking for large data with any bit offset
pub struct BitAlignedChunkedIterator<'a> {
    data: &'a [u8],
    bit_offset: usize,
    byte_offset: usize,
    buffer: Box<[usize; N_WORDS]>,
    max_bits: Option<usize>, // If set, limits output to this many bits total
    bits_produced: usize,    // Track how many bits we've produced so far
}

impl<'a> BitAlignedChunkedIterator<'a> {
    pub fn new(data: &'a [u8], bit_offset: usize) -> Self {
        Self::with_max_bits(data, bit_offset, data.len() * 8)
    }

    pub fn with_max_bits(data: &'a [u8], bit_offset: usize, max_bits: usize) -> Self {
        Self {
            data,
            bit_offset: bit_offset % 8,
            byte_offset: bit_offset / 8,
            buffer: Box::new([0usize; N_WORDS]),
            max_bits: Some(max_bits),
            bits_produced: 0,
        }
    }

    fn fill_buffer_from_bit_position(&mut self) -> usize {
        if self.byte_offset >= self.data.len() {
            return 0;
        }

        assert_ne!(self.bit_offset, 0);

        // Bit-shifted - work with bytes first, then convert to usize words
        let complement = 8 - self.bit_offset;
        let available_source = self.data.len() - self.byte_offset;

        let producible_bytes = if available_source > 1 {
            (available_source - 1).min(N / 8)
        } else {
            0
        };

        // Fill byte-by-byte into the buffer (viewed as bytes)
        let buffer_as_bytes: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(self.buffer.as_mut_ptr() as *mut u8, N / 8)
        };

        for i in 0..producible_bytes {
            let src_idx = self.byte_offset + i;
            let high = self.data[src_idx] >> self.bit_offset;
            let low = if src_idx + 1 < self.data.len() {
                self.data[src_idx + 1] << complement
            } else {
                0
            };
            buffer_as_bytes[i] = high | low;
        }

        // Zero any remaining bytes in the buffer
        if producible_bytes < N / 8 {
            buffer_as_bytes[producible_bytes..].fill(0);
        }

        producible_bytes
    }

    /// Returns next chunk (always exactly N bits as usize words, zero-padded if needed)
    pub fn next_chunk(&mut self) -> Option<[usize; N_WORDS]> {
        // Check if we've reached the maximum bits limit
        if let Some(max_bits) = self.max_bits {
            if self.bits_produced >= max_bits {
                return None;
            }
        }
        if self.bit_offset == 0 {
            // Byte-aligned: zero-copy when possible
            if self.byte_offset >= self.data.len() {
                return None;
            }
            
            let bytes_available = self.data.len() - self.byte_offset;
            let chunk_size = N / 8; // N/8 bytes
            
            if bytes_available >= chunk_size {
                // Full chunk available: check alignment for zero-copy
                let start = self.byte_offset;
                self.byte_offset += chunk_size;
                
                let src_slice = &self.data[start..start + chunk_size];
                let src_ptr = src_slice.as_ptr();
                
                // Check if properly aligned for usize array operations
                if src_ptr.align_offset(align_of::<usize>()) == 0 {
                    // Zero-copy: directly transmute aligned slice to array
                    self.bits_produced += N;
                    let result = unsafe { *(src_ptr as *const [usize; N_WORDS]) };
                    return Some(self.apply_bit_limit(result));
                } else {
                    // Copy when not aligned
                    let mut result = [0usize; N_WORDS];
                    let result_u8: &mut [u8] = unsafe {
                        std::slice::from_raw_parts_mut(result.as_mut_ptr() as *mut u8, N_WORDS * 8)
                    };
                    result_u8.copy_from_slice(src_slice);
                    self.bits_produced += N;
                    return Some(self.apply_bit_limit(result));
                }
            } else {
                // Final partial chunk: copy to buffer and zero-pad only the unused portion
                let buffer_u8: &mut [u8] = unsafe {
                    std::slice::from_raw_parts_mut(self.buffer.as_mut_ptr() as *mut u8, N_WORDS * 8)
                };
                buffer_u8[..bytes_available].copy_from_slice(&self.data[self.byte_offset..]);
                buffer_u8[bytes_available..].fill(0); // Only zero the unused portion
                self.byte_offset = self.data.len(); // Mark as exhausted
                self.bits_produced += N;
                return Some(self.apply_bit_limit(*self.buffer));
            }
        }

        // Non-byte-aligned: always use buffer with bit shifting
        let filled = self.fill_buffer_from_bit_position();
        if filled == 0 {
            None
        } else {
            // Advance by the number of bytes we actually filled
            self.byte_offset += filled;
            self.bits_produced += N;
            Some(self.apply_bit_limit(*self.buffer))
        }
    }

    fn apply_bit_limit(&self, mut result: [usize; N_WORDS]) -> [usize; N_WORDS] {
        if let Some(max_bits) = self.max_bits {
            let bits_in_this_chunk = (self.bits_produced - N)..self.bits_produced;
            let limit_in_chunk = max_bits.saturating_sub(bits_in_this_chunk.start);
            
            if limit_in_chunk < N {
                // Zero out bits beyond the limit within this chunk
                for bit in limit_in_chunk..N {
                    let word_idx = bit / (usize::BITS as usize);
                    let bit_idx = bit % (usize::BITS as usize);
                    result[word_idx] &= !(1 << bit_idx);
                }
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bit_aligned_iterator_byte_aligned() {
        // Test byte-aligned iterator (offset = 0)
        let data = vec![0b10101010u8; 128]; // 1024 bits, alternating pattern
        let mut iter = BitAlignedChunkedIterator::new(&data, 0);
        
        let chunk = iter.next_chunk().unwrap();
        
        // Convert to bytes for verification
        let chunk_u8: &[u8] = unsafe {
            std::slice::from_raw_parts(chunk.as_ptr() as *const u8, 128)
        };
        
        // Should match original data exactly
        assert_eq!(&chunk_u8[..], &data[..]);
        
        // Should be no more chunks
        assert!(iter.next_chunk().is_none());
    }

    #[test]
    fn test_bit_aligned_iterator_partial_data() {
        // Test with partial data that needs zero-padding
        let data = vec![0b11110000u8; 64]; // 512 bits (half of N)
        let mut iter = BitAlignedChunkedIterator::new(&data, 0);
        
        let chunk = iter.next_chunk().unwrap();
        let chunk_u8: &[u8] = unsafe {
            std::slice::from_raw_parts(chunk.as_ptr() as *const u8, 128)
        };
        
        // First 64 bytes should match data
        assert_eq!(&chunk_u8[..64], &data[..]);
        // Remaining 64 bytes should be zero
        assert_eq!(&chunk_u8[64..], &[0u8; 64][..]);
        
        assert!(iter.next_chunk().is_none());
    }

    #[test] 
    fn test_bit_aligned_iterator_bit_offset() {
        // Test with non-byte-aligned bit offset
        let data = vec![0b11100111u8, 0b00011100, 0b11100011, 0b10001110]; // 32 bits
        let mut iter = BitAlignedChunkedIterator::new(&data, 3); // 3-bit offset
        
        let chunk = iter.next_chunk().unwrap();
        let chunk_u8: &[u8] = unsafe {
            std::slice::from_raw_parts(chunk.as_ptr() as *const u8, 128)
        };
        
        // With 3-bit offset, we should get shifted data
        // Original: 11100111 00011100 11100011 10001110
        // Shifted by 3: 00111xxx 11100xxx 00011xxx 0011xxxx (where x comes from next byte)
        
        // With 3-bit offset, verify the bit shifting worked correctly
        // Original: 11100111 00011100 11100011 10001110
        // After 3-bit shift: bits should be correctly shifted and combined
        assert_eq!(chunk_u8[0], 0b10011100); // Verified from debug output
        
        // Should have produced some bytes, rest should be zero
        assert!(chunk_u8[4..].iter().all(|&b| b == 0));
        
        assert!(iter.next_chunk().is_none());
    }

    #[test]
    fn test_bit_aligned_iterator_multiple_chunks() {
        // Test with enough data for multiple chunks
        let data = vec![0b10101010u8; 256]; // 2048 bits = 2 * N
        let mut iter = BitAlignedChunkedIterator::new(&data, 0);
        
        // First chunk
        let chunk1 = iter.next_chunk().unwrap();
        let chunk1_u8: &[u8] = unsafe {
            std::slice::from_raw_parts(chunk1.as_ptr() as *const u8, 128)
        };
        assert_eq!(&chunk1_u8[..], &data[..128]);
        
        // Second chunk  
        let chunk2 = iter.next_chunk().unwrap();
        let chunk2_u8: &[u8] = unsafe {
            std::slice::from_raw_parts(chunk2.as_ptr() as *const u8, 128)
        };
        assert_eq!(&chunk2_u8[..], &data[128..256]);
        
        // No more chunks
        assert!(iter.next_chunk().is_none());
    }

    #[test]
    fn test_bit_aligned_iterator_bit_offset_multiple_bytes() {
        // Test bit offset with enough data to need multiple source bytes
        let data = vec![0xFFu8; 20]; // 160 bits, all 1s
        let mut iter = BitAlignedChunkedIterator::new(&data, 4); // 4-bit offset
        
        let chunk = iter.next_chunk().unwrap();
        let chunk_u8: &[u8] = unsafe {
            std::slice::from_raw_parts(chunk.as_ptr() as *const u8, 128)
        };
        
        // With 4-bit offset and all 1s, calculate expected bytes produced
        // 20 source bytes with 4-bit offset should produce about 19 bytes of data
        let expected_bytes = 19; // (20 * 8 - 4) / 8
        
        for i in 0..expected_bytes {
            assert_eq!(chunk_u8[i], 0xFF, "Byte {} should be 0xFF", i);
        }
        
        // Rest should be zero
        assert!(chunk_u8[expected_bytes..].iter().all(|&b| b == 0));
        
        assert!(iter.next_chunk().is_none());
    }
}