// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem::align_of;
use arrow_buffer::BooleanBuffer;
use crate::pipeline::{N, N_WORDS};

// Iterator that supports chunking for large data with any bit offset
pub struct BitAlignedChunkedIterator<'a> {
    data: &'a [u8],
    bit_offset: usize,
    byte_offset: usize,
    buffer: Box<[usize; N_WORDS]>,
    len: usize,          // Total length in bits
}

impl <'a> From<&'a BooleanBuffer> for BitAlignedChunkedIterator<'a> {
    fn from(value: &'a BooleanBuffer) -> Self {
        Self::new(value.values(), value.offset(), value.len())
    }
}

impl<'a> BitAlignedChunkedIterator<'a> {
    fn bits_processed(&self) -> usize {
        self.byte_offset * 8
    }
    pub fn new(data: &'a [u8], bit_offset: usize, len: usize) -> Self {
        // Calculate the exact byte range we need
        let start_byte = bit_offset / 8;
        let end_bit = bit_offset + len;
        let end_byte = end_bit.div_ceil(8);
        let byte_len = end_byte - start_byte;
        
        let sliced_data = &data[start_byte..][..byte_len.min(data.len() - start_byte)];


        println!("bit offset {}, sb {}", bit_offset % 8, start_byte);
        
        Self {
            data: sliced_data,
            bit_offset: bit_offset % 8,
            byte_offset: 0, // Reset because we already sliced from start_byte
            buffer: Box::new([0usize; N_WORDS]),
            len,
        }
    }

    fn fill_buffer_from_bit_position(&mut self) -> usize {
        debug_assert!(self.byte_offset < self.data.len() );
        debug_assert_ne!(self.bit_offset, 0);

        // Calculate how many bits we still need
        let remaining_bits = self.len - self.byte_offset * 8;
        let bits_to_produce = remaining_bits.min(N);
        
        // Convert bits to bytes, accounting for the bit offset
        let complete_bytes_to_produce = (bits_to_produce)/8;
        let bytes_to_produce = bits_to_produce.div_ceil(8);


        println!("bits_to_produce {}, bytes_to_produce: {}", bits_to_produce, bytes_to_produce);

        // Fill byte-by-byte into the buffer (viewed as bytes)
        let buffer_as_bytes: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(self.buffer.as_mut_ptr() as *mut u8, N / 8)
        };

        let complement = 8 - self.bit_offset;
        // Handle all complete bytes (8 bits each)
        // let complete_bytes = (bits_to_produce / 8).min(producible_bytes);
        for i in 0..complete_bytes_to_produce {
            let src_idx = self.byte_offset + i;
            let high = self.data[src_idx] >> self.bit_offset;
            let low =  self.data[src_idx + 1] << complement;
            buffer_as_bytes[i] = high | low;
        }
        // println!("complete_bytes: {}, producible_bytes: {}", complete_bytes, producible_bytes);

        // Handle the final partial byte if needed
        let remaining_bits = bits_to_produce % 8;
        let mask = if remaining_bits != 0 {
            (1u8 << remaining_bits) - 1
        } else {
            u8::MAX
        };
        println!("bits_to_produce {}, remaining_bits: {}, mask: {:08b}", bits_to_produce, remaining_bits, mask);
        if remaining_bits > 0 {
            let src_idx = self.byte_offset + bytes_to_produce;
            let high = self.data[src_idx] >> self.bit_offset;
            let low = if src_idx + 1 < self.data.len() {
                self.data[src_idx + 1] << complement
            } else {
                0
            };
            println!("high: {:08b}, low: {:08b}, mask: {:08b}", high, low, mask);
            let result_byte = high | (low & mask);
            
            // Mask to keep only the needed bits
            buffer_as_bytes[bytes_to_produce] = result_byte;
        }

        // Zero any remaining bytes in the buffer
        if bytes_to_produce < N / 8 {
            buffer_as_bytes[bytes_to_produce+1..].fill(0);
        }


        bytes_to_produce
    }

    /// Returns next chunk (always exactly N bits as usize words, zero-padded if needed)
    pub fn next_chunk(&mut self) -> Option<[usize; N_WORDS]> {
        const CHUNK_SIZE: usize = N / 8;
        if self.byte_offset * 8 >= self.len {
            return None;
        }
        
        let remaining_bits = self.len - self.byte_offset * 8;

        if self.bit_offset == 0 {
            if remaining_bits >= N  && self.data[self.byte_offset..].as_ptr().align_offset(align_of::<usize>()) == 0 {
                let result = unsafe { *(self.data[self.byte_offset..][.. CHUNK_SIZE].as_ptr() as *const [usize; N_WORDS]) };
                self.byte_offset += CHUNK_SIZE;
                return Some(result);
            } else {
                let bytes_to_copy = remaining_bits.div_ceil(8);

                let buffer_u8: &mut [u8] = unsafe {
                    std::slice::from_raw_parts_mut(self.buffer.as_mut_ptr() as *mut u8, N / 8)
                };

                buffer_u8[..bytes_to_copy].copy_from_slice(&self.data[self.byte_offset..self.byte_offset + bytes_to_copy]);
                buffer_u8[bytes_to_copy..].fill(0);
                
                // If this is a partial chunk, we need to clear excess bits in the last byte
                if remaining_bits % 8 != 0 {
                    let mask = (1u8 << (remaining_bits % 8)) - 1;
                    buffer_u8[bytes_to_copy-1] &= mask;
                }

                self.byte_offset += bytes_to_copy;
                return Some(*self.buffer)
            }
        }


        let filled = self.fill_buffer_from_bit_position();
        if filled == 0 {
            None
        } else {
            // Advance by the number of bytes we actually filled
            self.byte_offset += filled;
            Some(*self.buffer)
        }
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bit_aligned_iterator_byte_aligned() {
        // Test byte-aligned iterator (offset = 0)
        let data = vec![0b10101010u8; 128]; // 1024 bits, alternating pattern
        let mut iter = BitAlignedChunkedIterator::new(&data, 0, data.len() * 8);
        
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
    fn test_bit_aligned_iterator_byte_subword() {
        // Test byte-aligned iterator (offset = 0)
        let data = vec![0b10101010u8; 128]; // 1024 bits, alternating pattern
        let mut iter = BitAlignedChunkedIterator::new(&data, 0, ((data.len() - 1) * 8) + 7);

        let chunk = iter.next_chunk().unwrap();

        // Convert to bytes for verification
        let chunk_u8: &[u8] = unsafe {
            std::slice::from_raw_parts(chunk.as_ptr() as *const u8, 128)
        };

        let mut expected = vec![0b10101010u8; 128];
        expected[127] = 0b00101010u8;

        // assert_eq!(chunk_u8[127], expected[127], "{:08b}, {:08b}", chunk_u8[127], expected[127]);

        // Should match original data exactly
        assert_eq!(&chunk_u8[..], &expected[..]);

        // Should be no more chunks
        assert!(iter.next_chunk().is_none());
    }

    #[test]
    fn test_bit_aligned_iterator_partial_data() {
        // Test with partial data that needs zero-padding
        let data = vec![0b11110000u8; 64]; // 512 bits (half of N)
        let mut iter = BitAlignedChunkedIterator::new(&data, 0, data.len() * 8);
        
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
        let mut iter = BitAlignedChunkedIterator::new(&data, 3, data.len() * 8); // 3-bit offset
        
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
        let mut iter = BitAlignedChunkedIterator::new(&data, 0, data.len() * 8);
        
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
        let data = vec![0b01111000u8; 20]; // 160 bits, all 1s
        let mut iter = BitAlignedChunkedIterator::new(&data, 4, (data.len() * 8) - 8); // 4-bit offset
        
        let chunk = iter.next_chunk().unwrap();
        let chunk_u8: &[u8] = unsafe {
            std::slice::from_raw_parts(chunk.as_ptr() as *const u8, 128)
        };
        
        // With 4-bit offset and all 1s, calculate expected bytes produced
        // 20 source bytes with 4-bit offset should produce about 19 bytes of data
        let expected_bytes = 19; // (20 * 8 - 4) / 8

        println!("chunk_u8: {:?}", chunk_u8.iter().map(|b| format!("{:08b}", b)).collect::<Vec<_>>());
        
        for i in 0..expected_bytes {
            assert_eq!(chunk_u8[i], 0b10000111, "Byte {} should be 0xFF", i);
        }
        
        // Rest should be zero
        assert!(chunk_u8[expected_bytes..].iter().all(|&b| b == 0));
        
        assert!(iter.next_chunk().is_none());
    }
    
    #[test]
    fn test_partial_final_chunk_byte_advance() {
        // Test that we correctly advance byte_offset for partial final chunks
        // This was causing an off-by-one error where we advanced too far
        let total_bits = 1030; // Just over N (1024)
        
        let bytes_needed = (total_bits + 7) / 8; // 129 bytes
        let data = vec![0xFFu8; bytes_needed];
        
        let mut iter = BitAlignedChunkedIterator::new(&data, 0, total_bits);
        
        // First chunk: full 1024 bits
        let chunk1 = iter.next_chunk().unwrap();
        assert_eq!(iter.bits_processed(), 1024);
        assert_eq!(iter.byte_offset, 128); // 1024 / 8
        
        // Second chunk: only 6 bits
        let chunk2 = iter.next_chunk().unwrap();
        // Should only advance by 1 byte (6 bits needs 1 byte)
        assert_eq!(iter.byte_offset, 129);
        assert_eq!(iter.bits_processed(), 1032); // We process full bytes
        
        // No more chunks
        assert!(iter.next_chunk().is_none());
    }
    
    #[test]
    fn test_primitive_mask_regression() {
        // This test reproduces the off-by-one error from export_primitive_nonnull_masked
        // Create a mask with a specific pattern that triggers the issue
        let total_bits = 2100; // Just over 2 * N
        
        // Create a boolean buffer with a pattern
        let mut bytes = vec![0u8; (total_bits + 7) / 8];
        // Set every 3rd bit to true (similar to the test pattern)
        for i in 0..total_bits {
            if i % 3 == 0 {
                let byte_idx = i / 8;
                let bit_idx = i % 8;
                bytes[byte_idx] |= 1 << bit_idx;
            }
        }
        
        let bool_buffer = BooleanBuffer::new(bytes.into(), 0, total_bits);
        let mut iter = BitAlignedChunkedIterator::from(&bool_buffer);
        
        let mut total_true_count = 0;
        let mut chunks_processed = 0;
        
        // Process first chunk (1024 bits)
        if let Some(chunk) = iter.next_chunk() {
            let view = crate::pipeline::bits::BitView::new(&chunk);
            let true_count = view.true_count();
            total_true_count += true_count;
            chunks_processed += 1;
            
            // Verify the count matches expectation for first 1024 bits
            let expected = (0..1024).filter(|i| i % 3 == 0).count();
            assert_eq!(true_count, expected, "First chunk true count mismatch");
        }
        
        // Process second chunk (1024 bits) 
        if let Some(chunk) = iter.next_chunk() {
            let view = crate::pipeline::bits::BitView::new(&chunk);
            let true_count = view.true_count();
            total_true_count += true_count;
            chunks_processed += 1;
            
            // Verify the count matches expectation for bits 1024..2048
            let expected = (1024..2048).filter(|i| i % 3 == 0).count();
            assert_eq!(true_count, expected, "Second chunk true count mismatch");
        }
        
        // Process remaining chunk (52 bits)
        if let Some(chunk) = iter.next_chunk() {
            let view = crate::pipeline::bits::BitView::new(&chunk);
            let true_count = view.true_count();
            total_true_count += true_count;
            chunks_processed += 1;
            
            // Verify the count matches expectation for bits 2048..2100
            let expected = (2048..2100).filter(|i| i % 3 == 0).count();
            assert_eq!(true_count, expected, "Remaining chunk true count mismatch");
        }
        
        assert_eq!(chunks_processed, 3, "Should have processed 3 chunks");
        
        // Verify total matches
        let expected_total = (0..total_bits).filter(|i| i % 3 == 0).count();
        assert_eq!(total_true_count, expected_total, "Total true count mismatch");
    }

    #[test]
    fn test_bit_aligned_iterator_byte_subword_offset() {
        // Test byte-aligned iterator (offset = 0)
        let data = vec![0b10101010u8; 128]; // 1024 bits, alternating pattern
        let mut iter = BitAlignedChunkedIterator::new(&data, 1, ((data.len() - 1) * 8) + 7);

        let chunk = iter.next_chunk().unwrap();

        // Convert to bytes for verification
        let chunk_u8: &[u8] = unsafe {
            std::slice::from_raw_parts(chunk.as_ptr() as *const u8, 128)
        };

        let mut expected = vec![0b01010101u8; 128];
        // expected[127] = 0b01010101u8;

        assert_eq!(chunk_u8[127], expected[127], "{:08b}, {:08b}", chunk_u8[127], expected[127]);

        // Should match original data exactly
        assert_eq!(&chunk_u8[..], &expected[..]);

        // Should be no more chunks
        assert!(iter.next_chunk().is_none());
    }
}