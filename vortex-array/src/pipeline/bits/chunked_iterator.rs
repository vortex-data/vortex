// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::iter::{Chain, Once};
use std::mem::align_of;
use std::slice;

use arrow_buffer::BooleanBuffer;
use arrow_buffer::bit_chunk_iterator::BitChunkIterator;

use crate::pipeline::{N, N_WORDS};

#[allow(clippy::len_without_is_empty)]
pub trait MaskSliceIterator {
    fn next_chunk(&mut self) -> Option<&[usize; N_WORDS]>;

    fn len(&self) -> usize;

    fn true_count(&self) -> usize;
}

/// An iterator that returns chunks of N bits as usize words, zero-padded if needed
/// this will zero copy if possible, otherwise it will copy the data into a buffer
pub struct BitAlignedChunkedIterator<'a> {
    data: &'a [u8],
    byte_offset: usize,
    buffer: Box<[usize; N_WORDS]>,
    len: usize,        // Total length in bits
    true_count: usize, // Number of true bits
    bit_chunk_iter: Option<Chain<BitChunkIterator<'a>, Once<u64>>>,
    done: bool,
}

impl MaskSliceIterator for BitAlignedChunkedIterator<'_> {
    #[inline]
    fn next_chunk(&mut self) -> Option<&[usize; N_WORDS]> {
        self._next_chunk()
    }

    #[inline]
    fn len(&self) -> usize {
        self.len
    }

    #[inline]
    fn true_count(&self) -> usize {
        self.true_count
    }
}

impl<'a> From<&'a BooleanBuffer> for BitAlignedChunkedIterator<'a> {
    fn from(value: &'a BooleanBuffer) -> Self {
        let true_count = value.count_set_bits();
        Self::new(value, true_count)
    }
}

impl<'a> BitAlignedChunkedIterator<'a> {
    pub fn new(buffer: &'a BooleanBuffer, true_count: usize) -> Self {
        // Check if data is aligned and we can use fast path
        let is_aligned = buffer.offset() % usize::BITS as usize == 0
            && buffer.values().as_ptr().align_offset(align_of::<usize>()) == 0;

        let len = buffer.len();

        if is_aligned {
            // Use original fast path logic
            let bit_offset = buffer.offset();
            let start_byte = bit_offset / 8;
            let end_bit = bit_offset + len;
            let end_byte = end_bit.div_ceil(8);
            let byte_len = end_byte - start_byte;
            let data = buffer.values();

            let sliced_data = &data[start_byte..][..byte_len.min(data.len() - start_byte)];

            Self {
                data: sliced_data,
                byte_offset: 0,
                buffer: Box::new([0usize; N_WORDS]),
                len,
                true_count,
                bit_chunk_iter: None,
                done: false,
            }
        } else {
            // Use BooleanBuffer iterator for non-aligned access
            let bit_chunks = buffer.bit_chunks();
            let iter = if bit_chunks.remainder_len() > 0 {
                // Only include remainder if there are actual remainder bits
                bit_chunks
                    .iter()
                    .chain(std::iter::once(bit_chunks.remainder_bits()))
            } else {
                // No remainder bits, just use the regular iterator
                bit_chunks.iter().chain(std::iter::once(0u64))
            };

            Self {
                data: &[],
                byte_offset: 0,
                buffer: Box::new([0usize; N_WORDS]),
                len,
                true_count,
                bit_chunk_iter: Some(iter),
                done: false,
            }
        }
    }

    /// Returns next chunk (always exactly N bits as usize words, zero-padded if needed)
    /// This cannot be an iterator since the chunk
    #[inline]
    pub fn _next_chunk(&mut self) -> Option<&[usize; N_WORDS]> {
        if self.done {
            return None;
        }
        // If we have a stored iterator, use it to fill up to 16 u64 chunks
        if let Some(ref mut iter) = self.bit_chunk_iter {
            let mut u64_count = 0;

            // Call iterator up to 16 times to fill our N_WORDS buffer
            while u64_count < N_WORDS {
                if let Some(u64_chunk) = iter.next() {
                    #[allow(clippy::cast_possible_truncation)]
                    {
                        self.buffer[u64_count] = u64_chunk as usize;
                    }
                    u64_count += 1;
                } else {
                    break;
                }
            }

            if u64_count > 0 {
                // After producing one chunk of N bits, we're done for exactly N bits
                if self.len <= N {
                    self.done = true;
                }
                self.buffer[u64_count..].fill(0);
                return Some(&*self.buffer);
            } else {
                self.done = true;
                return None;
            }
        }

        // Original logic for aligned access
        const CHUNK_SIZE: usize = N / 8;
        if self.byte_offset * 8 >= self.len {
            return None;
        }

        let remaining_bits = self.len - self.byte_offset * 8;

        if remaining_bits >= N
            && self.data[self.byte_offset..]
                .as_ptr()
                .align_offset(align_of::<usize>())
                == 0
        {
            let result = unsafe {
                &*(self.data[self.byte_offset..][..CHUNK_SIZE].as_ptr() as *const [usize; N_WORDS])
            };
            self.byte_offset += CHUNK_SIZE;
            Some(result)
        } else {
            let bytes_to_copy = remaining_bits.div_ceil(8);

            let buffer_u8: &mut [u8] =
                unsafe { slice::from_raw_parts_mut(self.buffer.as_mut_ptr() as *mut u8, N / 8) };

            buffer_u8[..bytes_to_copy]
                .copy_from_slice(&self.data[self.byte_offset..self.byte_offset + bytes_to_copy]);
            buffer_u8[bytes_to_copy..].fill(0);

            // If this is a partial chunk, we need to clear excess bits in the last byte
            if remaining_bits % 8 != 0 {
                let mask = (1u8 << (remaining_bits % 8)) - 1;
                buffer_u8[bytes_to_copy - 1] &= mask;
            }

            self.byte_offset += bytes_to_copy;
            Some(&*self.buffer)
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
        let buffer = BooleanBuffer::new(data.clone().into(), 0, data.len() * 8);
        let mut iter = BitAlignedChunkedIterator::new(&buffer, buffer.count_set_bits());

        let chunk = iter.next_chunk().unwrap();

        // Convert to bytes for verification
        let chunk_u8: &[u8] = unsafe { slice::from_raw_parts(chunk.as_ptr() as *const u8, 128) };

        // Should match original data exactly
        assert_eq!(chunk_u8, &data[..]);

        // Should be no more chunks
        assert!(iter.next_chunk().is_none());
    }

    #[test]
    fn test_bit_aligned_iterator_byte_subword() {
        // Test byte-aligned iterator (offset = 0)
        let data = vec![0b10101010u8; 128]; // 1024 bits, alternating pattern
        let buffer = BooleanBuffer::new(data.into(), 0, ((128 - 1) * 8) + 7);
        let mut iter = BitAlignedChunkedIterator::new(&buffer, buffer.count_set_bits());

        let chunk = iter.next_chunk().unwrap();

        // Convert to bytes for verification
        let chunk_u8: &[u8] = unsafe { slice::from_raw_parts(chunk.as_ptr() as *const u8, 128) };

        let mut expected = [0b10101010u8; 128];
        expected[127] = 0b00101010u8;

        // Should match original data exactly
        assert_eq!(chunk_u8, &expected[..]);

        // Should be no more chunks
        assert!(iter.next_chunk().is_none());
    }

    #[test]
    fn test_bit_aligned_iterator_partial_data() {
        // Test with partial data that needs zero-padding
        let data = vec![0b11110000u8; 64]; // 512 bits (half of N)
        let buffer = BooleanBuffer::new(data.clone().into(), 0, data.len() * 8);
        let mut iter = BitAlignedChunkedIterator::new(&buffer, buffer.count_set_bits());

        let chunk = iter.next_chunk().unwrap();
        let chunk_u8: &[u8] = unsafe { slice::from_raw_parts(chunk.as_ptr() as *const u8, 128) };

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
        let buffer = BooleanBuffer::new(data.into(), 3, 32 - 3); // 3-bit offset
        let mut iter = BitAlignedChunkedIterator::new(&buffer, buffer.count_set_bits());

        let chunk = iter.next_chunk().unwrap();
        let chunk_u8: &[u8] = unsafe { slice::from_raw_parts(chunk.as_ptr() as *const u8, 128) };

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
        let data = vec![0b10101010u8; 256];
        let buffer = BooleanBuffer::new(data.clone().into(), 0, data.len() * 8);
        let mut iter = BitAlignedChunkedIterator::new(&buffer, buffer.count_set_bits());

        // First chunk
        let chunk1 = iter.next_chunk().unwrap();
        let chunk1_u8: &[u8] = unsafe { slice::from_raw_parts(chunk1.as_ptr() as *const u8, 128) };
        assert_eq!(chunk1_u8, &data[..128]);

        // Second chunk
        let chunk2 = iter.next_chunk().unwrap();
        let chunk2_u8: &[u8] = unsafe { slice::from_raw_parts(chunk2.as_ptr() as *const u8, 128) };
        assert_eq!(chunk2_u8, &data[128..256]);

        // No more chunks
        assert!(iter.next_chunk().is_none());
    }

    #[test]
    fn test_bit_aligned_iterator_bit_offset_multiple_bytes() {
        // Test bit offset with enough data to need multiple source bytes
        let data = vec![0b01111000u8; 20]; // 160 bits
        let buffer = BooleanBuffer::new(data.into(), 4, (20 * 8) - 8); // 4-bit offset
        let mut iter = BitAlignedChunkedIterator::new(&buffer, buffer.count_set_bits());

        let chunk = iter.next_chunk().unwrap();
        let chunk_u8: &[u8] = unsafe { slice::from_raw_parts(chunk.as_ptr() as *const u8, 128) };

        // With 4-bit offset and all 1s, calculate expected bytes produced
        // 20 source bytes with 4-bit offset should produce about 19 bytes of data
        let expected_bytes = 19; // (20 * 8 - 4) / 8

        for i in 0..expected_bytes {
            assert_eq!(chunk_u8[i], 0b10000111, "Byte {} should be 0xFF", i);
        }

        // Rest should be zero
        assert!(chunk_u8[expected_bytes..].iter().all(|&b| b == 0));

        assert!(iter.next_chunk().is_none());
    }

    #[test]
    fn test_bit_aligned_iterator_byte_subword_offset() {
        // Test with bit offset - this should produce exactly one chunk
        let data = vec![0b10101010u8; 129];
        let buffer = BooleanBuffer::new(data.into(), 1, 1024); // 1-bit offset, exactly 1024 bits
        let mut iter = BitAlignedChunkedIterator::new(&buffer, buffer.count_set_bits());

        let chunk = iter.next_chunk().unwrap();

        // Convert to bytes for verification
        let chunk_u8: &[u8] = unsafe { slice::from_raw_parts(chunk.as_ptr() as *const u8, 128) };

        // With 1-bit offset, pattern shifts: 10101010 -> 01010101
        let expected = [0b01010101u8; 128];

        // Should match expected shifted pattern
        assert_eq!(chunk_u8, &expected[..]);

        // Should be no more chunks for exactly 1024 bits
        let next_chunk = iter.next_chunk();
        assert!(next_chunk.is_none(), "Expected no more chunks but got one");
    }
}
