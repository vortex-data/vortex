// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::iter;

use vortex_error::VortexExpect;

use crate::ByteBuffer;
use crate::trusted_len::TrustedLen;

/// Aligned bitwise view over underlying bytes in u64 chunks
pub struct BitChunks {
    buffer: ByteBuffer,
    len: usize,
    bit_offset: usize,
    remainder_len: usize,
}

impl BitChunks {
    /// Construct new with given length and offset
    pub fn new(buffer: ByteBuffer, offset: usize, len: usize) -> Self {
        let byte_len = (offset + len).div_ceil(8);
        assert!(
            byte_len <= buffer.len(),
            "Buffer {} too small for given length {len} and offset {offset}",
            buffer.len()
        );

        let bit_offset = offset % 8;
        let byte_offset = offset / 8;
        let remainder_len = len % 64;

        Self {
            buffer: buffer.slice(byte_offset..byte_len),
            len,
            bit_offset,
            remainder_len,
        }
    }

    /// Length of the last non full slice of bits
    pub fn remainder_len(&self) -> usize {
        self.remainder_len
    }

    /// Last u64 chunk of the underlying buffer
    pub fn remainder_bits(&self) -> u64 {
        if self.remainder_len == 0 {
            return 0;
        }

        // Since we sliced the buffer on construction then remainder is aligned with the buffer
        // NOTE: you want the rounding behaviour of integer division i.e., it's not correct to simplify this to self.len / 8
        let remainder_bytes = &self.buffer[self.len / 64 * 8..];
        let mut result_bits = remainder_bytes[0] as u64 >> self.bit_offset;
        for (i, &byte) in remainder_bytes[1..].iter().enumerate() {
            result_bits |= (byte as u64) << ((i + 1) * 8 - self.bit_offset);
        }

        result_bits & ((1 << self.remainder_len) - 1)
    }

    /// Get an interator over the bitwise chunks including the trailer
    pub fn iter(&self) -> PaddedBitChunksIterator {
        BitChunksIterator {
            buffer: self.buffer.clone(),
            bit_offset: self.bit_offset,
            chunk_count: self.len / 64,
            index: 0,
        }
        .chain(iter::once(self.remainder_bits()))
    }
}

impl IntoIterator for BitChunks {
    type Item = u64;
    type IntoIter = PaddedBitChunksIterator;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

pub type PaddedBitChunksIterator = iter::Chain<BitChunksIterator, iter::Once<u64>>;

unsafe impl TrustedLen for PaddedBitChunksIterator {}

pub struct BitChunksIterator {
    buffer: ByteBuffer,
    bit_offset: usize,
    chunk_count: usize,
    index: usize,
}

impl Iterator for BitChunksIterator {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.chunk_count {
            return None;
        }

        let non_offset_chunk = u64::from_le_bytes(
            self.buffer[self.index * 8..(self.index + 1) * 8]
                .try_into()
                .vortex_expect("slice of 8 bytes"),
        );
        let result = if self.bit_offset == 0 {
            non_offset_chunk
        } else {
            let next_byte = self.buffer[(self.index + 1) * 8] as u64;
            (non_offset_chunk >> self.bit_offset) | (next_byte << (64 - self.bit_offset))
        };

        self.index += 1;
        Some(result)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = self.chunk_count - self.index;
        (size, Some(size))
    }
}

impl ExactSizeIterator for BitChunksIterator {}

unsafe impl TrustedLen for BitChunksIterator {}
