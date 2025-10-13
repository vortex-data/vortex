// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;

use crate::bit::get_bit_unchecked;
use crate::trusted_len::TrustedLen;
use crate::{Buffer, BufferIterator, ByteBuffer};

#[inline]
fn read_u64(input: &[u8]) -> u64 {
    let len = input.len().min(8);
    let mut buf = [0u8; 8];
    buf[..len].copy_from_slice(input);
    u64::from_le_bytes(buf)
}

#[inline]
fn compute_prefix_mask(lead_padding: usize) -> u64 {
    !((1 << lead_padding) - 1)
}

#[inline]
fn compute_suffix_mask(len: usize, lead_padding: usize) -> (u64, usize) {
    let trailing_bits = (len + lead_padding) % 64;

    if trailing_bits == 0 {
        return (u64::MAX, 0);
    }

    let trailing_padding = 64 - trailing_bits;
    let suffix_mask = (1 << trailing_bits) - 1;
    (suffix_mask, trailing_padding)
}

pub struct UnalignedBitChunks {
    lead_padding: usize,
    trailing_padding: usize,
    prefix: Option<u64>,
    chunks: Buffer<u64>,
    suffix: Option<u64>,
}

impl UnalignedBitChunks {
    pub fn new(buffer: ByteBuffer, offset: usize, len: usize) -> Self {
        if len == 0 {
            return Self {
                lead_padding: 0,
                trailing_padding: 0,
                prefix: None,
                chunks: Buffer::empty(),
                suffix: None,
            };
        }
        let byte_offset = offset / 8;
        let offset_padding = offset % 8;
        let bytes_len = (len + offset_padding).div_ceil(8);

        let buffer = buffer.slice(byte_offset..byte_offset + bytes_len);

        let prefix_mask = compute_prefix_mask(offset_padding);

        // If less than 8 bytes, read into prefix
        if buffer.len() <= 8 {
            let (suffix_mask, trailing_padding) = compute_suffix_mask(len, offset_padding);
            let prefix = read_u64(&buffer) & suffix_mask & prefix_mask;

            return Self {
                lead_padding: offset_padding,
                trailing_padding,
                prefix: Some(prefix),
                chunks: Buffer::empty(),
                suffix: None,
            };
        }

        // If less than 16 bytes, read into prefix and suffix
        if buffer.len() <= 16 {
            let (suffix_mask, trailing_padding) = compute_suffix_mask(len, offset_padding);
            let prefix = read_u64(&buffer[..8]) & prefix_mask;
            let suffix = read_u64(&buffer[8..]) & suffix_mask;

            return Self {
                lead_padding: offset_padding,
                trailing_padding,
                prefix: Some(prefix),
                chunks: Buffer::empty(),
                suffix: Some(suffix),
            };
        }

        let (prefix, mut chunks, suffix) = buffer.align_to::<u64>();
        assert!(
            prefix.len() < 8 && suffix.len() < 8,
            "align_to did not return largest possible aligned slice"
        );
        let (alignment_padding, prefix) = match (offset_padding, prefix.is_empty()) {
            (0, true) => (0, None),
            (_, true) => {
                let prefix = chunks[0] & prefix_mask;
                chunks = chunks.slice(1..);
                (0, Some(prefix))
            }
            (_, false) => {
                let alignment_padding = (8 - prefix.len()) * 8;

                let prefix = (read_u64(&prefix) & prefix_mask) << alignment_padding;
                (alignment_padding, Some(prefix))
            }
        };

        let lead_padding = offset_padding + alignment_padding;
        let (suffix_mask, trailing_padding) = compute_suffix_mask(len, lead_padding);

        let suffix = match (trailing_padding, suffix.is_empty()) {
            (0, _) => None,
            (_, true) => {
                let suffix = chunks[chunks.len() - 1] & suffix_mask;
                chunks = chunks.slice(..chunks.len() - 1);
                Some(suffix)
            }
            (_, false) => Some(read_u64(&suffix) & suffix_mask),
        };

        Self {
            lead_padding,
            trailing_padding,
            prefix,
            chunks,
            suffix,
        }
    }

    pub fn iter(&self) -> UnalignedBitChunkIterator {
        self.prefix
            .into_iter()
            .chain(self.chunks.clone())
            .chain(self.suffix)
    }

    pub fn prefix_padding(&self) -> usize {
        self.lead_padding
    }

    pub fn prefix(&self) -> Option<u64> {
        self.prefix
    }

    pub fn suffix_padding(&self) -> usize {
        self.trailing_padding
    }

    pub fn suffix(&self) -> Option<u64> {
        self.suffix
    }

    pub fn count_ones(&self) -> usize {
        self.iter().map(|x| x.count_ones() as usize).sum()
    }
}

pub type UnalignedBitChunkIterator = core::iter::Chain<
    core::iter::Chain<core::option::IntoIter<u64>, BufferIterator<u64>>,
    core::option::IntoIter<u64>,
>;

/// Iterator over bits in the byte buffer
pub struct BitIterator {
    buffer: ByteBuffer,
    current_offset: usize,
    end_offset: usize,
}

impl BitIterator {
    pub fn new(buffer: ByteBuffer, offset: usize, len: usize) -> Self {
        let end_offset = offset + len;
        assert!(
            buffer.len() >= end_offset.div_ceil(8),
            "Buffer {} too small for requested offset and len {}",
            buffer.len(),
            end_offset.div_ceil(8)
        );

        Self {
            buffer,
            current_offset: offset,
            end_offset,
        }
    }
}

impl Iterator for BitIterator {
    type Item = bool;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_offset == self.end_offset {
            return None;
        }
        // SAFETY: current_offset is in bounds
        let v = unsafe { get_bit_unchecked(self.buffer.as_ptr(), self.current_offset) };
        self.current_offset += 1;
        Some(v)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining_bits = self.end_offset - self.current_offset;
        (remaining_bits, Some(remaining_bits))
    }
}

unsafe impl TrustedLen for BitIterator {}

impl ExactSizeIterator for BitIterator {}

impl DoubleEndedIterator for BitIterator {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.current_offset == self.end_offset {
            return None;
        }
        self.end_offset -= 1;
        // Safety: end_offset is in bounds
        Some(unsafe { get_bit_unchecked(self.buffer.as_ptr(), self.end_offset) })
    }
}

pub struct BitIndexIterator {
    current_chunk: u64,
    chunk_offset: i64,
    iter: UnalignedBitChunkIterator,
}

impl BitIndexIterator {
    pub fn new(buffer: ByteBuffer, offset: usize, len: usize) -> Self {
        let chunks = UnalignedBitChunks::new(buffer, offset, len);
        let mut iter = chunks.iter();
        let current_chunk = iter.next().unwrap_or(0);
        let chunk_offset = -(chunks.prefix_padding() as i64);

        Self {
            current_chunk,
            chunk_offset,
            iter,
        }
    }
}

impl Iterator for BitIndexIterator {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.current_chunk != 0 {
                let bit_pos = self.current_chunk.trailing_zeros();
                self.current_chunk ^= 1 << bit_pos;
                return Some(
                    usize::try_from(self.chunk_offset + bit_pos as i64)
                        .vortex_expect("bit index must be a usize"),
                );
            }

            self.current_chunk = self.iter.next()?;
            self.chunk_offset += 64;
        }
    }
}

pub struct BitSliceIterator {
    iter: UnalignedBitChunkIterator,
    len: usize,
    current_offset: i64,
    current_chunk: u64,
}

impl BitSliceIterator {
    pub fn new(buffer: ByteBuffer, offset: usize, len: usize) -> Self {
        let chunks = UnalignedBitChunks::new(buffer, offset, len);
        let mut iter = chunks.iter();
        let current_chunk = iter.next().unwrap_or(0);
        let current_offset = -(chunks.prefix_padding() as i64);

        Self {
            iter,
            len,
            current_offset,
            current_chunk,
        }
    }

    /// Returns `Some((chunk_offset, bit_offset))` for the next chunk that has at
    /// least one bit set, or None if there is no such chunk.
    ///
    /// Where `chunk_offset` is the bit offset to the current `u64` chunk
    /// and `bit_offset` is the offset of the first `1` bit in that chunk
    fn advance_to_set_bit(&mut self) -> Option<(i64, u32)> {
        loop {
            if self.current_chunk != 0 {
                // Find the index of the first 1
                let bit_pos = self.current_chunk.trailing_zeros();
                return Some((self.current_offset, bit_pos));
            }

            self.current_chunk = self.iter.next()?;
            self.current_offset += 64;
        }
    }
}

impl Iterator for BitSliceIterator {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.len == 0 {
            return None;
        }

        let (start_chunk, start_bit) = self.advance_to_set_bit()?;

        // Set bits up to start
        self.current_chunk |= (1 << start_bit) - 1;

        loop {
            if self.current_chunk != u64::MAX {
                // Find the index of the first 0
                let end_bit = self.current_chunk.trailing_ones();

                // Zero out up to end_bit
                self.current_chunk &= !((1 << end_bit) - 1);

                return Some((
                    usize::try_from(start_chunk + start_bit as i64)
                        .vortex_expect("bit offset must be a usize"),
                    usize::try_from(self.current_offset + end_bit as i64)
                        .vortex_expect("bit offset must be a usize"),
                ));
            }

            match self.iter.next() {
                Some(next) => {
                    self.current_chunk = next;
                    self.current_offset += 64;
                }
                None => {
                    return Some((
                        usize::try_from(start_chunk + start_bit as i64)
                            .vortex_expect("bit offset must be a usize"),
                        std::mem::take(&mut self.len),
                    ));
                }
            }
        }
    }
}
