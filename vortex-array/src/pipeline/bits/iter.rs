// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_buffer::BooleanBuffer;
use arrow_buffer::bit_chunk_iterator::BitChunkIterator;
use bitvec::order::Lsb0;
use bitvec::slice::{BitSlice, ChunksExact};

use crate::pipeline::PIPELINE_STEP_COUNT;

pub fn iter_boolean_buffer<'a>(buffer: &'a BooleanBuffer) -> ChunksExact<'a, u64, Lsb0> {
    assert_eq!(buffer.offset(), 0, "BooleanBuffer must have an offset of 0");
    let ptr = buffer.inner().as_ptr().cast::<u64>();
    assert!(ptr.is_aligned(), "BooleanBuffer must be aligned to 64 bits");
    let data = unsafe { std::slice::from_raw_parts(ptr, buffer.len()) };
    let slice = BitSlice::<u64, Lsb0>::from_slice(data);
    slice.chunks_exact(PIPELINE_STEP_COUNT)
}

pub struct BooleanBufferChunksIter<'a> {
    bit_chunks: BitChunkIterator<'a>,
    remainder_bits: Option<u64>,
    finished: bool,
}

impl<'a> BooleanBufferChunksIter<'a> {
    pub fn new(buffer: &'a BooleanBuffer) -> Self {
        let bit_chunks = buffer.bit_chunks();
        let remainder_bits = bit_chunks.remainder_bits();
        BooleanBufferChunksIter {
            bit_chunks: bit_chunks.into_iter(),
            remainder_bits: Some(remainder_bits),
            finished: false,
        }
    }
}

impl Iterator for BooleanBufferChunksIter<'_> {
    type Item = [u64; PIPELINE_STEP_COUNT / 64];

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished && self.remainder_bits.is_none() {
            return None;
        }

        // Number of words in a BitVector.
        const W: usize = PIPELINE_STEP_COUNT / 64;
        let mut chunk = [0u64; W];

        // We copy bit-chunks into our bitvector, until we reach the end of the chunks.
        let mut words = 0;
        while words < W {
            match self.bit_chunks.next() {
                None => {
                    self.finished = true;
                    break;
                }
                Some(word) => {
                    chunk[words] = word;
                    words += 1;
                }
            }
        }

        if words < W
            && let Some(remainder_bits) = self.remainder_bits.take()
        {
            chunk[words] = remainder_bits;
        }

        Some(chunk)
    }
}
