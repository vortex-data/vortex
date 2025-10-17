// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_buffer::bit_chunk_iterator::BitChunks;

use crate::{Alignment, BufferMut, ByteBuffer};

impl ByteBuffer {
    /// Returns an accessor which can be used to perform bitwise operations in u64 sized chunks
    pub(super) fn bit_chunks(&self, bit_offset: usize, bit_length: usize) -> BitChunks<'_> {
        BitChunks::new(self.bytes.as_ref(), bit_offset, bit_length)
    }
}

pub(super) fn bitwise_unary_op<F: FnMut(u64) -> u64>(
    buffer: ByteBuffer,
    offset: usize,
    len: usize,
    op: F,
) -> ByteBuffer {
    let mut result = BufferMut::<u64>::empty();
    result.extend_trusted(buffer.bit_chunks(offset, len).iter().map(op));
    result
        .freeze()
        .into_byte_buffer()
        .aligned(Alignment::of::<u8>())
}

pub(super) fn bitwise_binary_op<F: FnMut(u64, u64) -> u64>(
    left_buffer: ByteBuffer,
    left_offset: usize,
    right_buffer: ByteBuffer,
    right_offset: usize,
    len: usize,
    mut op: F,
) -> ByteBuffer {
    let mut result = BufferMut::<u64>::empty();
    result.extend_trusted(
        left_buffer
            .bit_chunks(left_offset, len)
            .iter()
            .zip(right_buffer.bit_chunks(right_offset, len))
            .map(|(l, r)| op(l, r)),
    );
    result
        .freeze()
        .into_byte_buffer()
        .aligned(Alignment::of::<u8>())
}

pub(super) fn bitwise_and(
    left_buffer: ByteBuffer,
    left_offset: usize,
    right_buffer: ByteBuffer,
    right_offset: usize,
    len: usize,
) -> ByteBuffer {
    bitwise_binary_op(
        left_buffer,
        left_offset,
        right_buffer,
        right_offset,
        len,
        |l, r| l & r,
    )
}

pub(super) fn bitwise_or(
    left_buffer: ByteBuffer,
    left_offset: usize,
    right_buffer: ByteBuffer,
    right_offset: usize,
    len: usize,
) -> ByteBuffer {
    bitwise_binary_op(
        left_buffer,
        left_offset,
        right_buffer,
        right_offset,
        len,
        |l, r| l | r,
    )
}

pub(super) fn bitwise_xor(
    left_buffer: ByteBuffer,
    left_offset: usize,
    right_buffer: ByteBuffer,
    right_offset: usize,
    len: usize,
) -> ByteBuffer {
    bitwise_binary_op(
        left_buffer,
        left_offset,
        right_buffer,
        right_offset,
        len,
        |l, r| l ^ r,
    )
}

pub(super) fn bitwise_not(buffer: ByteBuffer, offset: usize, len: usize) -> ByteBuffer {
    bitwise_unary_op(buffer, offset, len, |l| !l)
}
