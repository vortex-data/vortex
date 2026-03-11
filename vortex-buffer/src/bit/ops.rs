// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::BitBuffer;
use crate::BitBufferMut;
use crate::Buffer;
use crate::trusted_len::TrustedLenExt;

#[inline]
pub(super) fn bitwise_unary_op<F: FnMut(u64) -> u64>(buffer: &BitBuffer, op: F) -> BitBuffer {
    let iter = buffer.unaligned_chunks().iter().map(op);
    let iter = unsafe { iter.trusted_len() };

    let result = Buffer::<u64>::from_trusted_len_iter(iter).into_byte_buffer();

    BitBuffer::new_with_offset(result, buffer.len(), buffer.offset())
}

#[inline]
pub(super) fn bitwise_unary_op_mut<F: FnMut(u64) -> u64>(buffer: &mut BitBufferMut, mut op: F) {
    let slice_mut = buffer.as_mut_slice();

    // The number of complete u64 words in the buffer (unaligned)
    let u64_len = slice_mut.len() / 8;
    let remainder = slice_mut.len() % 8;

    // Create a pointer to the *unaligned* u64 words
    let mut ptr = slice_mut.as_mut_ptr() as *mut u64;
    for _ in 0..u64_len {
        let value = unsafe { ptr.read_unaligned() };
        let value = op(value);
        unsafe { ptr.write_unaligned(value) };
        ptr = unsafe { ptr.add(1) };
    }

    // Read remainder into a u64;
    let mut remainder_u64 = 0u64;
    let ptr = ptr as *mut u8;
    for i in 0..remainder {
        let byte = unsafe { ptr.add(i).read() };
        remainder_u64 |= (byte as u64) << (i * 8);
    }
    let remainder_u64 = op(remainder_u64);

    // Write back remainder
    for i in 0..remainder {
        let byte = ((remainder_u64 >> (i * 8)) & 0xFF) as u8;
        unsafe { ptr.add(i).write(byte) };
    }
}

pub(super) fn bitwise_binary_op<F: FnMut(u64, u64) -> u64>(
    left: &BitBuffer,
    right: &BitBuffer,
    mut op: F,
) -> BitBuffer {
    assert_eq!(left.len(), right.len());

    // If the buffers are aligned, we can use the fast path.
    if left.offset().is_multiple_of(8) && right.offset().is_multiple_of(8) {
        let left_chunks = left.unaligned_chunks();
        let right_chunks = right.unaligned_chunks();
        if left_chunks.lead_padding() == 0
            && left_chunks.trailing_padding() == 0
            && right_chunks.lead_padding() == 0
            && right_chunks.trailing_padding() == 0
        {
            let iter = left_chunks
                .iter()
                .zip(right_chunks.iter())
                .map(|(l, r)| op(l, r));
            let iter = unsafe { iter.trusted_len() };
            let result = Buffer::<u64>::from_trusted_len_iter(iter).into_byte_buffer();
            return BitBuffer::new(result, left.len());
        }
    }

    let iter = left
        .chunks()
        .iter_padded()
        .zip(right.chunks().iter_padded())
        .map(|(l, r)| op(l, r));
    let iter = unsafe { iter.trusted_len() };

    let result = Buffer::<u64>::from_trusted_len_iter(iter).into_byte_buffer();

    BitBuffer::new(result, left.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitbuffer;
    use crate::buffer;

    #[test]
    fn test_bitwise_unary_not() {
        let buffer = BitBuffer::new(buffer![0b10101010u8], 4);
        let result = bitwise_unary_op(&buffer, |x| !x);
        assert_eq!(result, bitbuffer![true, false, true, false]);
    }

    #[test]
    fn test_bitwise_binary_and() {
        // 0b1111 (15) & 0b1010 (10) = 0b1010 (10)
        let left = BitBuffer::new(buffer![15u8], 4);
        let right = BitBuffer::new(buffer![10u8], 4);
        let result = bitwise_binary_op(&left, &right, |l, r| l & r);
        assert_eq!(result, bitbuffer![false, true, false, true]);
    }

    #[test]
    fn test_bitwise_binary_or() {
        // 0b1010 (10) | 0b0101 (5) = 0b1111 (15)
        let left = BitBuffer::new(buffer![10u8], 4);
        let right = BitBuffer::new(buffer![5u8], 4);
        let result = bitwise_binary_op(&left, &right, |l, r| l | r);
        assert_eq!(result, bitbuffer![true; 4]);
    }

    #[test]
    fn test_bitwise_binary_xor() {
        // 0b1100 (12) ^ 0b1010 (10) = 0b0110 (6)
        let left = BitBuffer::new(buffer![12u8], 4);
        let right = BitBuffer::new(buffer![10u8], 4);
        let result = bitwise_binary_op(&left, &right, |l, r| l ^ r);
        assert_eq!(result, bitbuffer![false, true, true, false]);
    }
}
