// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::trusted_len::TrustedLenExt;
use crate::{BitBuffer, Buffer};

pub(super) fn bitwise_unary_op<F: FnMut(u64) -> u64>(buffer: &BitBuffer, op: F) -> BitBuffer {
    let iter = buffer.chunks().iter_padded().map(op);
    let iter = unsafe { iter.trusted_len() };

    let result = Buffer::<u64>::from_trusted_len_iter(iter).into_byte_buffer();

    BitBuffer::new(result, buffer.len())
}

pub(super) fn bitwise_binary_op<F: FnMut(u64, u64) -> u64>(
    left: &BitBuffer,
    right: &BitBuffer,
    mut op: F,
) -> BitBuffer {
    assert_eq!(left.len(), right.len());

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
    use crate::{bitbuffer, buffer};

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
