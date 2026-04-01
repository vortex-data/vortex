// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Apply a [`BitBuffer`] mask to a typed buffer, zeroing elements where the mask bit is unset.

use std::ptr;

use crate::BitBuffer;
use crate::Buffer;
use crate::BufferMut;

/// Apply a bitmask to a [`Buffer<T>`], zeroing out elements where the corresponding bit is 0
/// and leaving elements unchanged where the bit is 1.
///
/// This attempts in-place mutation via [`Buffer::try_into_mut`] when the buffer has exclusive
/// ownership, falling back to an allocating path otherwise.
///
/// # Panics
///
/// Panics if the buffer length does not equal the mask length.
///
/// # Safety (logical)
///
/// The caller must ensure that all-zero bytes is a valid representation for `T`. This is true
/// for all primitive integer and floating-point types.
pub fn mask_zeroed<T: Copy>(buffer: Buffer<T>, mask: &BitBuffer) -> Buffer<T> {
    assert_eq!(
        buffer.len(),
        mask.len(),
        "buffer length ({}) must equal mask length ({})",
        buffer.len(),
        mask.len()
    );

    if mask.true_count() == mask.len() {
        // All bits set — nothing to zero.
        return buffer;
    }

    if mask.false_count() == mask.len() {
        // All bits unset — return an all-zeros buffer.
        return Buffer::zeroed(buffer.len());
    }

    match buffer.try_into_mut() {
        Ok(mut buf) => {
            mask_zeroed_slice(buf.as_mut_slice(), mask);
            buf.freeze()
        }
        Err(buffer) => mask_zeroed_alloc(buffer.as_slice(), mask),
    }
}

/// Zero out elements of `slice` in-place where the corresponding bit in `mask` is 0.
///
/// Processes 64 elements at a time using the [`BitBuffer`]'s u64 chunk representation,
/// with fast paths for all-ones and all-zeros chunks.
///
/// # Panics
///
/// Panics if the slice length does not equal the mask length.
pub fn mask_zeroed_slice<T: Copy>(slice: &mut [T], mask: &BitBuffer) {
    assert_eq!(
        slice.len(),
        mask.len(),
        "slice length ({}) must equal mask length ({})",
        slice.len(),
        mask.len()
    );

    let len = slice.len();
    if len == 0 {
        return;
    }

    let chunks = mask.chunks();
    let full_chunks = len / 64;
    let remainder = len % 64;

    for (chunk_idx, chunk) in chunks.iter().enumerate() {
        let base = chunk_idx * 64;

        if chunk == u64::MAX {
            // All 64 bits set — skip, values are already correct.
            continue;
        }

        if chunk == 0 {
            // All 64 bits unset — bulk zero.
            // SAFETY: base..base+64 is within bounds (we're iterating full chunks).
            unsafe {
                ptr::write_bytes(slice.as_mut_ptr().add(base), 0, 64);
            }
            continue;
        }

        // Mixed chunk: zero individual elements where the bit is 0.
        zero_by_chunk(slice, base, chunk, 64);
    }

    if remainder > 0 {
        let chunk = chunks.remainder_bits();
        let base = full_chunks * 64;
        zero_by_chunk(slice, base, chunk, remainder);
    }
}

/// For a single u64 chunk, zero out elements where the corresponding bit is 0.
#[inline(always)]
fn zero_by_chunk<T: Copy>(slice: &mut [T], base: usize, chunk: u64, count: usize) {
    // Walk the *unset* bits: invert the chunk and iterate set bits of the inverse.
    let mut unset = !chunk;
    // Mask off bits beyond `count` to avoid out-of-bounds.
    if count < 64 {
        unset &= (1u64 << count) - 1;
    }
    while unset != 0 {
        let bit_idx = unset.trailing_zeros() as usize;
        // SAFETY: bit_idx < count and base + bit_idx < slice.len().
        unsafe {
            ptr::write_bytes(slice.as_mut_ptr().add(base + bit_idx), 0, 1);
        }
        // Clear the lowest set bit.
        unset &= unset - 1;
    }
}

/// Allocating path: create a new buffer, copying values where the bit is set and writing zeros
/// where the bit is unset.
fn mask_zeroed_alloc<T: Copy>(src: &[T], mask: &BitBuffer) -> Buffer<T> {
    let len = src.len();
    let mut dst = BufferMut::<T>::zeroed(len);
    let dst_slice = dst.as_mut_slice();

    let chunks = mask.chunks();
    let full_chunks = len / 64;
    let remainder = len % 64;

    for (chunk_idx, chunk) in chunks.iter().enumerate() {
        let base = chunk_idx * 64;

        if chunk == 0 {
            // All zeros — dst is already zeroed.
            continue;
        }

        if chunk == u64::MAX {
            // All ones — bulk copy.
            // SAFETY: base..base+64 is within bounds.
            unsafe {
                ptr::copy_nonoverlapping(
                    src.as_ptr().add(base),
                    dst_slice.as_mut_ptr().add(base),
                    64,
                );
            }
            continue;
        }

        // Mixed: copy only where bits are set.
        let mut set = chunk;
        while set != 0 {
            let bit_idx = set.trailing_zeros() as usize;
            dst_slice[base + bit_idx] = src[base + bit_idx];
            set &= set - 1;
        }
    }

    if remainder > 0 {
        let chunk = chunks.remainder_bits();
        let base = full_chunks * 64;

        let mut set = chunk & ((1u64 << remainder) - 1);
        while set != 0 {
            let bit_idx = set.trailing_zeros() as usize;
            dst_slice[base + bit_idx] = src[base + bit_idx];
            set &= set - 1;
        }
    }

    dst.freeze()
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation)]
mod tests {
    use rstest::rstest;
    use vortex_error::VortexResult;

    use super::*;
    use crate::buffer;

    #[test]
    fn test_mask_zeroed_basic() -> VortexResult<()> {
        let buf = buffer![10u32, 20, 30, 40, 50];
        let mask = BitBuffer::from_iter([true, false, true, false, true]);

        let result = mask_zeroed(buf, &mask);
        assert_eq!(result.as_slice(), &[10u32, 0, 30, 0, 50]);
        Ok(())
    }

    #[test]
    fn test_mask_zeroed_all_true() -> VortexResult<()> {
        let buf = buffer![1u64, 2, 3, 4];
        let mask = BitBuffer::new_set(4);

        let result = mask_zeroed(buf, &mask);
        assert_eq!(result.as_slice(), &[1u64, 2, 3, 4]);
        Ok(())
    }

    #[test]
    fn test_mask_zeroed_all_false() -> VortexResult<()> {
        let buf = buffer![1u32, 2, 3, 4];
        let mask = BitBuffer::new_unset(4);

        let result = mask_zeroed(buf, &mask);
        assert_eq!(result.as_slice(), &[0u32, 0, 0, 0]);
        Ok(())
    }

    #[rstest]
    #[case(63)]
    #[case(64)]
    #[case(65)]
    #[case(128)]
    #[case(129)]
    #[case(1000)]
    fn test_mask_zeroed_large(#[case] len: usize) -> VortexResult<()> {
        let buf = Buffer::from(BufferMut::from_iter(0u32..len as u32));
        // Zero out every other element.
        let mask = BitBuffer::collect_bool(len, |i| i % 2 == 0);

        let result = mask_zeroed(buf, &mask);
        for i in 0..len {
            if i % 2 == 0 {
                assert_eq!(result.as_slice()[i], i as u32);
            } else {
                assert_eq!(result.as_slice()[i], 0);
            }
        }
        Ok(())
    }

    #[test]
    fn test_mask_zeroed_slice_in_place() -> VortexResult<()> {
        let mut values = vec![100i64, 200, 300, 400, 500];
        let mask = BitBuffer::from_iter([true, true, false, false, true]);

        mask_zeroed_slice(&mut values, &mask);
        assert_eq!(values, vec![100i64, 200, 0, 0, 500]);
        Ok(())
    }

    #[test]
    fn test_mask_zeroed_i8() -> VortexResult<()> {
        let buf = buffer![1i8, -2, 3, -4, 5, -6, 7, -8];
        let mask = BitBuffer::from_iter([false, true, false, true, false, true, false, true]);

        let result = mask_zeroed(buf, &mask);
        assert_eq!(result.as_slice(), &[0i8, -2, 0, -4, 0, -6, 0, -8]);
        Ok(())
    }

    #[test]
    #[should_panic(expected = "must equal mask length")]
    fn test_mask_zeroed_length_mismatch() {
        let buf = buffer![1u32, 2, 3];
        let mask = BitBuffer::new_set(5);
        mask_zeroed(buf, &mask);
    }
}
