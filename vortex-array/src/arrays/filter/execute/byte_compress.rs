// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Byte-level compress for primitive filtering using a `1 << 8 = 256`-entry lookup table.
//!
//! For each byte of the mask (8 bits -> 8 source elements), a precomputed
//! permutation table compacts the selected bytes in a single indexed copy,
//! avoiding the overhead of materializing indices or slices.

use std::mem::size_of;

use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_mask::MaskValues;

const BYTE_COMPRESS_DENSITY_THRESHOLD: f64 = 0.5;

/// For each mask byte (0..256), stores the element indices to keep and the count.
///
/// `BYTE_COMPRESS_LUT[mask_byte]` = `([i0, i1, ..., i7], popcount)` where
/// `i0..i_{popcount-1}` are the positions of set bits in `mask_byte`.
///
/// Total size: 256 * 9 = 2304 bytes, which trivially fits in L1 cache.
static BYTE_COMPRESS_LUT: &[([u8; 8], u8); 256] = &{
    let mut lut = [([0u8; 8], 0u8); 256];
    let mut mask: usize = 0;
    while mask < 256 {
        let mut indices = [0u8; 8];
        let mut count: u8 = 0;
        let mut bit: u8 = 0;
        while bit < 8 {
            if mask & (1 << bit) != 0 {
                indices[count as usize] = bit;
                count += 1;
            }
            bit += 1;
        }
        lut[mask] = (indices, count);
        mask += 1;
    }
    lut
};

/// Filter a `Buffer<T>` using the byte-compress LUT.
///
/// Processes the mask one byte at a time (8 source elements per byte),
/// using a precomputed permutation to compact selected elements.
pub(super) fn filter_buffer<T: Copy>(buffer: Buffer<T>, mask: &MaskValues) -> Buffer<T> {
    debug_assert_eq!(buffer.len(), mask.len());

    let src = buffer.as_slice();
    let true_count = mask.true_count();

    if true_count == 0 {
        return Buffer::empty();
    }

    let mask_buffer = mask.bit_buffer();
    let mask_bytes = mask_buffer.inner().as_ref();
    let mask_offset = mask_buffer.offset();

    // Fast path: byte-wide values benefit from avoiding index materialization more often. Wider
    // values need enough selected values to justify scanning every mask byte directly.
    if size_of::<T>() == 1 || mask.density() >= BYTE_COMPRESS_DENSITY_THRESHOLD {
        return filter_bitpacked(src, mask_bytes, mask_offset, true_count);
    }

    // Slow path: lower-density wide values are better handled by the generic path.
    super::slice::filter_slice_by_mask_values(src, mask)
}

fn filter_bitpacked<T: Copy>(
    src: &[T],
    mask_bytes: &[u8],
    mask_offset: usize,
    true_count: usize,
) -> Buffer<T> {
    let mut out = BufferMut::<T>::with_capacity(true_count);
    let mut write_pos: usize = 0;

    if mask_offset == 0 {
        filter_aligned_into(src, mask_bytes, &mut out, &mut write_pos);
    } else {
        let head_len = (8 - mask_offset).min(src.len());
        let head_mask = (mask_bytes[0] >> mask_offset) & low_bits_mask(head_len);
        filter_chunk_into(&src[..head_len], head_mask, &mut out, &mut write_pos);
        filter_aligned_into(&src[head_len..], &mask_bytes[1..], &mut out, &mut write_pos);
    }

    debug_assert_eq!(write_pos, true_count);
    // SAFETY: we wrote exactly true_count elements.
    unsafe { out.set_len(true_count) };
    out.freeze()
}

/// Aligned fast path: mask bits start at byte boundary.
fn filter_aligned_into<T: Copy>(
    src: &[T],
    mask_bytes: &[u8],
    out: &mut BufferMut<T>,
    write_pos: &mut usize,
) {
    let full_bytes = src.len() / 8;
    let remainder = src.len() % 8;

    for i in 0..full_bytes {
        let m = mask_bytes[i];
        if m == 0 {
            continue;
        }
        let chunk = &src[i * 8..i * 8 + 8];
        filter_chunk_into(chunk, m, out, write_pos);
    }

    // Handle the final partial chunk.
    if remainder > 0 {
        let m = mask_bytes[full_bytes] & low_bits_mask(remainder);
        if m != 0 {
            let chunk = &src[full_bytes * 8..];
            filter_chunk_into(chunk, m, out, write_pos);
        }
    }
}

fn filter_chunk_into<T: Copy>(
    chunk: &[T],
    mask_byte: u8,
    out: &mut BufferMut<T>,
    write_pos: &mut usize,
) {
    if mask_byte == 0 {
        return;
    }

    let out_ptr = out.spare_capacity_mut().as_mut_ptr();
    if chunk.len() == 8 && mask_byte == 0xFF {
        // All 8 selected, so bulk copy.
        // SAFETY: write_pos + 8 <= capacity.
        unsafe {
            std::ptr::copy_nonoverlapping(chunk.as_ptr(), out_ptr.add(*write_pos).cast::<T>(), 8);
        }
        *write_pos += 8;
        return;
    }

    let (perm, count) = &BYTE_COMPRESS_LUT[mask_byte as usize];
    let count = *count as usize;
    debug_assert_eq!(mask_byte & !low_bits_mask(chunk.len()), 0);
    // SAFETY: perm indices are all < chunk.len(), write_pos + count <= capacity.
    unsafe {
        for j in 0..count {
            out_ptr
                .add(*write_pos + j)
                .cast::<T>()
                .write(*chunk.get_unchecked(*perm.get_unchecked(j) as usize));
        }
    }
    *write_pos += count;
}

fn low_bits_mask(bits: usize) -> u8 {
    debug_assert!(bits <= 8);
    if bits == 8 {
        u8::MAX
    } else {
        (1u8 << bits) - 1
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::cast_possible_truncation)]
mod tests {
    use vortex_buffer::BitBuffer;
    use vortex_buffer::buffer;
    use vortex_mask::Mask;

    use super::*;

    fn mask_values(mask: &Mask) -> &MaskValues {
        match mask {
            Mask::Values(v) => v.as_ref(),
            _ => panic!("expected Mask::Values"),
        }
    }

    fn offset_mask<const N: usize>(values: [bool; N], offset: usize) -> Mask {
        let bit_buffer =
            BitBuffer::from_iter(std::iter::repeat_n(false, offset).chain(values.iter().copied()));
        Mask::from_buffer(BitBuffer::new_with_offset(
            bit_buffer.inner().clone(),
            values.len(),
            offset,
        ))
    }

    #[test]
    fn test_filter_all_selected() {
        let buf = buffer![1u8, 2, 3, 4, 5, 6, 7, 8, 9];
        let mask = Mask::from_iter([true, true, true, true, true, true, true, true, false]);
        let result = filter_buffer(buf, mask_values(&mask));
        assert_eq!(result, buffer![1u8, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn test_filter_mostly_false() {
        let buf = buffer![1u8, 2, 3, 4, 5, 6, 7, 8, 9];
        let mask = Mask::from_iter([false, false, false, false, false, false, false, false, true]);
        let result = filter_buffer(buf, mask_values(&mask));
        assert_eq!(result, buffer![9u8]);
    }

    #[test]
    fn test_filter_alternating() {
        let buf = buffer![10u8, 20, 30, 40, 50, 60, 70, 80];
        let mask = Mask::from_iter([true, false, true, false, true, false, true, false]);
        let result = filter_buffer(buf, mask_values(&mask));
        assert_eq!(result, buffer![10u8, 30, 50, 70]);
    }

    #[test]
    fn test_filter_with_remainder() {
        let buf = buffer![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let mask = Mask::from_iter([
            true, false, true, false, true, false, true, false, true, true,
        ]);
        let result = filter_buffer(buf, mask_values(&mask));
        assert_eq!(result, buffer![1u8, 3, 5, 7, 9, 10]);
    }

    #[test]
    fn test_filter_large() -> vortex_error::VortexResult<()> {
        let data: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();
        let buf = Buffer::from(BufferMut::from_iter(data.iter().copied()));
        let mask = Mask::from_iter((0..1000).map(|i| i % 3 == 0));
        let result = filter_buffer(buf, mask_values(&mask));
        let expected: Vec<u8> = data.iter().copied().step_by(3).collect();
        assert_eq!(result.as_slice(), &expected[..]);
        Ok(())
    }

    #[test]
    fn test_filter_signed_and_wider_integers() {
        let mask = Mask::from_iter([true, false, true, true, false, true, false, true, true]);

        let i8_result = filter_buffer(
            buffer![-5i8, -4, -3, -2, -1, 0, 1, 2, 3],
            mask_values(&mask),
        );
        assert_eq!(i8_result, buffer![-5i8, -3, -2, 0, 2, 3]);

        let u16_result = filter_buffer(
            buffer![10u16, 20, 30, 40, 50, 60, 70, 80, 90],
            mask_values(&mask),
        );
        assert_eq!(u16_result, buffer![10u16, 30, 40, 60, 80, 90]);

        let i32_result = filter_buffer(
            buffer![-100i32, -50, 0, 50, 100, 150, 200, 250, 300],
            mask_values(&mask),
        );
        assert_eq!(i32_result, buffer![-100i32, 0, 50, 150, 250, 300]);
    }

    #[test]
    fn test_filter_unaligned_byte_mask() {
        let mask = offset_mask(
            [
                false, false, true, false, false, false, false, false, true, false, false,
            ],
            3,
        );

        let result = filter_buffer(
            buffer![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            mask_values(&mask),
        );
        assert_eq!(result, buffer![3u8, 9]);
    }

    #[test]
    fn test_filter_unaligned_wide_mask() {
        let mask = offset_mask(
            [
                true, false, true, true, false, true, false, true, true, false, false, true,
            ],
            5,
        );

        let result = filter_buffer(
            buffer![10u16, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
            mask_values(&mask),
        );
        assert_eq!(result, buffer![10u16, 30, 40, 60, 80, 90, 120]);
    }
}
