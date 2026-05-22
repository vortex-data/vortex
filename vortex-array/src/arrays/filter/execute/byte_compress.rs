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

    let mask_bytes = mask.bit_buffer().inner().as_ref();
    let mask_offset = mask.bit_buffer().offset();

    // Fast path: byte-wide values benefit from avoiding index materialization more often. Wider
    // values need enough selected values to justify scanning every mask byte directly.
    if mask_offset == 0
        && (size_of::<T>() == 1 || mask.density() >= BYTE_COMPRESS_DENSITY_THRESHOLD)
    {
        return filter_aligned(src, mask_bytes, true_count);
    }

    // Slow path: lower-density wide values or unaligned masks are better handled by the generic path.
    super::slice::filter_slice_by_mask_values(src, mask)
}

/// Aligned fast path: mask bits start at byte boundary.
fn filter_aligned<T: Copy>(src: &[T], mask_bytes: &[u8], true_count: usize) -> Buffer<T> {
    let mut out = BufferMut::<T>::with_capacity(true_count);
    let out_ptr = out.spare_capacity_mut().as_mut_ptr();
    let mut write_pos: usize = 0;

    let full_bytes = src.len() / 8;
    let remainder = src.len() % 8;

    for i in 0..full_bytes {
        let m = mask_bytes[i];
        if m == 0 {
            continue;
        }
        let chunk = &src[i * 8..];
        if m == 0xFF {
            // All 8 selected, so bulk copy.
            // SAFETY: write_pos + 8 <= true_count <= capacity.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    chunk.as_ptr(),
                    out_ptr.add(write_pos).cast::<T>(),
                    8,
                );
            }
            write_pos += 8;
        } else {
            let (perm, count) = &BYTE_COMPRESS_LUT[m as usize];
            let count = *count as usize;
            // SAFETY: perm indices are all < 8, write_pos + count <= true_count.
            unsafe {
                for j in 0..count {
                    out_ptr
                        .add(write_pos + j)
                        .cast::<T>()
                        .write(*chunk.get_unchecked(*perm.get_unchecked(j) as usize));
                }
            }
            write_pos += count;
        }
    }

    // Handle the final partial chunk.
    if remainder > 0 {
        let m = mask_bytes[full_bytes] & ((1u8 << remainder) - 1);
        if m != 0 {
            let chunk = &src[full_bytes * 8..];
            let (perm, count) = &BYTE_COMPRESS_LUT[m as usize];
            let count = *count as usize;
            unsafe {
                for j in 0..count {
                    out_ptr
                        .add(write_pos + j)
                        .cast::<T>()
                        .write(*chunk.get_unchecked(*perm.get_unchecked(j) as usize));
                }
            }
            write_pos += count;
        }
    }

    debug_assert_eq!(write_pos, true_count);
    // SAFETY: we wrote exactly true_count bytes.
    unsafe { out.set_len(true_count) };
    out.freeze()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::cast_possible_truncation)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_mask::Mask;

    use super::*;

    fn mask_values(mask: &Mask) -> &MaskValues {
        match mask {
            Mask::Values(v) => v.as_ref(),
            _ => panic!("expected Mask::Values"),
        }
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
}
