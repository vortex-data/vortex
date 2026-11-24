// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Copying runs of bits between bitmaps at any bit offset, for blocks' validity.

/// The most bits one step moves: a read or write at any bit offset then fits in one `u64`.
const STEP: usize = 56;

/// Copy `len` bits of `src` from bit `src_bit` to `dst` from bit `dst_bit`, least significant bit
/// first, leaving `dst`'s other bits as they are.
///
/// # Panics
///
/// Panics if either range runs past its bitmap.
pub(super) fn copy_bits(src: &[u8], src_bit: usize, dst: &mut [u8], dst_bit: usize, len: usize) {
    assert!(src_bit + len <= src.len() * 8, "source bits out of range");
    assert!(
        dst_bit + len <= dst.len() * 8,
        "destination bits out of range"
    );
    let mut done = 0;
    while done < len {
        let take = (len - done).min(STEP);
        let bits = read_bits(src, src_bit + done, take);
        write_bits(dst, dst_bit + done, take, bits);
        done += take;
    }
}

/// The `take <= STEP` bits of `bytes` from bit `bit`, in the low bits of the result.
fn read_bits(bytes: &[u8], bit: usize, take: usize) -> u64 {
    let (byte, shift) = (bit / 8, bit % 8);
    let mut word = [0u8; 8];
    let end = (byte + 8).min(bytes.len());
    word[..end - byte].copy_from_slice(&bytes[byte..end]);
    (u64::from_le_bytes(word) >> shift) & low_bits(take)
}

/// Overwrite the `take <= STEP` bits of `bytes` from bit `bit` with the low bits of `bits`.
fn write_bits(bytes: &mut [u8], bit: usize, take: usize, bits: u64) {
    let (byte, shift) = (bit / 8, bit % 8);
    let end = (byte + 8).min(bytes.len());
    let mut word = [0u8; 8];
    word[..end - byte].copy_from_slice(&bytes[byte..end]);
    let mask = low_bits(take) << shift;
    let word = (u64::from_le_bytes(word) & !mask) | ((bits << shift) & mask);
    bytes[byte..end].copy_from_slice(&word.to_le_bytes()[..end - byte]);
}

/// A mask of the low `n <= 64` bits.
fn low_bits(n: usize) -> u64 {
    if n == 64 { u64::MAX } else { (1 << n) - 1 }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::copy_bits;

    fn bit(bytes: &[u8], i: usize) -> bool {
        bytes[i / 8] & (1 << (i % 8)) != 0
    }

    /// Every bit of the range is copied, and every other bit of the destination is kept.
    #[rstest]
    #[case::aligned(0, 0, 1024)]
    #[case::source_offset(3, 0, 1021)]
    #[case::destination_offset(0, 5, 700)]
    #[case::both_offsets(13, 517, 500)]
    #[case::one_bit(7, 9, 1)]
    #[case::to_the_last_byte(1, 0, 1023)]
    fn copies_any_range(#[case] src_bit: usize, #[case] dst_bit: usize, #[case] len: usize) {
        let src: Vec<u8> = (0..128u8)
            .map(|i| i.wrapping_mul(37).wrapping_add(11))
            .collect();
        let mut dst = vec![0b1010_0101u8; 192];
        let before = dst.clone();
        copy_bits(&src, src_bit, &mut dst, dst_bit, len);
        for i in 0..dst.len() * 8 {
            let expected = if (dst_bit..dst_bit + len).contains(&i) {
                bit(&src, src_bit + i - dst_bit)
            } else {
                bit(&before, i)
            };
            assert_eq!(bit(&dst, i), expected, "bit {i}");
        }
    }
}
