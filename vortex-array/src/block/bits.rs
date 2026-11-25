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
    // Into a byte-aligned destination, such as a block's window or the output at a block's
    // start, move whole words while the source has a byte past the word to shift in.
    if dst_bit.is_multiple_of(8) {
        let dst_byte = dst_bit / 8;
        while len - done >= 64 {
            let Some(word) = read_word(src, src_bit + done) else {
                break;
            };
            let at = dst_byte + done / 8;
            dst[at..at + 8].copy_from_slice(&word.to_le_bytes());
            done += 64;
        }
    }
    while done < len {
        let take = (len - done).min(STEP);
        let bits = read_bits(src, src_bit + done, take);
        write_bits(dst, dst_bit + done, take, bits);
        done += take;
    }
}

/// The 64 bits of `bytes` from bit `bit`, if `bytes` has a byte past them to shift in.
fn read_word(bytes: &[u8], bit: usize) -> Option<u64> {
    let (byte, shift) = (bit / 8, bit % 8);
    let (word, next) = bytes.get(byte..byte + 9)?.split_first_chunk::<8>()?;
    let word = u64::from_le_bytes(*word);
    Some(if shift == 0 {
        word
    } else {
        (word >> shift) | (u64::from(next[0]) << (64 - shift))
    })
}

/// The `take <= STEP` bits of `bytes` from bit `bit`, in the low bits of the result.
fn read_bits(bytes: &[u8], bit: usize, take: usize) -> u64 {
    let (byte, shift) = (bit / 8, bit % 8);
    let word = match bytes[byte..].first_chunk::<8>() {
        Some(word) => *word,
        // Within 8 bytes of the end, read the bytes that are there.
        None => {
            let mut word = [0u8; 8];
            word[..bytes.len() - byte].copy_from_slice(&bytes[byte..]);
            word
        }
    };
    (u64::from_le_bytes(word) >> shift) & low_bits(take)
}

/// Overwrite the `take <= STEP` bits of `bytes` from bit `bit` with the low bits of `bits`.
fn write_bits(bytes: &mut [u8], bit: usize, take: usize, bits: u64) {
    let (byte, shift) = (bit / 8, bit % 8);
    let mask = low_bits(take) << shift;
    let merge = |word: u64| (word & !mask) | ((bits << shift) & mask);
    if let Some(word) = bytes[byte..].first_chunk_mut::<8>() {
        *word = merge(u64::from_le_bytes(*word)).to_le_bytes();
        return;
    }
    // Within 8 bytes of the end, write back only the bytes that are there.
    let tail = &mut bytes[byte..];
    let mut word = [0u8; 8];
    word[..tail.len()].copy_from_slice(tail);
    let word = merge(u64::from_le_bytes(word)).to_le_bytes();
    tail.copy_from_slice(&word[..tail.len()]);
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
    #[case::byte_aligned_destination(13, 8, 900)]
    #[case::both_byte_aligned(8, 16, 960)]
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
