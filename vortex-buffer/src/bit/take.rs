// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::AsPrimitive;
use num_traits::Zero;
use vortex_error::VortexExpect;

use crate::BitBuffer;
use crate::BitBufferView;
use crate::BufferMut;
use crate::bit::get_bit_unchecked;

// If you have less than a Linux memory page of bits, it pays to convert to a
// Vec
const COLLECT_TO_VEC: usize = 4096;

/// Select bits at indices into a new BitBuffer
pub fn take_bits<I>(
    bits: BitBufferView<'_>,
    indices: &[I],
    validity: Option<BitBufferView<'_>>,
) -> BitBuffer
where
    I: AsPrimitive<usize> + Ord + Zero,
{
    if let Some(validity) = validity {
        assert_eq!(validity.len(), indices.len());
    }

    if indices.is_empty() || bits.is_empty() {
        return BitBuffer::new_unset(indices.len());
    }

    // If we don't have a lot of indices, mostly full/empty case is very fast.
    // This is, for example, the case of comparing a dict to a constant.
    if bits.len() <= indices.len() * 64 {
        match bits.true_count() {
            0 => return BitBuffer::new_unset(indices.len()),
            count if count == bits.len() => return BitBuffer::new_set(indices.len()),
            1 => {
                let target = bits.select(0).vortex_expect("one set bit");
                return BitBuffer::collect_bool_multiversioned(indices.len(), |i| {
                    // SAFETY: we're already iterating over every index, a bound
                    // is excessive
                    unsafe { indices.get_unchecked(i) }.as_() == target
                });
            }
            count if count == bits.len() - 1 => {
                let target = first_unset(bits).vortex_expect("one unset bit");
                return BitBuffer::collect_bool_multiversioned(indices.len(), |i| {
                    // SAFETY: we're already iterating over every index, a bound
                    // is excessive
                    unsafe { indices.get_unchecked(i) }.as_() != target
                });
            }
            _ => {}
        }
    }

    let Some(validity) = validity else {
        // iterating over indices here for a bounds check is faster than a
        // per-element branch in [] operator.
        let first = indices[0];
        let (min, max) = indices
            .iter()
            .fold((first, first), |(lo, hi), &x| (lo.min(x), hi.max(x)));
        assert!(
            min >= I::zero() && max.as_() < bits.len(),
            "take index out of bounds"
        );

        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        if let Some(taken) = lut::take(bits, indices) {
            return taken;
        }

        return if bits.len() <= COLLECT_TO_VEC {
            // The gather beats a byte per bit on CPUs with AVX-512, but not on CPUs with only
            // AVX2, such as Zen 3. Its 32-bit loads need the buffer padded, which a small copy,
            // at most 520 bytes, provides.
            #[cfg(target_arch = "x86_64")]
            if is_x86_feature_detected!("avx512f") {
                let padded = padded_table::<{ COLLECT_TO_VEC / 8 + 8 }>(bits);
                let padded = BitBufferView::new(&padded, bits.len());
                if let Some(taken) = avx2::take(padded, indices, None) {
                    return taken;
                }
            }

            let bools = expand_bools(bits);
            BitBuffer::collect_bool_multiversioned(indices.len(), |i| unsafe {
                // SAFETY: we're already iterating over every index, a bound
                // is excessive
                let idx = indices.get_unchecked(i).as_();
                // SAFETY: we've asserted index is in bounds before
                *bools.get_unchecked(idx)
            })
        } else {
            #[cfg(target_arch = "x86_64")]
            if let Some(taken) = avx2::take(bits, indices, None) {
                return taken;
            }

            let ptr = bits.inner().as_ptr();
            let offset = bits.offset();
            BitBuffer::collect_bool_multiversioned(indices.len(), |i| unsafe {
                // SAFETY: we're already iterating over every index, a bound
                // is excessive
                let idx = indices.get_unchecked(i).as_();
                // SAFETY: we've asserted index is in bounds before
                get_bit_unchecked(ptr, offset + idx)
            })
        };
    };

    #[cfg(target_arch = "x86_64")]
    {
        // The gather needs its 32-bit loads in bounds. Copy a small bitmap into a padded table,
        // because the buffer can end before the last load.
        let taken = if bits.len() <= COLLECT_TO_VEC {
            let padded = padded_table::<{ COLLECT_TO_VEC / 8 + 8 }>(bits);
            avx2::take(
                BitBufferView::new(&padded, bits.len()),
                indices,
                Some(validity),
            )
        } else {
            avx2::take(bits, indices, Some(validity))
        };
        if let Some(taken) = taken {
            return taken;
        }
    }

    if bits.len() <= COLLECT_TO_VEC {
        let bools = expand_bools(bits);
        return take_with_validity(indices, validity, bits.len() - 1, |idx| unsafe {
            // SAFETY: we clamp index so it's never out of bounds
            *bools.get_unchecked(idx)
        });
    }

    let buf = bits.inner().as_ptr();
    let bits_offset = bits.offset();
    take_with_validity(indices, validity, bits.len() - 1, |idx| unsafe {
        // SAFETY: we clamp index so it's never out of bounds
        get_bit_unchecked(buf, bits_offset + idx)
    })
}

fn take_with_validity<I>(
    indices: &[I],
    validity: BitBufferView<'_>,
    max_index: usize,
    read: impl Fn(usize) -> bool,
) -> BitBuffer
where
    I: AsPrimitive<usize>,
{
    let total = indices.len();
    let groups = total / 8;

    let validity_bytes = validity.inner();
    let validity_offset = validity.offset();
    let validity_shift = validity_offset % 8;
    let validity_first_byte = validity_offset / 8;
    let has_non_byte_shift = usize::from(validity_shift != 0);

    let mut out = BufferMut::<u8>::with_capacity(total.div_ceil(8));
    let mut violations = 0usize;
    for group in 0..groups {
        let byte = validity_first_byte + group;
        let next_byte = byte + has_non_byte_shift;
        // SAFETY: caller verified validity and indices length are equal
        let lo: u16 = unsafe { *validity_bytes.get_unchecked(byte) }.as_();
        let hi: u16 = unsafe { *validity_bytes.get_unchecked(next_byte) }.as_();
        let valid = (lo | hi << 8) >> validity_shift;

        let mut packed = 0u8;
        for bit in 0..8 {
            // SAFETY: full groups stay within indices
            let idx = unsafe { indices.get_unchecked(group * 8 + bit) }.as_();
            let keep = (usize::from(valid >> bit) & 1).wrapping_neg();
            let masked = idx & keep;
            let clamped = masked.min(max_index);
            violations += usize::from(masked != clamped);
            packed |= (read(clamped) as u8) << bit;
        }
        // SAFETY: out has sufficient capacity
        unsafe { out.push_unchecked(packed) };
    }

    assert_eq!(violations, 0, "take index out of bounds");

    let tail = total % 8;
    if tail == 0 {
        return BitBuffer::new(out.freeze(), total);
    }

    let start = groups * 8;
    let ptr = validity_bytes.as_ptr();
    let mut packed = 0u8;
    for bit in 0..tail {
        let pos = start + bit;
        // SAFETY: pos stays within indices
        let idx = unsafe { indices.get_unchecked(pos) }.as_();
        // SAFETY: validity has same length as indices
        let keep =
            (unsafe { get_bit_unchecked(ptr, validity_offset + pos) } as usize).wrapping_neg();
        let masked = idx & keep;
        assert!(masked <= max_index, "take index out of bounds");
        packed |= (read(masked) as u8) << bit;
    }
    // SAFETY: out has sufficient capacity
    unsafe { out.push_unchecked(packed) };
    BitBuffer::new(out.freeze(), total)
}

/// Expands each bit into a bool, with a table that holds the 8 bools of each byte value.
///
/// The result is padded with `false` to a multiple of 64 bits, with at least one bit of padding.
fn expand_bools(bits: BitBufferView<'_>) -> Vec<bool> {
    const PATTERNS: [[u8; 8]; 256] = {
        let mut patterns = [[0u8; 8]; 256];
        let mut byte = 0;
        while byte < 256 {
            let mut bit = 0;
            while bit < 8 {
                if (byte >> bit) & 1 == 1 {
                    patterns[byte][bit] = 1;
                }
                bit += 1;
            }
            byte += 1;
        }
        patterns
    };

    // iter_padded always adds a remainder word, also when the length is a multiple of 64.
    let len = (bits.len() / 64 + 1) * 64;
    let mut bools = Vec::<bool>::with_capacity(len);
    let out = bools.spare_capacity_mut().as_mut_ptr().cast::<[u8; 8]>();
    let words = bits.chunks().iter_padded().take(len / 64);
    for (word_idx, word) in words.enumerate() {
        for (byte_idx, byte) in word.to_le_bytes().into_iter().enumerate() {
            // SAFETY: take keeps the writes below len. Each pattern byte is 0 or 1, which is a
            // valid bool.
            unsafe {
                out.add(word_idx * 8 + byte_idx)
                    .write(PATTERNS[usize::from(byte)])
            };
        }
    }

    // SAFETY: iter_padded gives len / 64 words, so the loop above wrote all len bools.
    unsafe { bools.set_len(len) };
    bools
}

/// Copies the bits to the start of a zeroed table of `N` bytes, without the offset.
///
/// The bits must fit in the table.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn padded_table<const N: usize>(bits: BitBufferView<'_>) -> [u8; N] {
    let mut table = [0u8; N];
    for (word, chunk) in bits
        .chunks()
        .iter_padded()
        .zip(table.as_chunks_mut::<8>().0)
    {
        *chunk = word.to_le_bytes();
    }
    table
}

/// Packs the bits that `kernel` returns for each 64 indices into a buffer.
///
/// The last chunk is padded with zeros, so index 0 must be valid.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[inline]
fn collect_words<T: Copy + Default>(
    indices: &[T],
    mut kernel: impl FnMut(&[T; 64]) -> u64,
) -> BitBuffer {
    let total = indices.len();
    let mut out = BufferMut::<u64>::with_capacity(total.div_ceil(64));

    let (chunks, tail) = indices.as_chunks::<64>();
    for chunk in chunks {
        // SAFETY: out has capacity for a word per 64 indices.
        unsafe { out.push_unchecked(kernel(chunk)) };
    }

    if !tail.is_empty() {
        let mut padded = [T::default(); 64];
        padded[..tail.len()].copy_from_slice(tail);

        // SAFETY: out has capacity for a word for the tail.
        unsafe { out.push_unchecked(kernel(&padded)) };
    }

    BitBuffer::new(out.freeze().into_byte_buffer(), total)
}

fn first_unset(bits: BitBufferView<'_>) -> Option<usize> {
    let chunks = bits.chunks();
    for (word_idx, word) in chunks.iter().enumerate() {
        if word != u64::MAX {
            return Some(word_idx * 64 + (!word).trailing_zeros() as usize);
        }
    }

    let remainder_start = bits.len() / 64 * 64;
    let target = remainder_start + (!chunks.remainder_bits()).trailing_zeros() as usize;
    (target < bits.len()).then_some(target)
}

#[cfg(target_arch = "x86_64")]
#[path = "take_avx2.rs"]
mod avx2;

#[cfg(target_arch = "x86_64")]
#[path = "take_lut_x86.rs"]
mod lut;

#[cfg(target_arch = "aarch64")]
#[path = "take_lut_neon.rs"]
mod lut;

#[cfg(test)]
mod tests {
    use hegel::TestCase;
    use hegel::generators as gs;
    use num_traits::AsPrimitive;
    use num_traits::Zero;
    use rand::RngExt;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rstest::rstest;

    use super::take_bits;
    use crate::BitBuffer;

    fn random_bits(len: usize) -> BitBuffer {
        let mut rng = StdRng::seed_from_u64(42);
        BitBuffer::collect_bool(len, |_| rng.random())
    }

    fn random_indices(len: usize, bound: usize) -> Vec<usize> {
        let mut rng = StdRng::seed_from_u64(43);
        (0..len).map(|_| rng.random_range(0..bound)).collect()
    }

    fn gathers<I>(bits_len: usize, garbage: [I; 2])
    where
        I: AsPrimitive<usize> + TryFrom<usize> + Ord + Zero,
        <I as TryFrom<usize>>::Error: std::fmt::Debug,
    {
        let bits = random_bits(bits_len + 11);
        let view = bits.as_view().slice(11..);
        let len = 403;
        let indices: Vec<I> = random_indices(len, bits_len)
            .into_iter()
            .map(|idx| I::try_from(idx).unwrap())
            .collect();

        let taken = take_bits(view, &indices, None);
        assert_eq!(taken.len(), len);
        for (i, &idx) in indices.iter().enumerate() {
            assert_eq!(taken.value(i), bits.value(idx.as_() + 11));
        }

        let validity = BitBuffer::collect_bool(len, |i| i % 3 != 0);
        let with_garbage: Vec<I> = indices
            .iter()
            .enumerate()
            .map(|(i, &idx)| match i % 6 {
                0 => garbage[0],
                3 => garbage[1],
                _ => idx,
            })
            .collect();

        let taken = take_bits(view, &with_garbage, Some(validity.as_view()));
        assert_eq!(taken.len(), len);
        for (i, &idx) in with_garbage.iter().enumerate() {
            if validity.value(i) {
                assert_eq!(taken.value(i), bits.value(idx.as_() + 11));
            }
        }
    }

    fn draw_take<I>(tc: &TestCase, max_len: usize)
    where
        I: AsPrimitive<usize> + TryFrom<usize> + Ord + Zero,
        <I as TryFrom<usize>>::Error: std::fmt::Debug,
    {
        let len = tc.draw(gs::integers::<usize>().min_value(1).max_value(max_len));
        let offset = tc.draw(gs::integers::<usize>().max_value(63));
        let bits: Vec<bool> = tc.draw(
            gs::vecs(gs::booleans())
                .min_size(offset + len)
                .max_size(offset + len),
        );
        let buf = BitBuffer::from(bits.clone());
        let view = buf.as_view().slice(offset..);

        let bound = len.min(1 << (8 * size_of::<I>()).min(16));
        let indices: Vec<usize> = tc.draw(
            gs::vecs(gs::integers::<usize>().max_value(bound - 1))
                .min_size(1)
                .max_size(300),
        );
        let typed: Vec<I> = indices
            .iter()
            .map(|&idx| I::try_from(idx).unwrap())
            .collect();

        let taken = take_bits(view, &typed, None);
        assert_eq!(taken.len(), indices.len());
        for (i, &idx) in indices.iter().enumerate() {
            assert_eq!(taken.value(i), bits[offset + idx], "index {idx} at {i}");
        }
    }

    #[hegel::test]
    #[cfg_attr(miri, ignore)]
    fn takes_u8_from_small_bitmaps(tc: TestCase) {
        draw_take::<u8>(&tc, 600);
    }

    #[hegel::test]
    #[cfg_attr(miri, ignore)]
    fn takes_u16_from_small_bitmaps(tc: TestCase) {
        draw_take::<u16>(&tc, 4096);
    }

    #[rstest]
    #[case(300)]
    #[case(5000)]
    fn gathers_i64(#[case] bits_len: usize) {
        gathers::<i64>(bits_len, [i64::MAX, -1]);
    }

    #[rstest]
    #[case(300)]
    #[case(4096)]
    #[case(5000)]
    fn gathers_u16(#[case] bits_len: usize) {
        gathers::<u16>(bits_len, [u16::MAX, u16::MAX - 1]);
    }

    #[rstest]
    #[case(300)]
    #[case(4096)]
    #[case(5000)]
    fn gathers_u32(#[case] bits_len: usize) {
        gathers::<u32>(bits_len, [u32::MAX, u32::MAX - 1]);
    }

    #[rstest]
    #[case(BitBuffer::new_set(100), |_: usize| true)]
    #[case(BitBuffer::new_unset(100), |_: usize| false)]
    #[case(BitBuffer::collect_bool(100, |i| i == 37), |idx: usize| idx == 37)]
    #[case(BitBuffer::collect_bool(100, |i| i != 37), |idx: usize| idx != 37)]
    fn nearly_full(#[case] bits: BitBuffer, #[case] expected: fn(usize) -> bool) {
        let indices: Vec<u8> = random_indices(300, bits.len())
            .into_iter()
            .map(|idx| u8::try_from(idx).unwrap())
            .collect();

        let taken = take_bits(bits.as_view(), &indices, None);
        for (i, &idx) in indices.iter().enumerate() {
            assert_eq!(taken.value(i), expected(usize::from(idx)));
        }
    }

    #[test]
    fn small_indices() {
        let bits = BitBuffer::collect_bool(10_000, |i| i != 37);
        let taken = take_bits(bits.as_view(), &[36u32, 37, 38], None);
        assert!(taken.value(0));
        assert!(!taken.value(1));
        assert!(taken.value(2));
    }

    #[test]
    fn empty() {
        let bits = random_bits(64);
        assert_eq!(take_bits::<u16>(bits.as_view(), &[], None).len(), 0);

        let bits = BitBuffer::new_unset(0);
        let validity = BitBuffer::new_unset(3);
        let taken = take_bits(bits.as_view(), &[7u32, 8, 9], Some(validity.as_view()));
        assert_eq!(taken, BitBuffer::new_unset(3));
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn out_of_bounds() {
        let bits = BitBuffer::collect_bool(10_000, |i| i % 2 == 0);
        let view = bits.as_view().slice(3..);
        let mut indices: Vec<u32> = random_indices(403, 9_000)
            .into_iter()
            .map(|idx| u32::try_from(idx).unwrap())
            .collect();
        indices[80] = u32::MAX;
        let validity = BitBuffer::collect_bool(403, |i| i != 200);
        let _taken = take_bits(view, &indices, Some(validity.as_view()));
    }
}
