// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::AsPrimitive;
use num_traits::Zero;
use vortex_error::VortexExpect;

use crate::BitBuffer;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::BitBufferMut;
use crate::BitBufferView;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::BufferAllocatorRef;
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
        return take_all_valid(bits, indices);
    };

    take_nullable(bits, indices, validity)
}

/// Takes the bits at `indices`, which the caller checked to be in bounds.
fn take_all_valid<I: AsPrimitive<usize>>(bits: BitBufferView<'_>, indices: &[I]) -> BitBuffer {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    if let Some(taken) = lut::take(bits, indices) {
        return taken;
    }

    // On a small bitmap, the gather beats a byte per bit on CPUs with AVX-512, but not on CPUs
    // with only AVX2, such as Zen 3.
    #[cfg(target_arch = "x86_64")]
    if (bits.len() > COLLECT_TO_VEC || is_x86_feature_detected!("avx512f"))
        && let Some(taken) = gather(bits, indices, None)
    {
        return taken;
    }

    if bits.len() <= COLLECT_TO_VEC {
        let bools = expand_bools(bits);
        return BitBuffer::collect_bool_multiversioned(indices.len(), |i| unsafe {
            // SAFETY: i is less than indices.len(), and the caller checked that every index is in
            // bounds.
            *bools.get_unchecked(indices.get_unchecked(i).as_())
        });
    }

    let ptr = bits.inner().as_ptr();
    let offset = bits.offset();
    BitBuffer::collect_bool_multiversioned(indices.len(), |i| unsafe {
        // SAFETY: i is less than indices.len(), and the caller checked that every index is in
        // bounds.
        get_bit_unchecked(ptr, offset + indices.get_unchecked(i).as_())
    })
}

/// Takes the bits at `indices` where `validity` is set. The other indices can hold any value.
fn take_nullable<I: AsPrimitive<usize>>(
    bits: BitBufferView<'_>,
    indices: &[I],
    validity: BitBufferView<'_>,
) -> BitBuffer {
    #[cfg(target_arch = "x86_64")]
    if let Some(taken) = gather(bits, indices, Some(validity)) {
        return taken;
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

/// Gathers the bits with AVX2, or returns `None` if the CPU has no AVX2.
///
/// The gather's 32-bit loads can pass the end of a small bitmap's buffer, so a small bitmap is
/// copied into a padded table first, at most 520 bytes.
#[cfg(target_arch = "x86_64")]
fn gather<I: AsPrimitive<usize>>(
    bits: BitBufferView<'_>,
    indices: &[I],
    validity: Option<BitBufferView<'_>>,
) -> Option<BitBuffer> {
    if bits.len() > COLLECT_TO_VEC {
        return avx2::take(bits, indices, validity);
    }

    let padded = padded_table::<{ COLLECT_TO_VEC / 8 + 8 }>(bits);
    avx2::take(BitBufferView::new(&padded, bits.len()), indices, validity)
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

/// Packs the bits that `fill` writes, one word for each chunk of 64 indices, into a buffer.
///
/// `fill` gets the whole chunks first, then the last chunk padded with zeros, so index 0 must be
/// valid. The bits of the padding are cleared, so the bits after `len` are zero, as with the other
/// producers. `fill` takes a slice of chunks, so a SIMD kernel loops over them in one call.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[inline]
fn collect_words<T: Copy + Default>(
    indices: &[T],
    mut fill: impl FnMut(&[[T; 64]], &mut [u64]),
) -> BitBuffer {
    let allocator = BufferAllocatorRef::statically_allocated();
    BitBufferMut::collect_words_in(indices.len(), allocator, |words| {
        let (chunks, tail) = indices.as_chunks::<64>();
        let (words, tail_word) = words.split_at_mut(chunks.len());
        fill(chunks, words);

        if !tail.is_empty() {
            let mut padded = [T::default(); 64];
            padded[..tail.len()].copy_from_slice(tail);
            fill(&[padded], tail_word);

            // The tail holds fewer than 64 indices, so the shift does not overflow.
            tail_word[0] &= (1u64 << tail.len()) - 1;
        }
    })
    .freeze()
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

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[path = "take_lut.rs"]
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

    use super::first_unset;
    use super::take_bits;
    use crate::BitBuffer;
    use crate::BitBufferView;

    fn random_bits(len: usize) -> BitBuffer {
        let mut rng = StdRng::seed_from_u64(42);
        BitBuffer::collect_bool(len, |_| rng.random())
    }

    fn random_indices(len: usize, bound: usize) -> Vec<usize> {
        let mut rng = StdRng::seed_from_u64(43);
        (0..len).map(|_| rng.random_range(0..bound)).collect()
    }

    fn random_model(len: usize) -> Vec<bool> {
        let mut rng = StdRng::seed_from_u64(44);
        (0..len).map(|_| rng.random()).collect()
    }

    /// Packs `model` LSB-first from bit `offset`, padded so that the AVX2 kernel's 32-bit
    /// gathers stay in bounds and it does not fall back to the scalar path.
    fn pack(model: &[bool], offset: usize) -> Vec<u8> {
        let mut bytes = vec![0u8; (offset + model.len()).div_ceil(32) * 4 + 8];
        for (i, _) in model.iter().enumerate().filter(|&(_, &bit)| bit) {
            bytes[(offset + i) / 8] |= 1 << ((offset + i) % 8);
        }
        bytes
    }

    fn take_matches_model<I>(
        bits_len: usize,
        bits_offset: usize,
        validity_offset: Option<usize>,
        garbage: I,
    ) where
        I: AsPrimitive<usize> + TryFrom<usize> + Ord + Zero,
        <I as TryFrom<usize>>::Error: std::fmt::Debug,
    {
        let model = random_model(bits_len);
        let bytes = pack(&model, bits_offset);
        let bits = BitBufferView::new_with_offset(&bytes, bits_len, bits_offset);

        let len = 403;
        let mut positions = random_indices(len, bits_len);
        positions[0] = bits_len - 1;
        positions[len - 1] = bits_len - 1;
        let valid: Vec<bool> = (0..len)
            .map(|i| validity_offset.is_none() || i % 3 != 1)
            .collect();
        let indices: Vec<I> = positions
            .iter()
            .zip(&valid)
            .map(|(&idx, &v)| {
                if v {
                    I::try_from(idx).unwrap()
                } else {
                    garbage
                }
            })
            .collect();

        let validity_bytes = pack(&valid, validity_offset.unwrap_or(0));
        let validity = validity_offset
            .map(|offset| BitBufferView::new_with_offset(&validity_bytes, len, offset));

        let taken = take_bits(bits, &indices, validity);
        assert_eq!(taken.len(), len);
        for (i, &idx) in positions.iter().enumerate() {
            if valid[i] {
                assert_eq!(taken.value(i), model[idx], "index {i}");
            }
        }
    }

    #[rstest]
    fn take_matches_model_u16(
        #[values(20, 300, 5000)] bits_len: usize,
        #[values(0, 5)] bits_offset: usize,
        #[values(None, Some(0), Some(3))] validity_offset: Option<usize>,
    ) {
        take_matches_model(bits_len, bits_offset, validity_offset, u16::MAX);
    }

    #[rstest]
    fn take_matches_model_u32(
        #[values(20, 300, 5000)] bits_len: usize,
        #[values(0, 5)] bits_offset: usize,
        #[values(None, Some(0), Some(3))] validity_offset: Option<usize>,
    ) {
        take_matches_model(bits_len, bits_offset, validity_offset, u32::MAX);
    }

    #[rstest]
    fn take_matches_model_u64(
        #[values(20, 300, 5000)] bits_len: usize,
        #[values(0, 5)] bits_offset: usize,
        #[values(None, Some(0), Some(3))] validity_offset: Option<usize>,
    ) {
        take_matches_model(bits_len, bits_offset, validity_offset, u64::MAX);
    }

    #[rstest]
    fn take_matches_model_i64(
        #[values(300, 5000)] bits_len: usize,
        #[values(None, Some(3))] validity_offset: Option<usize>,
    ) {
        take_matches_model(bits_len, 5, validity_offset, -1i64);
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

    #[hegel::test]
    #[cfg_attr(miri, ignore)]
    fn takes_from_small_bitmaps(tc: TestCase) {
        // u8 indices address at most 256 bits, u16 indices reach the 4096 bits of a small bitmap.
        let narrow = tc.draw(gs::booleans());
        let max_len = if narrow { 600 } else { 4096 };
        let len = tc.draw(gs::integers::<usize>().min_value(1).max_value(max_len));
        let offset = tc.draw(gs::integers::<usize>().max_value(63));
        let bits: Vec<bool> = tc.draw(
            gs::vecs(gs::booleans())
                .min_size(offset + len)
                .max_size(offset + len),
        );
        let buf = BitBuffer::from(bits.clone());
        let view = buf.as_view().slice(offset..);

        let bound = if narrow { len.min(256) } else { len };
        let indices: Vec<usize> = tc.draw(
            gs::vecs(gs::integers::<usize>().max_value(bound - 1))
                .min_size(1)
                .max_size(300),
        );

        let taken = if narrow {
            let typed: Vec<u8> = indices
                .iter()
                .map(|&idx| u8::try_from(idx).unwrap())
                .collect();
            take_bits(view, &typed, None)
        } else {
            let typed: Vec<u16> = indices
                .iter()
                .map(|&idx| u16::try_from(idx).unwrap())
                .collect();
            take_bits(view, &typed, None)
        };

        assert_eq!(taken.len(), indices.len());
        for (i, &idx) in indices.iter().enumerate() {
            assert_eq!(taken.value(i), bits[offset + idx], "index {idx} at {i}");
        }
    }

    #[rstest]
    #[case(300)]
    #[case(5000)]
    fn gathers_i64(#[case] bits_len: usize) {
        gathers::<i64>(bits_len, [i64::MAX, -1]);
    }

    #[rstest]
    #[case(300)]
    #[case(5000)]
    fn gathers_u16(#[case] bits_len: usize) {
        gathers::<u16>(bits_len, [u16::MAX, u16::MAX - 1]);
    }

    #[rstest]
    #[case(300)]
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

    #[rstest]
    fn single_differing_bit(
        #[values(0, 63, 64, 150, 199)] pos: usize,
        #[values(0, 5)] offset: usize,
        #[values(true, false)] set: bool,
    ) {
        let len = 200;
        let model: Vec<bool> = (0..len).map(|i| (i == pos) == set).collect();
        let bytes = pack(&model, offset);
        let bits = BitBufferView::new_with_offset(&bytes, len, offset);
        let indices: Vec<usize> = (0..len).rev().chain(0..len).collect();

        let taken = take_bits(bits, &indices, None);
        for (i, &idx) in indices.iter().enumerate() {
            assert_eq!(taken.value(i), model[idx], "index {i}");
        }
    }

    #[rstest]
    #[case(0, 0, None)]
    #[case(1, 0, None)]
    #[case(64, 0, None)]
    #[case(130, 0, None)]
    #[case(130, 3, None)]
    #[case(10, 0, Some(9))]
    #[case(130, 0, Some(0))]
    #[case(130, 0, Some(63))]
    #[case(130, 0, Some(64))]
    #[case(130, 0, Some(127))]
    #[case(130, 0, Some(128))]
    #[case(130, 0, Some(129))]
    #[case(130, 3, Some(70))]
    #[case(130, 3, Some(129))]
    fn first_unset_matches_model(
        #[case] len: usize,
        #[case] offset: usize,
        #[case] unset: Option<usize>,
    ) {
        let model: Vec<bool> = (0..len).map(|i| Some(i) != unset).collect();
        let bytes = pack(&model, offset);
        let bits = BitBufferView::new_with_offset(&bytes, len, offset);
        assert_eq!(first_unset(bits), unset);
    }

    // A single index only takes the all-set/all-unset shortcut for up to 64 bits, so these
    // lengths reach the bounds check.
    #[rstest]
    #[case(BitBuffer::new_set(65))]
    #[case(BitBuffer::new_unset(65))]
    #[case(BitBuffer::new_set(1000))]
    #[case(BitBuffer::new_unset(1000))]
    #[should_panic(expected = "take index out of bounds")]
    fn index_equal_to_len_panics(#[case] bits: BitBuffer) {
        let _taken = take_bits(bits.as_view(), &[bits.len()], None);
    }

    // u64 indices skip the AVX2 kernel, so this covers both scalar paths with validity.
    #[rstest]
    #[case::collected_group(300, 5)]
    #[case::collected_tail(300, 401)]
    #[case::unchecked_group(5000, 5)]
    #[case::unchecked_tail(5000, 401)]
    #[should_panic(expected = "take index out of bounds")]
    fn valid_index_equal_to_len_panics(#[case] bits_len: usize, #[case] pos: usize) {
        let model = random_model(bits_len);
        let bytes = pack(&model, 0);
        let bits = BitBufferView::new(&bytes, bits_len);
        let mut indices: Vec<u64> = random_indices(403, bits_len)
            .into_iter()
            .map(|idx| idx as u64)
            .collect();
        indices[pos] = bits_len as u64;
        let validity = BitBuffer::new_set(403);
        let _taken = take_bits(bits, &indices, Some(validity.as_view()));
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
    fn clears_bits_after_len() {
        // Bit 0 is set, so the padded indices of the tail chunk look up a set bit.
        let bits = BitBuffer::collect_bool(100, |i| i % 3 == 0);

        let taken = take_bits(bits.as_view(), &[1u8; 65], None);
        assert_eq!(taken.inner().as_slice(), &[0u8; 9]);

        let taken = take_bits(bits.as_view(), &[1u16; 65], None);
        assert_eq!(taken.inner().as_slice(), &[0u8; 9]);
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

    #[test]
    #[should_panic(expected = "take index out of bounds")]
    fn small_bitmap_out_of_bounds_in_tail() {
        let bits = BitBuffer::collect_bool(100, |i| i % 3 == 0);
        let mut indices = [0u16; 9];
        indices[8] = 120;
        let validity = BitBuffer::new_set(9);
        let _taken = take_bits(bits.as_view(), &indices, Some(validity.as_view()));
    }
}
