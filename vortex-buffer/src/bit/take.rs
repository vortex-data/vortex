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

        return if bits.len() <= COLLECT_TO_VEC {
            #[cfg(target_arch = "x86_64")]
            {
                // Gathering the bits beats a byte per bit, but its 32-bit loads need the buffer
                // padded, which a small copy, at most 520 bytes, provides.
                let mut padded = [0u8; COLLECT_TO_VEC / 8 + 8];
                for (word, chunk) in bits
                    .chunks()
                    .iter_padded()
                    .zip(padded.as_chunks_mut::<8>().0)
                {
                    *chunk = word.to_le_bytes();
                }
                let padded = BitBufferView::new(&padded, bits.len());
                if let Some(taken) = avx2::take(padded, indices, None) {
                    return taken;
                }
            }
            let bools: Vec<bool> = bits.iter().collect();
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
    if let Some(taken) = avx2::take(bits, indices, Some(validity)) {
        return taken;
    }

    if bits.len() <= COLLECT_TO_VEC {
        let bools: Vec<bool> = bits.iter().collect();
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

#[cfg(test)]
mod tests {
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
