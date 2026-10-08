// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::AsPrimitive;
use num_traits::Zero;
use vortex_error::VortexExpect;

use crate::BitBuffer;
use crate::BitBufferView;
use crate::bit::get_bit;
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
            let bools: Vec<bool> = bits.iter().collect();
            BitBuffer::collect_bool_multiversioned(indices.len(), |i| unsafe {
                // SAFETY: we're already iterating over every index, a bound
                // is excessive
                let idx = indices.get_unchecked(i).as_();
                // SAFETY: we've asserted index is in bounds before
                *bools.get_unchecked(idx)
            })
        } else {
            let ptr = bits.inner().as_ptr();
            let offset = bits.offset();
            BitBuffer::collect_bool(indices.len(), |i| unsafe {
                // SAFETY: we're already iterating over every index, a bound
                // is excessive
                let idx = indices.get_unchecked(i).as_();
                // SAFETY: we've asserted index is in bounds before
                get_bit_unchecked(ptr, offset + idx)
            })
        };
    };

    let ptr = validity.inner().as_ptr();
    if bits.len() <= COLLECT_TO_VEC {
        let offset = validity.offset();
        let bools: Vec<bool> = bits.iter().collect();
        return BitBuffer::collect_bool(indices.len(), |i| {
            // SAFETY: we're already iterating over every index, a bound
            // is excessive
            let idx = unsafe { indices.get_unchecked(i).as_() };
            // SAFETY: we're verified validity has same size as indices before
            let mask_idx: usize = unsafe { get_bit_unchecked(ptr, offset + i) }.as_();
            bools[idx & mask_idx.wrapping_neg()]
        });
    }

    let validity_offset = validity.offset();
    let buf = bits.inner();
    let bits_offset = bits.offset();
    BitBuffer::collect_bool(indices.len(), |i| {
        // SAFETY: we're already iterating over every index, a bound
        // is excessive
        let idx = unsafe { indices.get_unchecked(i).as_() };
        // SAFETY: we're verified validity has same size as indices before
        let mask_idx: usize = unsafe { get_bit_unchecked(ptr, validity_offset + i) }.as_();
        let masked_idx = idx & mask_idx.wrapping_neg();
        get_bit(buf, bits_offset + masked_idx)
    })
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

#[cfg(test)]
mod tests {
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

    #[rstest]
    #[case(300)]
    #[case(5000)]
    fn gathers(#[case] bits_len: usize) {
        let bits = random_bits(bits_len + 11);
        let view = bits.as_view().slice(11..);
        let len = 400;
        let indices: Vec<i64> = random_indices(len, bits_len)
            .into_iter()
            .map(|idx| i64::try_from(idx).unwrap())
            .collect();

        let taken = take_bits(view, &indices, None);
        assert_eq!(taken.len(), len);
        for (i, &idx) in indices.iter().enumerate() {
            assert_eq!(
                taken.value(i),
                bits.value(usize::try_from(idx).unwrap() + 11)
            );
        }

        let validity = BitBuffer::collect_bool(len, |i| i % 3 != 0);
        let with_garbage: Vec<i64> = indices
            .iter()
            .enumerate()
            .map(|(i, &idx)| match i % 6 {
                0 => i64::MAX,
                3 => -1,
                _ => idx,
            })
            .collect();

        let taken = take_bits(view, &with_garbage, Some(validity.as_view()));
        assert_eq!(taken.len(), len);
        for (i, &idx) in with_garbage.iter().enumerate() {
            if validity.value(i) {
                assert_eq!(
                    taken.value(i),
                    bits.value(usize::try_from(idx).unwrap() + 11)
                );
            }
        }
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
}
