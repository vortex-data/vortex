// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_buffer::BitBuffer;
use vortex_buffer::BitChunkIterator;
use vortex_buffer::BufferMut;

use crate::Mask;
use crate::MaskValues;

trait DepositBits {
    fn deposit_bits(source: u64, mask: u64, mask_count: usize) -> u64;
}

struct PortableDeposit;

impl DepositBits for PortableDeposit {
    #[inline]
    fn deposit_bits(source: u64, mask: u64, mask_count: usize) -> u64 {
        if mask_count >= 16 && count_ones(source) * 8 < mask_count {
            return deposit_sparse_source(source, mask);
        }

        deposit_by_mask(source, mask)
    }
}

#[inline]
fn deposit_by_mask(mut source: u64, mut mask: u64) -> u64 {
    let mut result = 0u64;
    while mask != 0 {
        let bit = mask & mask.wrapping_neg();
        if source & 1 != 0 {
            result |= bit;
        }
        source >>= 1;
        mask &= mask - 1;
    }
    result
}

#[inline]
fn deposit_sparse_source(mut source: u64, mask: u64) -> u64 {
    let mut result = 0u64;
    while source != 0 {
        result |= select_set_bit(mask, trailing_zeros(source));
        source &= source - 1;
    }
    result
}

#[inline]
fn select_set_bit(word: u64, mut rank: usize) -> u64 {
    debug_assert!(rank < count_ones(word));
    let mut bit_offset = 0usize;
    for byte in word.to_le_bytes() {
        let count = count_ones_byte(byte);
        if rank < count {
            let mut bits = byte;
            for _ in 0..rank {
                bits &= bits - 1;
            }

            return 1u64 << (bit_offset + trailing_zeros_byte(bits));
        }

        rank -= count;
        bit_offset += 8;
    }

    0
}

#[inline]
fn count_ones_byte(value: u8) -> usize {
    value.count_ones() as usize
}

#[inline]
fn trailing_zeros(value: u64) -> usize {
    value.trailing_zeros() as usize
}

#[inline]
fn trailing_zeros_byte(value: u8) -> usize {
    value.trailing_zeros() as usize
}

#[cfg(target_arch = "x86_64")]
struct Bmi2Deposit;

#[cfg(target_arch = "x86_64")]
impl DepositBits for Bmi2Deposit {
    #[inline]
    fn deposit_bits(source: u64, mask: u64, _mask_count: usize) -> u64 {
        // SAFETY: callers only instantiate this implementation after checking BMI2 support.
        unsafe { pdep_bmi2(source, mask) }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2")]
unsafe fn pdep_bmi2(source: u64, mask: u64) -> u64 {
    core::arch::x86_64::_pdep_u64(source, mask)
}

struct RankBitReader<'a> {
    chunks: BitChunkIterator<'a>,
    remainder: u64,
    current: u64,
    bit_offset: usize,
    loaded_remainder: bool,
}

impl<'a> RankBitReader<'a> {
    fn new(buffer: &'a BitBuffer) -> Self {
        let chunks = buffer.chunks();
        let remainder = chunks.remainder_bits();
        let mut iter = chunks.iter();
        let Some(current) = iter.next() else {
            return Self {
                chunks: iter,
                remainder,
                current: remainder,
                bit_offset: 0,
                loaded_remainder: true,
            };
        };

        Self {
            chunks: iter,
            remainder,
            current,
            bit_offset: 0,
            loaded_remainder: false,
        }
    }

    #[inline]
    fn read(&mut self, bit_count: usize) -> u64 {
        debug_assert!(bit_count <= 64);
        if bit_count == 0 {
            return 0;
        }

        let available = 64 - self.bit_offset;
        if bit_count <= available {
            let result = (self.current >> self.bit_offset) & low_bits(bit_count);
            self.bit_offset += bit_count;
            if self.bit_offset == 64 {
                self.advance();
            }
            return result;
        }

        let low = self.current >> self.bit_offset;
        self.advance();

        let high_bit_count = bit_count - available;
        let high = self.current & low_bits(high_bit_count);
        self.bit_offset = high_bit_count;
        low | (high << available)
    }

    #[inline]
    fn advance(&mut self) {
        if let Some(current) = self.chunks.next() {
            self.current = current;
            self.loaded_remainder = false;
        } else if !self.loaded_remainder {
            self.current = self.remainder;
            self.loaded_remainder = true;
        } else {
            self.current = 0;
        }
        self.bit_offset = 0;
    }
}

#[inline]
fn low_bits(bit_count: usize) -> u64 {
    debug_assert!(bit_count <= 64);
    if bit_count == 64 {
        u64::MAX
    } else {
        (1u64 << bit_count) - 1
    }
}

#[inline]
fn count_ones(value: u64) -> usize {
    value.count_ones() as usize
}

#[inline]
fn mask_from_buffer(buffer: BitBuffer, true_count: usize) -> Mask {
    let len = buffer.len();
    if true_count == 0 {
        return Mask::new_false(len);
    }
    if true_count == len {
        return Mask::new_true(len);
    }

    Mask::Values(Arc::new(MaskValues {
        buffer,
        indices: Default::default(),
        slices: Default::default(),
        true_count,
        density: true_count as f64 / len as f64,
    }))
}

#[inline]
fn push_result_chunk<D: DepositBits>(
    result: &mut BufferMut<u64>,
    self_chunk: u64,
    self_count: usize,
    rank_bits: u64,
) {
    let chunk = if rank_bits == 0 {
        0
    } else if self_chunk == u64::MAX {
        rank_bits
    } else {
        D::deposit_bits(rank_bits, self_chunk, self_count)
    };

    // SAFETY: callers allocate enough capacity for every output chunk.
    unsafe { result.push_unchecked(chunk) };
}

fn intersect_bit_buffers<D: DepositBits>(
    self_buffer: &BitBuffer,
    mask_buffer: &BitBuffer,
    true_count: usize,
) -> Mask {
    let len = self_buffer.len();
    let mut result = BufferMut::with_capacity(len.div_ceil(64));
    let mut reader = RankBitReader::new(mask_buffer);
    let self_chunks = self_buffer.chunks();

    for self_chunk in self_chunks.iter() {
        let self_count = count_ones(self_chunk);
        let rank_bits = reader.read(self_count);
        push_result_chunk::<D>(&mut result, self_chunk, self_count, rank_bits);
    }

    if self_chunks.remainder_len() != 0 {
        let self_chunk = self_chunks.remainder_bits();
        let self_count = count_ones(self_chunk);
        let rank_bits = reader.read(self_count);
        push_result_chunk::<D>(&mut result, self_chunk, self_count, rank_bits);
    }

    mask_from_buffer(
        BitBuffer::new(result.freeze().into_byte_buffer(), len),
        true_count,
    )
}

fn intersect_bit_buffer_by_rank_indices<D: DepositBits>(
    self_buffer: &BitBuffer,
    mask_indices: &[usize],
) -> Mask {
    let len = self_buffer.len();
    let mut result = BufferMut::with_capacity(len.div_ceil(64));
    let self_chunks = self_buffer.chunks();
    let mut rank_base = 0usize;
    let mut rank_idx = 0usize;

    for self_chunk in self_chunks.iter() {
        let self_count = count_ones(self_chunk);
        let next_rank_base = rank_base + self_count;
        let rank_bits = rank_bits_for_chunk(mask_indices, &mut rank_idx, rank_base, next_rank_base);
        push_result_chunk::<D>(&mut result, self_chunk, self_count, rank_bits);
        rank_base = next_rank_base;
    }

    if self_chunks.remainder_len() != 0 {
        let self_chunk = self_chunks.remainder_bits();
        let self_count = count_ones(self_chunk);
        let next_rank_base = rank_base + self_count;
        let rank_bits = rank_bits_for_chunk(mask_indices, &mut rank_idx, rank_base, next_rank_base);
        push_result_chunk::<D>(&mut result, self_chunk, self_count, rank_bits);
    }

    debug_assert_eq!(rank_idx, mask_indices.len());

    mask_from_buffer(
        BitBuffer::new(result.freeze().into_byte_buffer(), len),
        mask_indices.len(),
    )
}

fn intersect_bit_buffer_by_rank_index_iter<D: DepositBits>(
    self_buffer: &BitBuffer,
    mask_indices: impl Iterator<Item = usize>,
    true_count: usize,
) -> Mask {
    let len = self_buffer.len();
    let mut result = BufferMut::with_capacity(len.div_ceil(64));
    let self_chunks = self_buffer.chunks();
    let mut rank_base = 0usize;
    let mut mask_indices = mask_indices.peekable();

    for self_chunk in self_chunks.iter() {
        let self_count = count_ones(self_chunk);
        let next_rank_base = rank_base + self_count;
        let rank_bits = rank_bits_for_chunk_iter(&mut mask_indices, rank_base, next_rank_base);
        push_result_chunk::<D>(&mut result, self_chunk, self_count, rank_bits);
        rank_base = next_rank_base;
    }

    if self_chunks.remainder_len() != 0 {
        let self_chunk = self_chunks.remainder_bits();
        let self_count = count_ones(self_chunk);
        let next_rank_base = rank_base + self_count;
        let rank_bits = rank_bits_for_chunk_iter(&mut mask_indices, rank_base, next_rank_base);
        push_result_chunk::<D>(&mut result, self_chunk, self_count, rank_bits);
    }

    let exhausted_mask_indices = mask_indices.next().is_none();
    debug_assert!(exhausted_mask_indices);

    mask_from_buffer(
        BitBuffer::new(result.freeze().into_byte_buffer(), len),
        true_count,
    )
}

#[inline]
fn rank_bits_for_chunk(
    mask_indices: &[usize],
    rank_idx: &mut usize,
    rank_base: usize,
    next_rank_base: usize,
) -> u64 {
    let mut rank_bits = 0u64;
    while let Some(&rank) = mask_indices.get(*rank_idx) {
        if rank >= next_rank_base {
            break;
        }
        rank_bits |= 1u64 << (rank - rank_base);
        *rank_idx += 1;
    }
    rank_bits
}

fn intersect_by_rank_indices(len: usize, self_indices: &[usize], mask_indices: &[usize]) -> Mask {
    Mask::from_indices(
        len,
        mask_indices
            .iter()
            .map(|idx| {
                // SAFETY: mask indices are ranks into self_indices, because
                // mask.len() == self.true_count() == self_indices.len().
                unsafe { *self_indices.get_unchecked(*idx) }
            })
            .collect(),
    )
}

#[inline]
fn rank_bits_for_chunk_iter(
    mask_indices: &mut std::iter::Peekable<impl Iterator<Item = usize>>,
    rank_base: usize,
    next_rank_base: usize,
) -> u64 {
    let mut rank_bits = 0u64;
    while let Some(&rank) = mask_indices.peek() {
        if rank >= next_rank_base {
            break;
        }
        rank_bits |= 1u64 << (rank - rank_base);
        mask_indices.next();
    }
    rank_bits
}

#[inline]
fn intersect_bit_buffers_dispatch(
    self_buffer: &BitBuffer,
    mask_buffer: &BitBuffer,
    true_count: usize,
) -> Mask {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("bmi2") {
        return intersect_bit_buffers::<Bmi2Deposit>(self_buffer, mask_buffer, true_count);
    }

    intersect_bit_buffers::<PortableDeposit>(self_buffer, mask_buffer, true_count)
}

#[inline]
fn intersect_rank_indices_dispatch(self_buffer: &BitBuffer, mask_indices: &[usize]) -> Mask {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("bmi2") {
        return intersect_bit_buffer_by_rank_indices::<Bmi2Deposit>(self_buffer, mask_indices);
    }

    intersect_bit_buffer_by_rank_indices::<PortableDeposit>(self_buffer, mask_indices)
}

#[inline]
fn intersect_rank_index_iter_dispatch(
    self_buffer: &BitBuffer,
    mask_indices: impl Iterator<Item = usize>,
    true_count: usize,
) -> Mask {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("bmi2") {
        return intersect_bit_buffer_by_rank_index_iter::<Bmi2Deposit>(
            self_buffer,
            mask_indices,
            true_count,
        );
    }

    intersect_bit_buffer_by_rank_index_iter::<PortableDeposit>(
        self_buffer,
        mask_indices,
        true_count,
    )
}

impl Mask {
    /// Take the intersection of the `mask` with the set of true values in `self`.
    ///
    /// The hot path keeps bit-buffer-backed masks as bit buffers. It scans the set bits of `self`
    /// by rank and deposits selected rank bits into their original positions.
    ///
    /// # Examples
    ///
    /// Keep the third and fifth set values from mask `m1`:
    /// ```
    /// use vortex_mask::Mask;
    ///
    /// let m1 = Mask::from_iter([true, false, false, true, true, true, false, true]);
    /// let m2 = Mask::from_iter([false, false, true, false, true]);
    /// assert_eq!(
    ///     m1.intersect_by_rank(&m2),
    ///     Mask::from_iter([false, false, false, false, true, false, false, true])
    /// );
    /// ```
    pub fn intersect_by_rank(&self, mask: &Mask) -> Mask {
        assert_eq!(self.true_count(), mask.len());

        match (self, mask) {
            (Self::AllTrue(_), _) => mask.clone(),
            (_, Self::AllTrue(_)) => self.clone(),
            (Self::AllFalse(_), _) | (_, Self::AllFalse(_)) => Self::new_false(self.len()),
            (Self::Values(self_values), Self::Values(mask_values)) => {
                let self_is_very_sparse = self_values.true_count() < self.len().div_ceil(64);

                if let Some(mask_indices) = mask_values.indices.get() {
                    if let Some(self_indices) = self_values.indices.get()
                        && mask_indices.len() < self.len().div_ceil(64)
                    {
                        return intersect_by_rank_indices(self.len(), self_indices, mask_indices);
                    }

                    if self_is_very_sparse {
                        return intersect_by_rank_indices(
                            self.len(),
                            self_values.indices(),
                            mask_indices,
                        );
                    }

                    if mask_indices.len().saturating_mul(4) > mask.len() {
                        return intersect_bit_buffers_dispatch(
                            self_values.bit_buffer(),
                            mask_values.bit_buffer(),
                            mask_values.true_count(),
                        );
                    }

                    return intersect_rank_indices_dispatch(self_values.bit_buffer(), mask_indices);
                }

                if self_is_very_sparse {
                    return intersect_by_rank_indices(
                        self.len(),
                        self_values.indices(),
                        mask_values.indices(),
                    );
                }

                if mask_values.true_count().saturating_mul(32) < mask.len() {
                    return intersect_rank_index_iter_dispatch(
                        self_values.bit_buffer(),
                        mask_values.bit_buffer().set_indices(),
                        mask_values.true_count(),
                    );
                }

                intersect_bit_buffers_dispatch(
                    self_values.bit_buffer(),
                    mask_values.bit_buffer(),
                    mask_values.true_count(),
                )
            }
        }
    }
}

#[cfg(test)]
mod test {
    use rstest::rstest;
    use vortex_buffer::BitBuffer;

    use crate::Mask;

    #[test]
    fn mask_bitand_all_as_bit_and() {
        let this = Mask::from_buffer(BitBuffer::from_iter(vec![true, true, true, true, true]));
        let mask = Mask::from_buffer(BitBuffer::from_iter(vec![false, true, false, true, true]));
        assert_eq!(
            this.intersect_by_rank(&mask),
            Mask::from_indices(5, vec![1, 3, 4])
        );
    }

    #[test]
    fn mask_bitand_all_true() {
        let this = Mask::from_buffer(BitBuffer::from_iter(vec![false, false, true, true, true]));
        let mask = Mask::from_buffer(BitBuffer::from_iter(vec![true, true, true]));
        assert_eq!(
            this.intersect_by_rank(&mask),
            Mask::from_indices(5, vec![2, 3, 4])
        );
    }

    #[test]
    fn mask_bitand_true() {
        let this = Mask::from_buffer(BitBuffer::from_iter(vec![true, false, false, true, true]));
        let mask = Mask::from_buffer(BitBuffer::from_iter(vec![true, false, true]));
        assert_eq!(
            this.intersect_by_rank(&mask),
            Mask::from_indices(5, vec![0, 4])
        );
    }

    #[test]
    fn mask_bitand_false() {
        let this = Mask::from_buffer(BitBuffer::from_iter(vec![true, false, false, true, true]));
        let mask = Mask::from_buffer(BitBuffer::from_iter(vec![false, false, false]));
        assert_eq!(this.intersect_by_rank(&mask), Mask::from_indices(5, vec![]));
    }

    #[test]
    fn mask_intersect_by_rank_all_false() {
        let this = Mask::AllFalse(10);
        let mask = Mask::AllFalse(0);
        assert_eq!(this.intersect_by_rank(&mask), Mask::AllFalse(10));
    }

    #[rstest]
    #[case::all_true_with_all_true(
        Mask::new_true(5),
        Mask::new_true(5),
        vec![0, 1, 2, 3, 4]
    )]
    #[case::all_true_with_all_false(
        Mask::new_true(5),
        Mask::new_false(5),
        vec![]
    )]
    #[case::all_false_with_any(
        Mask::new_false(10),
        Mask::new_true(0),
        vec![]
    )]
    #[case::indices_with_all_true(
        Mask::from_indices(10, vec![2, 5, 7, 9]),
        Mask::new_true(4),
        vec![2, 5, 7, 9]
    )]
    #[case::indices_with_all_false(
        Mask::from_indices(10, vec![2, 5, 7, 9]),
        Mask::new_false(4),
        vec![]
    )]
    fn test_intersect_by_rank_special_cases(
        #[case] base_mask: Mask,
        #[case] rank_mask: Mask,
        #[case] expected_indices: Vec<usize>,
    ) {
        let result = base_mask.intersect_by_rank(&rank_mask);

        match result.indices() {
            crate::AllOr::All => assert_eq!(expected_indices.len(), result.len()),
            crate::AllOr::None => assert!(expected_indices.is_empty()),
            crate::AllOr::Some(indices) => assert_eq!(indices, &expected_indices[..]),
        }
    }

    #[test]
    fn test_intersect_by_rank_example() {
        // Example from the documentation
        let m1 = Mask::from_iter([true, false, false, true, true, true, false, true]);
        let m2 = Mask::from_iter([false, false, true, false, true]);
        let result = m1.intersect_by_rank(&m2);
        let expected = Mask::from_iter([false, false, false, false, true, false, false, true]);
        assert_eq!(result, expected);
    }

    #[test]
    #[should_panic]
    fn test_intersect_by_rank_wrong_length() {
        let m1 = Mask::from_indices(10, vec![2, 5, 7]); // 3 true values
        let m2 = Mask::new_true(5); // 5 true values - doesn't match
        m1.intersect_by_rank(&m2);
    }

    #[rstest]
    #[case::single_element(
        vec![3],
        vec![true],
        vec![3]
    )]
    #[case::single_element_masked(
        vec![3],
        vec![false],
        vec![]
    )]
    #[case::alternating(
        vec![0, 2, 4, 6, 8],
        vec![true, false, true, false, true],
        vec![0, 4, 8]
    )]
    #[case::consecutive(
        vec![5, 6, 7, 8, 9],
        vec![false, true, true, true, false],
        vec![6, 7, 8]
    )]
    fn test_intersect_by_rank_patterns(
        #[case] base_indices: Vec<usize>,
        #[case] rank_pattern: Vec<bool>,
        #[case] expected_indices: Vec<usize>,
    ) {
        let base = Mask::from_indices(10, base_indices);
        let rank = Mask::from_iter(rank_pattern);
        let result = base.intersect_by_rank(&rank);

        match result.indices() {
            crate::AllOr::Some(indices) => assert_eq!(indices, &expected_indices[..]),
            crate::AllOr::None => assert!(expected_indices.is_empty()),
            _ => panic!("Unexpected result"),
        }
    }

    #[rstest]
    #[case::short(37, 0, 0)]
    #[case::base_offset(257, 5, 0)]
    #[case::rank_offset(257, 0, 3)]
    #[case::both_offsets(513, 6, 5)]
    fn test_intersect_by_rank_bitbuffer_paths_with_offsets(
        #[case] base_len: usize,
        #[case] base_offset: usize,
        #[case] rank_offset: usize,
    ) {
        let base_source: Vec<bool> = (0..base_len + base_offset + 16)
            .map(|i| (i % 3 == 0) ^ (i % 11 == 0) ^ (i % 17 == 0))
            .collect();
        let base_bits = base_source[base_offset..base_offset + base_len].to_vec();
        let base = Mask::from_buffer(
            BitBuffer::from(base_source).slice(base_offset..base_offset + base_len),
        );

        let rank_len = base.true_count();
        let rank_bits: Vec<bool> = (0..rank_len)
            .map(|i| (i % 5 == 0) || (i % 13 == 3))
            .collect();
        let mut rank_source = vec![false; rank_offset];
        rank_source.extend(rank_bits.iter().copied());
        rank_source.extend([true, false, true, false, true, false, true, false]);

        let rank_from_buffer = Mask::from_buffer(
            BitBuffer::from(rank_source).slice(rank_offset..rank_offset + rank_len),
        );
        let rank_indices = rank_bits
            .iter()
            .enumerate()
            .filter_map(|(idx, &value)| value.then_some(idx))
            .collect::<Vec<_>>();
        let rank_from_indices = Mask::from_indices(rank_len, rank_indices);

        let expected = expected_intersect_by_rank(&base_bits, &rank_bits);

        assert_eq!(base.intersect_by_rank(&rank_from_buffer), expected);
        assert_eq!(base.intersect_by_rank(&rank_from_indices), expected);
    }

    fn expected_intersect_by_rank(base_bits: &[bool], rank_bits: &[bool]) -> Mask {
        let mut rank = 0usize;
        Mask::from_iter(base_bits.iter().map(|&is_set| {
            if is_set {
                let keep = rank_bits[rank];
                rank += 1;
                keep
            } else {
                false
            }
        }))
    }
}
