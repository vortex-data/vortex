// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BitBuffer;
use vortex_buffer::BufferMut;

use crate::AllOr;
use crate::Mask;

/// Portable implementation of PDEP (parallel bit deposit) using a 64K LUT.
///
/// Processes the mask one byte at a time: for each mask byte, looks up the
/// deposited bits in a precomputed table. 8 iterations per u64 chunk,
/// ~2x faster than the bit-serial fallback and portable across all architectures
/// (including Apple M-series which lacks BMI2).
#[inline]
fn pdep_lut(mut source: u64, mask: u64) -> u64 {
    let mut result = 0u64;
    for byte_idx in 0..8u32 {
        let mask_byte = ((mask >> (byte_idx * 8)) & 0xFF) as usize;
        if mask_byte == 0 {
            continue;
        }
        let count = PDEP_LUT.counts[mask_byte];
        let src_byte = (source & 0xFF) as usize;
        result |= (PDEP_LUT.table[mask_byte][src_byte] as u64) << (byte_idx * 8);
        source >>= count;
    }
    result
}

/// Precomputed LUT for byte-level PDEP: `table[mask_byte][source_byte]` gives the
/// deposited result, `counts[mask_byte]` gives the popcount of that mask byte.
struct PdepLut {
    table: [[u8; 256]; 256],
    counts: [u8; 256],
}

impl PdepLut {
    #[allow(clippy::cast_possible_truncation)] // values are bounded to 0..256
    const fn new() -> Self {
        let mut table = [[0u8; 256]; 256];
        let mut counts = [0u8; 256];
        let mut mask_byte = 0usize;
        while mask_byte < 256 {
            let mut m = mask_byte as u8;
            let mut c = 0u8;
            while m != 0 {
                c += 1;
                m &= m.wrapping_sub(1);
            }
            counts[mask_byte] = c;
            let mut source_val = 0usize;
            while source_val < 256 {
                let mut src = source_val as u8;
                let mut m = mask_byte as u8;
                let mut res = 0u8;
                while m != 0 {
                    let lowest = m & m.wrapping_neg();
                    if src & 1 != 0 {
                        res |= lowest;
                    }
                    src >>= 1;
                    m &= m.wrapping_sub(1);
                }
                table[mask_byte][source_val] = res;
                source_val += 1;
            }
            mask_byte += 1;
        }
        PdepLut { table, counts }
    }
}

static PDEP_LUT: PdepLut = PdepLut::new();

/// Extract 64 bits starting at bit position `bit_pos` from a flat mask array.
///
/// The flat array must contain full chunks + remainder + sentinel (at least one
/// element past the last valid chunk), so `mask_flat[bit_pos >> 6 + 1]` is always valid.
#[inline]
#[allow(clippy::cast_possible_truncation)] // bit_pos & 63 always fits in u32
fn extract_bits_portable(mask_flat: &[u64], bit_pos: usize) -> u64 {
    let chunk_idx = bit_pos >> 6;
    let shift = (bit_pos & 63) as u32;
    let lo = unsafe { *mask_flat.get_unchecked(chunk_idx) };
    let hi = unsafe { *mask_flat.get_unchecked(chunk_idx + 1) };
    // Branchless: when shift == 0, mask is 0 so hi contribution is zeroed out.
    // (64 - shift) & 63 maps shift=0 → 0, but `& mask` kills the unwanted hi << 0.
    let mask = (shift != 0) as u64 * u64::MAX;
    (lo >> shift) | ((hi << ((64u32.wrapping_sub(shift)) & 63)) & mask)
}

/// Extract 64 bits from two adjacent u64s using x86 SHRD instruction.
///
/// `SHRD(lo, hi, count)` computes `(hi:lo >> count)[0:63]`.
/// When `count == 0`, SHRD is a no-op and returns `lo` — eliminating
/// the branch that a shift-based approach would need.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn extract_shrd(lo: u64, hi: u64, count: u8) -> u64 {
    let result: u64;
    unsafe {
        core::arch::asm!(
            "shrd {lo}, {hi}, cl",
            lo = inout(reg) lo => result,
            hi = in(reg) hi,
            in("cl") count,
            options(pure, nomem, nostack),
        );
    }
    result
}

/// Get the full u64 chunks of a `BitBuffer` as a zero-copy `&[u64]` slice,
/// plus the remainder bits, when the buffer has offset 0 and is u64-aligned.
///
/// Returns `None` if the buffer has a non-zero bit offset or is not u64-aligned,
/// in which case the caller must fall back to the `BitChunks` iterator.
fn bit_buffer_raw_u64s(buf: &BitBuffer) -> Option<(&[u64], u64)> {
    if buf.offset() != 0 {
        return None;
    }
    let bytes = buf.inner().as_slice();
    if bytes.as_ptr().align_offset(align_of::<u64>()) != 0 {
        return None;
    }
    let num_full = buf.len() / 64;
    let full_chunks = unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<u64>(), num_full) };
    // Read remainder bytes into a u64
    let remainder = if !buf.len().is_multiple_of(64) {
        let rem_bytes = &bytes[num_full * 8..];
        let mut val = 0u64;
        for (i, &b) in rem_bytes.iter().enumerate() {
            val |= (b as u64) << (i * 8);
        }
        // Mask to only the valid bits
        val & ((1u64 << (buf.len() % 64)) - 1)
    } else {
        0
    };
    Some((full_chunks, remainder))
}

/// Build a flat mask array: full chunks + remainder + sentinel.
///
/// Uses `extend_from_slice` (memcpy) when possible instead of per-element iteration.
/// The sentinel element ensures `mask_flat[chunk_idx + 1]` is always valid for
/// SHRD / shift-based bit extraction.
fn build_mask_flat(buf: &BitBuffer) -> Vec<u64> {
    let chunks = buf.chunks();
    let total = chunks.chunk_len() + 2; // full chunks + remainder + sentinel

    let mut flat = Vec::with_capacity(total);
    if let Some((raw_chunks, remainder)) = bit_buffer_raw_u64s(buf) {
        flat.extend_from_slice(raw_chunks);
        flat.push(remainder);
    } else {
        flat.extend(chunks.iter());
        flat.push(chunks.remainder_bits());
    }
    flat.push(0); // sentinel for SHRD reading chunk_idx+1
    flat
}

/// The entire dense chunk loop with BMI2 PDEP + hardware POPCNT enabled.
///
/// Uses a sliding-window cursor with SHRD for fully branchless bit extraction.
/// SHRD handles the `bit_offset == 0` case natively (no-op), so there are
/// no branches in the hot loop beyond the loop counter itself.
///
/// Accepts full u64 chunks + separate remainder to allow zero-copy passthrough
/// of the underlying `BitBuffer` data without collecting into a Vec.
///
/// By putting `target_feature` on the whole loop, the compiler can:
/// 1. Inline PDEP (no function call overhead)
/// 2. Use hardware POPCNT for count_ones() (1 instruction vs 15)
/// 3. Better register allocation across the loop body
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2,popcnt")]
#[allow(clippy::cast_possible_truncation)] // bit_offset is masked to 0..63, always fits in u8
unsafe fn dense_loop_bmi2(
    self_full_chunks: &[u64],
    self_remainder: u64,
    mask_flat: &[u64],
    len: usize,
) -> Mask {
    let has_remainder = !len.is_multiple_of(64);
    let num_out = self_full_chunks.len() + has_remainder as usize;
    let mask_ptr = mask_flat.as_ptr();

    let mut buffer: BufferMut<u64> = BufferMut::with_capacity(num_out);
    let mut bit_pos: usize = 0;

    // Main loop over full u64 chunks — operates on zero-copy &[u64] from BitBuffer.
    // Branchless: PDEP(anything, 0) == 0, SHRD handles bit_offset == 0 as no-op.
    // Uses a single `bit_pos` cursor instead of separate chunk_idx + bit_offset:
    //   chunk_idx = bit_pos >> 6, bit_offset = bit_pos & 63 (implicit via SHRD mod 64).
    for &self_chunk in self_full_chunks {
        let chunk_idx = bit_pos >> 6;
        let lo = unsafe { *mask_ptr.add(chunk_idx) };
        let hi = unsafe { *mask_ptr.add(chunk_idx + 1) };
        let rank_bits = unsafe { extract_shrd(lo, hi, bit_pos as u8) };

        let result_chunk = core::arch::x86_64::_pdep_u64(rank_bits, self_chunk);
        unsafe { buffer.push_unchecked(result_chunk) };

        bit_pos += self_chunk.count_ones() as usize;
    }

    // Handle remainder (partial chunk at the end)
    if has_remainder {
        let chunk_idx = bit_pos >> 6;
        let lo = unsafe { *mask_ptr.add(chunk_idx) };
        let hi = unsafe { *mask_ptr.add(chunk_idx + 1) };
        let rank_bits = unsafe { extract_shrd(lo, hi, bit_pos as u8) };
        let result_chunk = core::arch::x86_64::_pdep_u64(rank_bits, self_remainder);
        unsafe { buffer.push_unchecked(result_chunk) };
    }

    buffer.truncate(len.div_ceil(8));
    Mask::from_buffer(BitBuffer::new(buffer.freeze().into_byte_buffer(), len))
}

impl Mask {
    /// Indices-based implementation for sparse masks.
    ///
    /// O(mask.true_count) iterations with random access into self's index list.
    fn intersect_by_rank_sparse(&self, mask: &Mask) -> Mask {
        match (self.indices(), mask.indices()) {
            (AllOr::All, _) => mask.clone(),
            (_, AllOr::All) => self.clone(),
            (AllOr::None, _) | (_, AllOr::None) => Self::new_false(self.len()),

            (AllOr::Some(self_indices), AllOr::Some(mask_indices)) => {
                let len = self.len();
                if mask_indices.is_empty() {
                    return Self::new_false(len);
                }

                let num_chunks = len.div_ceil(64);
                let mut buffer: BufferMut<u64> = BufferMut::zeroed(num_chunks);
                let chunks = buffer.as_mut_slice();

                for &idx in mask_indices.iter() {
                    // SAFETY: mask_indices values are < mask.len() == self.true_count() == self_indices.len()
                    let bit_pos = unsafe { *self_indices.get_unchecked(idx) };
                    // SAFETY: bit_pos < self.len() and we allocated ceil(self.len()/64) chunks
                    unsafe {
                        *chunks.get_unchecked_mut(bit_pos / 64) |= 1u64 << (bit_pos % 64);
                    }
                }

                buffer.truncate(len.div_ceil(8));
                Self::from_buffer(BitBuffer::new(buffer.freeze().into_byte_buffer(), len))
            }
        }
    }

    /// Take the intersection of the `mask` with the set of true values in `self`.
    ///
    /// Uses adaptive dispatch: sparse masks use an indices-based approach, while dense
    /// masks use a chunk-based PDEP approach that processes 64 bits at a time.
    ///
    /// On x86_64 with BMI2, the dense path uses hardware PDEP for optimal performance.
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

        // Adaptive dispatch: use sparse (indices-based) when either mask is sparse.
        // Sparse is O(mask.true_count) with random access; chunk-based is O(self.len/64) sequential.
        let self_sparse = self.true_count().saturating_mul(20) <= self.len();
        let mask_sparse = mask.true_count().saturating_mul(20) <= mask.len();
        if self_sparse || mask_sparse {
            return self.intersect_by_rank_sparse(mask);
        }

        match (self.bit_buffer(), mask.bit_buffer()) {
            (AllOr::All, _) => mask.clone(),
            (_, AllOr::All) => self.clone(),
            (AllOr::None, _) | (_, AllOr::None) => Self::new_false(self.len()),

            (AllOr::Some(self_buffer), AllOr::Some(mask_buffer)) => {
                let len = self.len();

                // On x86_64 with BMI2+POPCNT, use the optimized dense loop
                // where PDEP is inlined and count_ones() uses hardware POPCNT.
                #[cfg(target_arch = "x86_64")]
                if std::arch::is_x86_feature_detected!("bmi2") {
                    // Build flat mask: full chunks + remainder + sentinel (for SHRD lookahead).
                    // Uses memcpy (extend_from_slice) when the buffer is u64-aligned.
                    let mask_flat = build_mask_flat(mask_buffer);

                    // For self: zero-copy &[u64] when aligned, otherwise collect.
                    // This avoids a full Vec allocation+copy in the common case.
                    let self_chunks = self_buffer.chunks();
                    return if let Some((raw_chunks, remainder)) = bit_buffer_raw_u64s(self_buffer) {
                        // SAFETY: We just verified BMI2 is available.
                        unsafe { dense_loop_bmi2(raw_chunks, remainder, &mask_flat, len) }
                    } else {
                        // Fallback: collect through iterator (handles non-zero offset)
                        let full: Vec<u64> = self_chunks.iter().collect();
                        let remainder = self_chunks.remainder_bits();
                        unsafe { dense_loop_bmi2(&full, remainder, &mask_flat, len) }
                    };
                }

                // Portable fallback: LUT PDEP + flat mask with sliding window.
                // Same structure as BMI2 path but with software PDEP.
                let mask_flat = build_mask_flat(mask_buffer);

                let self_chunks = self_buffer.chunks();
                let has_remainder = !len.is_multiple_of(64);
                let num_out = self_chunks.chunk_len() + has_remainder as usize;
                let mut buffer: BufferMut<u64> = BufferMut::with_capacity(num_out);
                let mut bit_pos = 0usize;

                for self_chunk in self_chunks.iter() {
                    let popcount = self_chunk.count_ones() as usize;
                    let result_chunk = if self_chunk == 0 {
                        0u64
                    } else {
                        let rank_bits = extract_bits_portable(&mask_flat, bit_pos);
                        pdep_lut(rank_bits, self_chunk)
                    };
                    unsafe { buffer.push_unchecked(result_chunk) };
                    bit_pos += popcount;
                }

                if has_remainder {
                    let self_chunk = self_chunks.remainder_bits();
                    let result_chunk = if self_chunk == 0 {
                        0u64
                    } else {
                        let rank_bits = extract_bits_portable(&mask_flat, bit_pos);
                        pdep_lut(rank_bits, self_chunk)
                    };
                    unsafe { buffer.push_unchecked(result_chunk) };
                }

                buffer.truncate(len.div_ceil(8));
                Self::from_buffer(BitBuffer::new(buffer.freeze().into_byte_buffer(), len))
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation)]
mod tests {
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
    #[case::all_true_with_all_true(Mask::new_true(5), Mask::new_true(5), vec![0, 1, 2, 3, 4])]
    #[case::all_true_with_all_false(Mask::new_true(5), Mask::new_false(5), vec![])]
    #[case::all_false_with_any(Mask::new_false(10), Mask::new_true(0), vec![])]
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
        let m1 = Mask::from_iter([true, false, false, true, true, true, false, true]);
        let m2 = Mask::from_iter([false, false, true, false, true]);
        let result = m1.intersect_by_rank(&m2);
        let expected = Mask::from_iter([false, false, false, false, true, false, false, true]);
        assert_eq!(result, expected);
    }

    #[test]
    #[should_panic]
    fn test_intersect_by_rank_wrong_length() {
        let m1 = Mask::from_indices(10, vec![2, 5, 7]);
        let m2 = Mask::new_true(5);
        m1.intersect_by_rank(&m2);
    }

    #[rstest]
    #[case::single_element(vec![3], vec![true], vec![3])]
    #[case::single_element_masked(vec![3], vec![false], vec![])]
    #[case::alternating(vec![0, 2, 4, 6, 8], vec![true, false, true, false, true], vec![0, 4, 8])]
    #[case::consecutive(vec![5, 6, 7, 8, 9], vec![false, true, true, true, false], vec![6, 7, 8])]
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

    /// Verify the optimized implementation matches a naive reference across density combinations.
    #[rstest]
    #[case::sparse_base_sparse_rank(0.1, 0.1)]
    #[case::sparse_base_dense_rank(0.1, 0.9)]
    #[case::dense_base_sparse_rank(0.5, 0.1)]
    #[case::dense_base_dense_rank(0.5, 0.9)]
    #[case::very_sparse(0.01, 0.5)]
    #[case::very_dense_rank(0.1, 0.99)]
    fn test_intersect_by_rank_densities(#[case] base_density: f64, #[case] rank_density: f64) {
        let base_len = 1000;
        let step = (1.0 / base_density).ceil() as usize;
        let base_indices: Vec<usize> = (0..base_len).step_by(step).collect();
        let base = Mask::from_indices(base_len, base_indices.clone());

        let rank_len = base.true_count();
        let rank = Mask::from_buffer(BitBuffer::from_iter((0..rank_len).map(|i| {
            let threshold = (rank_density * 1000.0) as usize;
            (i * 7 + 13) % 1000 < threshold
        })));

        let result = base.intersect_by_rank(&rank);

        let result_indices: Vec<usize> = match result.indices() {
            crate::AllOr::Some(indices) => indices.to_vec(),
            crate::AllOr::None => vec![],
            crate::AllOr::All => (0..result.len()).collect(),
        };

        let expected: Vec<usize> = base_indices
            .iter()
            .enumerate()
            .filter(|(rank_idx, _)| match rank.bit_buffer() {
                crate::AllOr::Some(buf) => unsafe { buf.value_unchecked(*rank_idx) },
                crate::AllOr::All => true,
                crate::AllOr::None => false,
            })
            .map(|(_, &idx)| idx)
            .collect();

        assert_eq!(result_indices, expected);
    }

    #[test]
    fn test_large_mask() {
        let base_len = 200;
        let base = Mask::from_buffer(BitBuffer::from_iter((0..base_len).map(|i| i % 3 == 0)));
        let rank_len = base.true_count();
        let rank = Mask::from_buffer(BitBuffer::from_iter((0..rank_len).map(|i| i % 2 == 0)));

        let result = base.intersect_by_rank(&rank);
        assert!(result.true_count() > 0);
    }

    #[test]
    fn test_all_ones_chunk() {
        let base_len = 128;
        let base = Mask::new_true(base_len);
        let rank = Mask::from_buffer(BitBuffer::from_iter((0..base_len).map(|i| i % 2 == 0)));

        let result = base.intersect_by_rank(&rank);
        assert_eq!(result.true_count(), 64);
    }

    #[test]
    fn test_consecutive_runs() {
        let base_len = 128;
        let base = Mask::from_buffer(BitBuffer::from_iter(
            (0..base_len).map(|i| (i < 10) || (30..60).contains(&i) || (100 <= i)),
        ));
        let rank_len = base.true_count();
        let rank = Mask::from_buffer(BitBuffer::from_iter((0..rank_len).map(|i| i % 3 != 0)));

        let result = base.intersect_by_rank(&rank);
        assert!(result.true_count() > 0);
    }

    #[test]
    fn test_pdep_lut() {
        use super::pdep_lut;

        assert_eq!(pdep_lut(0b11, 0b01010100), 0b00010100);
        assert_eq!(pdep_lut(u64::MAX, 0b10101010), 0b10101010);
        assert_eq!(pdep_lut(0, 0b11111111), 0);
        assert_eq!(pdep_lut(1, 0b00001000), 0b00001000);
        assert_eq!(pdep_lut(0, 0b00001000), 0);
    }

    #[test]
    fn test_extract_bits_portable() {
        use super::extract_bits_portable;

        // flat array: [chunk0, chunk1, sentinel]
        let flat = &[0xAAAAAAAAAAAAAAAAu64, 0x5555555555555555u64, 0u64];

        assert_eq!(extract_bits_portable(flat, 0), flat[0]);
        assert_eq!(extract_bits_portable(flat, 64), flat[1]);

        let result = extract_bits_portable(flat, 32);
        let expected = (flat[0] >> 32) | (flat[1] << 32);
        assert_eq!(result, expected);
    }
}
