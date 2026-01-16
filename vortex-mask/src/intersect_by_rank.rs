// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BitBuffer;
use vortex_buffer::BufferMut;

use crate::AllOr;
use crate::Mask;

/// Extract up to 64 bits starting at bit position `start` from pre-computed chunks.
#[inline]
fn extract_bits_from_chunks(chunks: &[u64], remainder: u64, start: usize) -> u64 {
    let chunk_idx = start / 64;
    let bit_offset = start % 64;
    let num_full_chunks = chunks.len();

    let first_chunk = if chunk_idx < num_full_chunks {
        unsafe { *chunks.get_unchecked(chunk_idx) }
    } else {
        remainder
    };

    if bit_offset == 0 {
        first_chunk
    } else {
        let bits_from_first = first_chunk >> bit_offset;
        let second_chunk = if chunk_idx + 1 < num_full_chunks {
            unsafe { *chunks.get_unchecked(chunk_idx + 1) }
        } else if chunk_idx + 1 == num_full_chunks {
            remainder
        } else {
            0
        };
        bits_from_first | (second_chunk << (64 - bit_offset))
    }
}

/// Portable implementation of PDEP (parallel bit deposit).
///
/// This is the fallback when hardware BMI2 is not available.
#[inline]
fn pdep_portable(mut source: u64, mut mask: u64) -> u64 {
    let mut result = 0u64;
    while mask != 0 {
        let lowest_bit = mask & mask.wrapping_neg();
        if source & 1 != 0 {
            result |= lowest_bit;
        }
        source >>= 1;
        mask &= mask - 1;
    }
    result
}

/// Hardware PDEP using BMI2 instruction.
#[inline]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2")]
unsafe fn pdep_bmi2(source: u64, mask: u64) -> u64 {
    core::arch::x86_64::_pdep_u64(source, mask)
}

/// PDEP with runtime BMI2 detection on x86_64, falls back to portable.
#[inline]
#[cfg(target_arch = "x86_64")]
fn pdep(source: u64, mask: u64) -> u64 {
    if std::arch::is_x86_feature_detected!("bmi2") {
        // SAFETY: We just verified BMI2 is available
        unsafe { pdep_bmi2(source, mask) }
    } else {
        pdep_portable(source, mask)
    }
}

/// PDEP fallback for non-x86_64 platforms.
#[inline]
#[cfg(not(target_arch = "x86_64"))]
fn pdep(source: u64, mask: u64) -> u64 {
    pdep_portable(source, mask)
}

/// Lookup table for select within a byte: SELECT_IN_BYTE[byte | (rank << 8)]
/// gives the position of the rank-th set bit in byte (0-indexed).
/// Based on Vigna's broadword implementation.
#[rustfmt::skip]
static SELECT_IN_BYTE: [u8; 2048] = {
    let mut table = [0u8; 2048];
    let mut byte = 0usize;
    while byte < 256 {
        let mut rank = 0usize;
        while rank < 8 {
            // Find the rank-th set bit in byte
            let mut pos = 0u8;
            let mut count = 0usize;
            let mut b = byte;
            while b != 0 && count <= rank {
                if b & 1 != 0 {
                    if count == rank {
                        break;
                    }
                    count += 1;
                }
                b >>= 1;
                pos += 1;
            }
            table[byte | (rank << 8)] = if count == rank && (byte >> pos) & 1 != 0 { pos } else { 8 };
            rank += 1;
        }
        byte += 1;
    }
    table
};

/// Select the k-th set bit in a u64 (0-indexed). Returns bit position.
/// Uses PDEP on BMI2 hardware, otherwise broadword algorithm (Vigna 2008).
#[inline]
fn select_bit(word: u64, k: usize) -> usize {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("bmi2") {
        // PDEP trick: O(1) on BMI2 hardware
        let selected = unsafe { core::arch::x86_64::_pdep_u64(1u64 << k, word) };
        return selected.trailing_zeros() as usize;
    }

    // Broadword algorithm (Vigna, improved by Gog & Petri)
    // Step 1: Compute byte-level cumulative popcounts
    const ONES_STEP8: u64 = 0x0101010101010101;
    const MSB_STEP8: u64 = 0x8080808080808080;

    let mut s = word;
    s = s - ((s >> 1) & 0x5555555555555555);
    s = (s & 0x3333333333333333) + ((s >> 2) & 0x3333333333333333);
    s = (s + (s >> 4)) & 0x0f0f0f0f0f0f0f0f;
    // s now has popcount of each byte

    // byte_sums has cumulative sum in each byte position
    let byte_sums = s.wrapping_mul(ONES_STEP8);

    // Step 2: Find which byte contains the k-th bit
    let k64 = k as u64;
    let step8 = k64.wrapping_mul(ONES_STEP8);
    let geq_step8 = ((step8 | MSB_STEP8).wrapping_sub(byte_sums)) & MSB_STEP8;
    let place = (geq_step8.count_ones() as usize) * 8;

    // Step 3: Find position within that byte using lookup table
    let byte_rank = k64.wrapping_sub((byte_sums << 8) >> place & 0xFF);
    let byte_val = (word >> place) & 0xFF;
    // SAFETY: byte_val <= 255, byte_rank <= 7, so idx <= 255 | (7 << 8) = 2047 < usize::MAX on any platform
    #[allow(clippy::cast_possible_truncation)]
    let idx = (byte_val | (byte_rank << 8)) as usize;

    place + SELECT_IN_BYTE[idx] as usize
}

impl Mask {
    /// Original implementation (pre-PR) using indices and Vec collection.
    #[doc(hidden)]
    pub fn intersect_by_rank_original(&self, mask: &Mask) -> Mask {
        assert_eq!(self.true_count(), mask.len());

        match (self.indices(), mask.indices()) {
            (AllOr::All, _) => mask.clone(),
            (_, AllOr::All) => self.clone(),
            (AllOr::None, _) => Self::new_false(0),
            (_, AllOr::None) => Self::new_false(self.len()),
            (AllOr::Some(self_indices), AllOr::Some(mask_indices)) => Self::from_indices(
                self.len(),
                mask_indices
                    .iter()
                    .map(|idx| unsafe { *self_indices.get_unchecked(*idx) })
                    .collect(),
            ),
        }
    }

    /// Simple baseline implementation using indices lookup (for benchmarking).
    #[doc(hidden)]
    pub fn intersect_by_rank_simple(&self, mask: &Mask) -> Mask {
        assert_eq!(self.true_count(), mask.len());

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

                for &mask_idx in mask_indices {
                    // SAFETY: mask_idx < mask.len() == self.true_count() == self_indices.len()
                    let result_idx = unsafe { *self_indices.get_unchecked(mask_idx) };
                    chunks[result_idx / 64] |= 1u64 << (result_idx % 64);
                }

                buffer.truncate(len.div_ceil(8));
                Self::from_buffer(BitBuffer::new(buffer.freeze().into_byte_buffer(), len))
            }
        }
    }

    /// Streaming version: gets mask indices (small when mask sparse), scans self chunks with popcount.
    /// Avoids allocating self's indices when self is dense but mask is sparse.
    /// Also avoids collecting self's chunks into a vec.
    #[doc(hidden)]
    pub fn intersect_by_rank_streaming(&self, mask: &Mask) -> Mask {
        assert_eq!(self.true_count(), mask.len());

        match (self.bit_buffer(), mask.indices()) {
            (AllOr::All, _) => mask.clone(),
            (_, AllOr::All) => self.clone(),
            (AllOr::None, _) | (_, AllOr::None) => Self::new_false(self.len()),

            (AllOr::Some(self_buffer), AllOr::Some(mask_indices)) => {
                let len = self.len();
                let num_chunks = len.div_ceil(64);
                let mut buffer: BufferMut<u64> = BufferMut::zeroed(num_chunks);
                let out_chunks = buffer.as_mut_slice();

                // Get self's chunks as iterator - NO allocation
                let self_chunks = self_buffer.chunks();
                let remainder = self_chunks.remainder_bits();
                let mut chunk_iter = self_chunks.iter().peekable();

                // State for scanning self's chunks
                let mut chunk_idx = 0usize;
                let mut rank_before_chunk = 0usize;
                let mut current_chunk = chunk_iter.next().unwrap_or(remainder);
                let mut used_remainder = chunk_iter.peek().is_none() && remainder != 0;

                for &target_rank in mask_indices.iter() {
                    // Skip entire chunks using popcount until we find the chunk containing target_rank
                    loop {
                        let chunk_pop = current_chunk.count_ones() as usize;

                        if rank_before_chunk + chunk_pop > target_rank {
                            // Target is in this chunk
                            break;
                        }
                        // Skip this chunk entirely
                        rank_before_chunk += chunk_pop;
                        chunk_idx += 1;

                        // Advance to next chunk
                        if let Some(next) = chunk_iter.next() {
                            current_chunk = next;
                        } else if !used_remainder && remainder != 0 {
                            current_chunk = remainder;
                            used_remainder = true;
                        } else {
                            current_chunk = 0;
                            break;
                        }
                    }

                    // Find the exact bit within this chunk using PDEP-based select
                    let rank_in_chunk = target_rank - rank_before_chunk;
                    let bit_pos = select_bit(current_chunk, rank_in_chunk);
                    let global_pos = chunk_idx * 64 + bit_pos;

                    out_chunks[global_pos / 64] |= 1u64 << (global_pos % 64);
                }

                buffer.truncate(len.div_ceil(8));
                Self::from_buffer(BitBuffer::new(buffer.freeze().into_byte_buffer(), len))
            }
        }
    }

    /// Fully broadword-based: no trailing_zeros anywhere.
    /// Iterates mask chunks directly and uses broadword select for both mask and self.
    #[doc(hidden)]
    pub fn intersect_by_rank_broadword(&self, mask: &Mask) -> Mask {
        assert_eq!(self.true_count(), mask.len());

        match (self.bit_buffer(), mask.bit_buffer()) {
            (AllOr::All, _) => mask.clone(),
            (_, AllOr::All) => self.clone(),
            (AllOr::None, _) | (_, AllOr::None) => Self::new_false(self.len()),

            (AllOr::Some(self_buffer), AllOr::Some(mask_buffer)) => {
                let len = self.len();
                let num_chunks = len.div_ceil(64);
                let mut buffer: BufferMut<u64> = BufferMut::zeroed(num_chunks);
                let out_chunks = buffer.as_mut_slice();

                // Get chunks as iterators - no allocation
                let self_chunks = self_buffer.chunks();
                let self_remainder = self_chunks.remainder_bits();
                let mut self_iter = self_chunks.iter();
                let mut self_chunk = self_iter.next().unwrap_or(self_remainder);
                let mut self_chunk_idx = 0usize;
                let mut self_rank_before = 0usize;
                let mut self_used_remainder = false;

                let mask_chunks = mask_buffer.chunks();
                let mask_remainder = mask_chunks.remainder_bits();

                // Iterate through mask chunks
                for (mask_chunk_idx, mask_chunk) in mask_chunks.iter().enumerate() {
                    let mask_pop = mask_chunk.count_ones() as usize;
                    if mask_pop == 0 {
                        continue;
                    }

                    // For each set bit in this mask chunk (using broadword select)
                    for i in 0..mask_pop {
                        // The i-th set bit in mask_chunk is at local position select_bit(mask_chunk, i)
                        // Its global position in mask is: mask_chunk_idx * 64 + local_pos
                        // That's the rank we need from self
                        let mask_bit_local = select_bit(mask_chunk, i);
                        let wanted_rank = mask_chunk_idx * 64 + mask_bit_local;

                        // Now find the wanted_rank-th set bit in self
                        // Skip self chunks until we find the one containing this rank
                        while self_rank_before + (self_chunk.count_ones() as usize) <= wanted_rank {
                            self_rank_before += self_chunk.count_ones() as usize;
                            self_chunk_idx += 1;
                            if let Some(next) = self_iter.next() {
                                self_chunk = next;
                            } else if !self_used_remainder && self_remainder != 0 {
                                self_chunk = self_remainder;
                                self_used_remainder = true;
                            } else {
                                self_chunk = 0;
                                break;
                            }
                        }

                        // Find exact bit in self_chunk
                        let rank_in_self_chunk = wanted_rank - self_rank_before;
                        let self_bit_pos = select_bit(self_chunk, rank_in_self_chunk);
                        let global_pos = self_chunk_idx * 64 + self_bit_pos;

                        out_chunks[global_pos / 64] |= 1u64 << (global_pos % 64);
                    }
                }

                // Handle mask remainder
                if mask_remainder != 0 {
                    let mask_chunk_idx = mask_chunks.iter().count();
                    let mask_pop = mask_remainder.count_ones() as usize;

                    for i in 0..mask_pop {
                        let mask_bit_local = select_bit(mask_remainder, i);
                        let wanted_rank = mask_chunk_idx * 64 + mask_bit_local;

                        while self_rank_before + (self_chunk.count_ones() as usize) <= wanted_rank {
                            self_rank_before += self_chunk.count_ones() as usize;
                            self_chunk_idx += 1;
                            if let Some(next) = self_iter.next() {
                                self_chunk = next;
                            } else if !self_used_remainder && self_remainder != 0 {
                                self_chunk = self_remainder;
                                self_used_remainder = true;
                            } else {
                                self_chunk = 0;
                                break;
                            }
                        }

                        let rank_in_self_chunk = wanted_rank - self_rank_before;
                        let self_bit_pos = select_bit(self_chunk, rank_in_self_chunk);
                        let global_pos = self_chunk_idx * 64 + self_bit_pos;

                        out_chunks[global_pos / 64] |= 1u64 << (global_pos % 64);
                    }
                }

                buffer.truncate(len.div_ceil(8));
                Self::from_buffer(BitBuffer::new(buffer.freeze().into_byte_buffer(), len))
            }
        }
    }

    /// Arrow-style with inlined trailing_zeros iteration.
    /// Based on the working streaming implementation but uses trailing_zeros
    /// to iterate through mask bits instead of precomputed indices.
    #[doc(hidden)]
    pub fn intersect_by_rank_arrow(&self, mask: &Mask) -> Mask {
        assert_eq!(self.true_count(), mask.len());

        match (self.bit_buffer(), mask.bit_buffer()) {
            (AllOr::All, _) => mask.clone(),
            (_, AllOr::All) => self.clone(),
            (AllOr::None, _) | (_, AllOr::None) => Self::new_false(self.len()),

            (AllOr::Some(self_buffer), AllOr::Some(mask_buffer)) => {
                let len = self.len();
                let num_chunks = len.div_ceil(64);
                let mut buffer: BufferMut<u64> = BufferMut::zeroed(num_chunks);
                let out_chunks = buffer.as_mut_slice();

                // Self chunks - use select_bit like streaming
                let self_chunks = self_buffer.chunks();
                let self_remainder = self_chunks.remainder_bits();
                let mut self_iter = self_chunks.iter().peekable();
                let mut self_chunk_idx = 0usize;
                let mut self_rank_before = 0usize;
                let mut self_current = self_iter.next().unwrap_or(self_remainder);

                // Mask chunks - iterate with trailing_zeros
                let mask_chunks = mask_buffer.chunks();
                let mask_remainder = mask_chunks.remainder_bits();

                // Process full mask chunks
                for (mask_chunk_idx, mut mask_chunk) in mask_chunks.iter().enumerate() {
                    while mask_chunk != 0 {
                        // Get position of lowest set bit using trailing_zeros
                        let mask_bit_pos = mask_chunk.trailing_zeros() as usize;
                        let wanted_rank = mask_chunk_idx * 64 + mask_bit_pos;
                        mask_chunk ^= 1u64 << mask_bit_pos; // Clear this bit

                        // Skip self chunks until we find the one containing wanted_rank
                        while self_rank_before + (self_current.count_ones() as usize) <= wanted_rank
                        {
                            self_rank_before += self_current.count_ones() as usize;
                            self_chunk_idx += 1;
                            if let Some(&next) = self_iter.peek() {
                                self_iter.next();
                                self_current = next;
                            } else if self_remainder != 0
                                && self_chunk_idx == mask_chunks.iter().count()
                            {
                                self_current = self_remainder;
                            } else {
                                self_current = 0;
                                break;
                            }
                        }

                        // Find exact bit in self_chunk using select_bit
                        let rank_in_self_chunk = wanted_rank - self_rank_before;
                        let self_bit_pos = select_bit(self_current, rank_in_self_chunk);
                        let global_pos = self_chunk_idx * 64 + self_bit_pos;
                        out_chunks[global_pos / 64] |= 1u64 << (global_pos % 64);
                    }
                }

                // Process mask remainder
                if mask_remainder != 0 {
                    let mask_chunk_idx = mask_chunks.iter().count();
                    let mut mask_chunk = mask_remainder;

                    while mask_chunk != 0 {
                        let mask_bit_pos = mask_chunk.trailing_zeros() as usize;
                        let wanted_rank = mask_chunk_idx * 64 + mask_bit_pos;
                        mask_chunk ^= 1u64 << mask_bit_pos;

                        while self_rank_before + (self_current.count_ones() as usize) <= wanted_rank
                        {
                            self_rank_before += self_current.count_ones() as usize;
                            self_chunk_idx += 1;
                            if let Some(next) = self_iter.next() {
                                self_current = next;
                            } else if self_remainder != 0 {
                                self_current = self_remainder;
                            } else {
                                self_current = 0;
                                break;
                            }
                        }

                        let rank_in_self_chunk = wanted_rank - self_rank_before;
                        let self_bit_pos = select_bit(self_current, rank_in_self_chunk);
                        let global_pos = self_chunk_idx * 64 + self_bit_pos;
                        out_chunks[global_pos / 64] |= 1u64 << (global_pos % 64);
                    }
                }

                buffer.truncate(len.div_ceil(8));
                Self::from_buffer(BitBuffer::new(buffer.freeze().into_byte_buffer(), len))
            }
        }
    }

    /// Unrolled 2x version (for benchmarking).
    #[doc(hidden)]
    pub fn intersect_by_rank_unrolled(&self, mask: &Mask) -> Mask {
        assert_eq!(self.true_count(), mask.len());

        match (self.bit_buffer(), mask.bit_buffer()) {
            (AllOr::All, _) => mask.clone(),
            (_, AllOr::All) => self.clone(),
            (AllOr::None, _) | (_, AllOr::None) => Self::new_false(self.len()),

            (AllOr::Some(self_buffer), AllOr::Some(mask_buffer)) => {
                let len = self.len();
                let num_chunks = len.div_ceil(64);
                let mut buffer: BufferMut<u64> = BufferMut::with_capacity(num_chunks);
                let mut rank = 0usize;

                let self_chunks = self_buffer.chunks();
                let mask_chunks = mask_buffer.chunks();
                let mask_chunk_vec: Vec<u64> = mask_chunks.iter().collect();
                let mask_remainder = mask_chunks.remainder_bits();

                // Collect self chunks for indexed access
                let self_chunk_slice: Vec<u64> = self_chunks.iter().collect();
                let num_full_chunks = self_chunk_slice.len();

                // Process pairs of chunks (2x unroll)
                let mut i = 0;
                while i + 1 < num_full_chunks {
                    let c0 = self_chunk_slice[i];
                    let c1 = self_chunk_slice[i + 1];

                    let pop0 = c0.count_ones() as usize;
                    let pop1 = c1.count_ones() as usize;

                    // Process chunk 0
                    let r0 = if c0 == 0 {
                        0u64
                    } else if c0 == u64::MAX {
                        extract_bits_from_chunks(&mask_chunk_vec, mask_remainder, rank)
                    } else {
                        let rank_bits =
                            extract_bits_from_chunks(&mask_chunk_vec, mask_remainder, rank);
                        pdep(rank_bits, c0)
                    };

                    // Process chunk 1
                    let r1 = if c1 == 0 {
                        0u64
                    } else if c1 == u64::MAX {
                        extract_bits_from_chunks(&mask_chunk_vec, mask_remainder, rank + pop0)
                    } else {
                        let rank_bits =
                            extract_bits_from_chunks(&mask_chunk_vec, mask_remainder, rank + pop0);
                        pdep(rank_bits, c1)
                    };

                    rank += pop0 + pop1;
                    unsafe {
                        buffer.push_unchecked(r0);
                        buffer.push_unchecked(r1);
                    }
                    i += 2;
                }

                // Handle remaining chunk (if odd number)
                while i < num_full_chunks {
                    let self_chunk = self_chunk_slice[i];
                    let popcount = self_chunk.count_ones() as usize;

                    let result_chunk = if self_chunk == 0 {
                        0u64
                    } else if self_chunk == u64::MAX {
                        extract_bits_from_chunks(&mask_chunk_vec, mask_remainder, rank)
                    } else {
                        let rank_bits =
                            extract_bits_from_chunks(&mask_chunk_vec, mask_remainder, rank);
                        pdep(rank_bits, self_chunk)
                    };

                    rank += popcount;
                    unsafe { buffer.push_unchecked(result_chunk) };
                    i += 1;
                }

                // Handle remainder bits
                let remainder = len % 64;
                if remainder != 0 {
                    let self_chunk = self_chunks.remainder_bits();

                    let result_chunk = if self_chunk == 0 {
                        0u64
                    } else {
                        let rank_bits =
                            extract_bits_from_chunks(&mask_chunk_vec, mask_remainder, rank);
                        pdep(rank_bits, self_chunk)
                    };

                    unsafe { buffer.push_unchecked(result_chunk) };
                }

                buffer.truncate(len.div_ceil(8));
                Self::from_buffer(BitBuffer::new(buffer.freeze().into_byte_buffer(), len))
            }
        }
    }

    /// Portable PDEP implementation (for benchmarking without BMI2).
    #[doc(hidden)]
    pub fn intersect_by_rank_portable(&self, mask: &Mask) -> Mask {
        assert_eq!(self.true_count(), mask.len());

        match (self.bit_buffer(), mask.bit_buffer()) {
            (AllOr::All, _) => mask.clone(),
            (_, AllOr::All) => self.clone(),
            (AllOr::None, _) | (_, AllOr::None) => Self::new_false(self.len()),

            (AllOr::Some(self_buffer), AllOr::Some(mask_buffer)) => {
                let len = self.len();
                let num_chunks = len.div_ceil(64);
                let mut buffer: BufferMut<u64> = BufferMut::with_capacity(num_chunks);
                let mut rank = 0usize;

                let self_chunks = self_buffer.chunks();
                let mask_chunks = mask_buffer.chunks();
                let mask_chunk_vec: Vec<u64> = mask_chunks.iter().collect();
                let mask_remainder = mask_chunks.remainder_bits();

                for self_chunk in self_chunks.iter() {
                    let popcount = self_chunk.count_ones() as usize;

                    let result_chunk = if self_chunk == 0 {
                        0u64
                    } else if self_chunk == u64::MAX {
                        extract_bits_from_chunks(&mask_chunk_vec, mask_remainder, rank)
                    } else {
                        let rank_bits =
                            extract_bits_from_chunks(&mask_chunk_vec, mask_remainder, rank);
                        pdep_portable(rank_bits, self_chunk)
                    };

                    rank += popcount;
                    unsafe { buffer.push_unchecked(result_chunk) };
                }

                let remainder = len % 64;
                if remainder != 0 {
                    let self_chunk = self_chunks.remainder_bits();

                    let result_chunk = if self_chunk == 0 {
                        0u64
                    } else {
                        let rank_bits =
                            extract_bits_from_chunks(&mask_chunk_vec, mask_remainder, rank);
                        pdep_portable(rank_bits, self_chunk)
                    };

                    unsafe { buffer.push_unchecked(result_chunk) };
                }

                buffer.truncate(len.div_ceil(8));
                Self::from_buffer(BitBuffer::new(buffer.freeze().into_byte_buffer(), len))
            }
        }
    }

    /// Take the intersection of the `mask` with the set of true values in `self`.
    ///
    /// This method adaptively chooses between two algorithms based on mask density:
    /// - For sparse masks (≤5% density), uses indices-based O(true_count) algorithm
    /// - For dense masks (>5% density), uses chunk-based PDEP algorithm
    ///
    /// The chunk-based algorithm is optimized for correlated data patterns.
    /// For chunks that are all 1s (runs of consecutive trues), it directly copies
    /// 64 bits from the rank mask. For mixed chunks, it uses PDEP-style bit scattering.
    ///
    /// On x86_64 with BMI2 support, this uses the hardware PDEP instruction for
    /// significant performance gains (5-6x faster than portable implementation).
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

        // Adaptive dispatch: use simple (indices-based) when either mask is sparse.
        // - Simple is O(mask.true_count) iterations with random access
        // - Chunk-based is O(self.len/64) iterations with sequential access
        //
        // Use simple when either:
        // 1. Self is sparse: true_count * 20 <= len (≈5% density)
        // 2. Mask is sparse: true_count * 20 <= len (≈5% density)
        //
        // Using integer arithmetic avoids floating-point overhead.
        let self_sparse = self.true_count().saturating_mul(20) <= self.len();
        let mask_sparse = mask.true_count().saturating_mul(20) <= mask.len();
        if self_sparse || mask_sparse {
            return self.intersect_by_rank_simple(mask);
        }

        match (self.bit_buffer(), mask.bit_buffer()) {
            (AllOr::All, _) => mask.clone(),
            (_, AllOr::All) => self.clone(),
            (AllOr::None, _) | (_, AllOr::None) => Self::new_false(self.len()),

            (AllOr::Some(self_buffer), AllOr::Some(mask_buffer)) => {
                let len = self.len();
                let num_chunks = len.div_ceil(64);
                let mut buffer: BufferMut<u64> = BufferMut::with_capacity(num_chunks);
                let mut rank = 0usize;

                let self_chunks = self_buffer.chunks();
                let mask_chunks = mask_buffer.chunks();
                let mask_chunk_vec: Vec<u64> = mask_chunks.iter().collect();
                let mask_remainder = mask_chunks.remainder_bits();

                // Process full 64-bit chunks
                for self_chunk in self_chunks.iter() {
                    let popcount = self_chunk.count_ones() as usize;

                    let result_chunk = if self_chunk == 0 {
                        // All zeros - skip
                        0u64
                    } else if self_chunk == u64::MAX {
                        // All ones - copy directly from mask
                        extract_bits_from_chunks(&mask_chunk_vec, mask_remainder, rank)
                    } else {
                        // Mixed - scatter bits using PDEP
                        let rank_bits =
                            extract_bits_from_chunks(&mask_chunk_vec, mask_remainder, rank);
                        pdep(rank_bits, self_chunk)
                    };

                    rank += popcount;
                    // SAFETY: we allocated enough capacity
                    unsafe { buffer.push_unchecked(result_chunk) };
                }

                // Handle remainder bits
                let remainder = len % 64;
                if remainder != 0 {
                    let self_chunk = self_chunks.remainder_bits();

                    let result_chunk = if self_chunk == 0 {
                        0u64
                    } else {
                        let rank_bits =
                            extract_bits_from_chunks(&mask_chunk_vec, mask_remainder, rank);
                        pdep(rank_bits, self_chunk)
                    };

                    // SAFETY: we allocated enough capacity
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

    #[rstest]
    #[case::random_sparse(0.1, 0.5)]
    #[case::random_dense(0.5, 0.5)]
    #[case::all_true_base(1.0, 0.5)]
    fn test_simple_matches_optimized(#[case] base_density: f64, #[case] rank_density: f64) {
        let base_len = 200;
        let base = Mask::from_buffer(BitBuffer::from_iter((0..base_len).map(|i| {
            let threshold = (base_density * 100.0) as usize;
            (i * 7 + 13) % 100 < threshold
        })));
        let rank_len = base.true_count();
        if rank_len == 0 {
            return;
        }
        let rank = Mask::from_buffer(BitBuffer::from_iter((0..rank_len).map(|i| {
            let threshold = (rank_density * 100.0) as usize;
            (i * 11 + 7) % 100 < threshold
        })));

        let result_simple = base.intersect_by_rank_simple(&rank);
        let result_optimized = base.intersect_by_rank(&rank);
        assert_eq!(result_simple, result_optimized);
    }

    /// Test all four density combinations to ensure all implementations match.
    #[rstest]
    #[case::self_sparse_mask_sparse(0.05, 0.05)]
    #[case::self_sparse_mask_dense(0.05, 0.50)]
    #[case::self_dense_mask_sparse(0.50, 0.05)]
    #[case::self_dense_mask_dense(0.50, 0.50)]
    fn test_all_implementations_match(#[case] self_density: f64, #[case] mask_density: f64) {
        let base_len = 10_000;
        let base = Mask::from_buffer(BitBuffer::from_iter((0..base_len).map(|i| {
            let threshold = (self_density * 1000.0) as usize;
            (i * 7 + 13) % 1000 < threshold
        })));
        let rank_len = base.true_count();
        if rank_len == 0 {
            return;
        }
        let rank = Mask::from_buffer(BitBuffer::from_iter((0..rank_len).map(|i| {
            let threshold = (mask_density * 1000.0) as usize;
            (i * 11 + 7) % 1000 < threshold
        })));

        let result_original = base.intersect_by_rank_original(&rank);
        let result_simple = base.intersect_by_rank_simple(&rank);
        let result_streaming = base.intersect_by_rank_streaming(&rank);
        let result_broadword = base.intersect_by_rank_broadword(&rank);
        let result_arrow = base.intersect_by_rank_arrow(&rank);
        let result_portable = base.intersect_by_rank_portable(&rank);
        let result_unrolled = base.intersect_by_rank_unrolled(&rank);
        let result_optimized = base.intersect_by_rank(&rank);

        // All implementations should produce the same result
        assert_eq!(result_original, result_simple, "original != simple");
        assert_eq!(result_simple, result_streaming, "simple != streaming");
        assert_eq!(result_streaming, result_broadword, "streaming != broadword");
        assert_eq!(result_broadword, result_arrow, "broadword != arrow");
        assert_eq!(result_arrow, result_portable, "arrow != portable");
        assert_eq!(result_portable, result_unrolled, "portable != unrolled");
        assert_eq!(result_unrolled, result_optimized, "unrolled != optimized");
    }

    #[test]
    fn test_pdep_portable() {
        use super::pdep_portable;

        assert_eq!(pdep_portable(0b11, 0b01010100), 0b00010100);
        assert_eq!(pdep_portable(u64::MAX, 0b10101010), 0b10101010);
        assert_eq!(pdep_portable(0, 0b11111111), 0);
        assert_eq!(pdep_portable(1, 0b00001000), 0b00001000);
        assert_eq!(pdep_portable(0, 0b00001000), 0);
    }

    #[test]
    fn test_extract_bits_from_chunks() {
        use super::extract_bits_from_chunks;

        let chunks = &[0xAAAAAAAAAAAAAAAAu64, 0x5555555555555555u64];
        let remainder = 0u64;

        assert_eq!(extract_bits_from_chunks(chunks, remainder, 0), chunks[0]);
        assert_eq!(extract_bits_from_chunks(chunks, remainder, 64), chunks[1]);

        let result = extract_bits_from_chunks(chunks, remainder, 32);
        let expected = (chunks[0] >> 32) | (chunks[1] << 32);
        assert_eq!(result, expected);
    }
}
