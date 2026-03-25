// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
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
    out_true_count: usize,
) -> Mask {
    let has_remainder = !len.is_multiple_of(64);
    let num_out = self_full_chunks.len() + has_remainder as usize;
    let mask_ptr = mask_flat.as_ptr();

    let mut buffer: BufferMut<u64> = BufferMut::with_capacity(num_out);
    let mut bit_pos: usize = 0;

    for &self_chunk in self_full_chunks {
        let chunk_idx = bit_pos >> 6;
        let lo = unsafe { *mask_ptr.add(chunk_idx) };
        let hi = unsafe { *mask_ptr.add(chunk_idx + 1) };
        let rank_bits = unsafe { extract_shrd(lo, hi, bit_pos as u8) };

        let result_chunk = core::arch::x86_64::_pdep_u64(rank_bits, self_chunk);
        unsafe { buffer.push_unchecked(result_chunk) };

        bit_pos += self_chunk.count_ones() as usize;
    }

    if has_remainder {
        let chunk_idx = bit_pos >> 6;
        let lo = unsafe { *mask_ptr.add(chunk_idx) };
        let hi = unsafe { *mask_ptr.add(chunk_idx + 1) };
        let rank_bits = unsafe { extract_shrd(lo, hi, bit_pos as u8) };
        let result_chunk = core::arch::x86_64::_pdep_u64(rank_bits, self_remainder);
        unsafe { buffer.push_unchecked(result_chunk) };
    }

    buffer.truncate(len.div_ceil(8));
    Mask::from_buffer_with_true_count(
        BitBuffer::new(buffer.freeze().into_byte_buffer(), len),
        out_true_count,
    )
}

/// In-place BMI2+SHRD loop: overwrites self's buffer with intersection results.
///
/// Reads each self_chunk from `buf_ptr[i]`, computes PDEP with the corresponding
/// rank bits from the mask, and writes the result back to `buf_ptr[i]`. This avoids
/// allocating a separate output buffer, reducing memory streams from 3 to 2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2,popcnt")]
#[allow(clippy::cast_possible_truncation)]
unsafe fn dense_loop_bmi2_inplace(
    buf_ptr: *mut u64,
    num_full: usize,
    self_remainder: u64,
    mask_flat: &[u64],
    len: usize,
) {
    let mask_ptr = mask_flat.as_ptr();
    let mut bit_pos: usize = 0;

    for i in 0..num_full {
        let self_chunk = unsafe { *buf_ptr.add(i) };
        let chunk_idx = bit_pos >> 6;
        let lo = unsafe { *mask_ptr.add(chunk_idx) };
        let hi = unsafe { *mask_ptr.add(chunk_idx + 1) };
        let rank_bits = unsafe { extract_shrd(lo, hi, bit_pos as u8) };
        let result_chunk = core::arch::x86_64::_pdep_u64(rank_bits, self_chunk);
        unsafe { *buf_ptr.add(i) = result_chunk };
        bit_pos += self_chunk.count_ones() as usize;
    }

    if !len.is_multiple_of(64) {
        let chunk_idx = bit_pos >> 6;
        let lo = unsafe { *mask_ptr.add(chunk_idx) };
        let hi = unsafe { *mask_ptr.add(chunk_idx + 1) };
        let rank_bits = unsafe { extract_shrd(lo, hi, bit_pos as u8) };
        let result_chunk = core::arch::x86_64::_pdep_u64(rank_bits, self_remainder);
        // Write remainder as individual bytes to avoid overwriting past the buffer
        let rem_bytes = (len % 64).div_ceil(8);
        let dst = unsafe { (buf_ptr as *mut u8).add(num_full * 8) };
        for b in 0..rem_bytes {
            unsafe { *dst.add(b) = (result_chunk >> (b * 8)) as u8 };
        }
    }
}

/// Dense portable loop operating on raw u64 slices for zero-copy performance.
///
/// Same algorithm as the iterator-based portable path, but accepts pre-extracted
/// raw u64 chunks + remainder to avoid BitChunks iterator overhead.
#[allow(clippy::cast_possible_truncation)]
fn dense_loop_portable(
    self_full_chunks: &[u64],
    self_remainder: u64,
    mask_flat: &[u64],
    len: usize,
    out_true_count: usize,
) -> Mask {
    let has_remainder = !len.is_multiple_of(64);
    let num_out = self_full_chunks.len() + has_remainder as usize;
    let mut buffer: BufferMut<u64> = BufferMut::with_capacity(num_out);
    let mut bit_pos = 0usize;

    for &self_chunk in self_full_chunks {
        let popcount = self_chunk.count_ones() as usize;
        let result_chunk = if self_chunk == 0 {
            0u64
        } else {
            let rank_bits = extract_bits_portable(mask_flat, bit_pos);
            pdep_lut(rank_bits, self_chunk)
        };
        unsafe { buffer.push_unchecked(result_chunk) };
        bit_pos += popcount;
    }

    if has_remainder {
        let result_chunk = if self_remainder == 0 {
            0u64
        } else {
            let rank_bits = extract_bits_portable(mask_flat, bit_pos);
            pdep_lut(rank_bits, self_remainder)
        };
        unsafe { buffer.push_unchecked(result_chunk) };
    }

    buffer.truncate(len.div_ceil(8));
    Mask::from_buffer_with_true_count(
        BitBuffer::new(buffer.freeze().into_byte_buffer(), len),
        out_true_count,
    )
}

/// In-place portable loop: overwrites self's buffer with intersection results.
#[allow(clippy::cast_possible_truncation)]
fn dense_loop_portable_inplace(
    buf_ptr: *mut u64,
    num_full: usize,
    self_remainder: u64,
    mask_flat: &[u64],
    len: usize,
) {
    let mut bit_pos = 0usize;

    for i in 0..num_full {
        let self_chunk = unsafe { *buf_ptr.add(i) };
        let popcount = self_chunk.count_ones() as usize;
        let result_chunk = if self_chunk == 0 {
            0u64
        } else {
            let rank_bits = extract_bits_portable(mask_flat, bit_pos);
            pdep_lut(rank_bits, self_chunk)
        };
        unsafe { *buf_ptr.add(i) = result_chunk };
        bit_pos += popcount;
    }

    if !len.is_multiple_of(64) {
        let result_chunk = if self_remainder == 0 {
            0u64
        } else {
            let rank_bits = extract_bits_portable(mask_flat, bit_pos);
            pdep_lut(rank_bits, self_remainder)
        };
        let rem_bytes = (len % 64).div_ceil(8);
        let dst = unsafe { (buf_ptr as *mut u8).add(num_full * 8) };
        for b in 0..rem_bytes {
            unsafe { *dst.add(b) = (result_chunk >> (b * 8)) as u8 };
        }
    }
}

/// Try to get a mutable `*mut u64` pointer from a `BitBufferMut`, returning
/// the pointer, num_full chunks, and remainder bits if the buffer is u64-aligned
/// with zero offset.
fn try_inplace_ptr(buf: &mut BitBufferMut) -> Option<(*mut u64, usize, u64)> {
    if buf.offset() != 0 {
        return None;
    }
    let ptr = buf.as_mut_ptr();
    if ptr.align_offset(align_of::<u64>()) != 0 {
        return None;
    }
    let len = buf.len();
    let num_full = len / 64;
    let remainder = if !len.is_multiple_of(64) {
        let bytes = buf.as_slice();
        let rem_bytes = &bytes[num_full * 8..];
        let mut val = 0u64;
        for (i, &b) in rem_bytes.iter().enumerate() {
            val |= (b as u64) << (i * 8);
        }
        val & ((1u64 << (len % 64)) - 1)
    } else {
        0
    };
    Some((ptr.cast::<u64>(), num_full, remainder))
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
                let out_true_count = mask_indices.len();
                if out_true_count == 0 {
                    return Self::new_false(len);
                }

                let num_chunks = len.div_ceil(64);
                let mut buffer: BufferMut<u64> = BufferMut::zeroed(num_chunks);
                let chunks = buffer.as_mut_slice();

                for &idx in mask_indices.iter() {
                    let bit_pos = unsafe { *self_indices.get_unchecked(idx) };
                    unsafe {
                        *chunks.get_unchecked_mut(bit_pos / 64) |= 1u64 << (bit_pos % 64);
                    }
                }

                buffer.truncate(len.div_ceil(8));
                Self::from_buffer_with_true_count(
                    BitBuffer::new(buffer.freeze().into_byte_buffer(), len),
                    out_true_count,
                )
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
                let out_true_count = mask.true_count();
                let mask_flat = build_mask_flat(mask_buffer);

                let (self_full, self_remainder) =
                    if let Some((raw, rem)) = bit_buffer_raw_u64s(self_buffer) {
                        (std::borrow::Cow::Borrowed(raw), rem)
                    } else {
                        let chunks = self_buffer.chunks();
                        let full: Vec<u64> = chunks.iter().collect();
                        let rem = chunks.remainder_bits();
                        (std::borrow::Cow::Owned(full), rem)
                    };

                #[cfg(target_arch = "x86_64")]
                if std::arch::is_x86_feature_detected!("bmi2") {
                    return unsafe {
                        dense_loop_bmi2(
                            &self_full,
                            self_remainder,
                            &mask_flat,
                            len,
                            out_true_count,
                        )
                    };
                }

                dense_loop_portable(
                    &self_full,
                    self_remainder,
                    &mask_flat,
                    len,
                    out_true_count,
                )
            }
        }
    }

    /// Owned variant of [`intersect_by_rank`](Self::intersect_by_rank) that takes `self` by value.
    ///
    /// When `self` has unique ownership (no other clones), this performs the intersection
    /// in-place, avoiding output buffer allocation and reducing memory streams from 3 to 2.
    /// When ownership is shared, falls back to the allocating path.
    ///
    /// Callers that don't need `self` after the call should prefer this method.
    pub fn intersect_by_rank_owned(self, mask: &Mask) -> Mask {
        assert_eq!(self.true_count(), mask.len());

        let self_sparse = self.true_count().saturating_mul(20) <= self.len();
        let mask_sparse = mask.true_count().saturating_mul(20) <= mask.len();
        if self_sparse || mask_sparse {
            return self.intersect_by_rank_sparse(mask);
        }

        // Handle trivial variants before consuming self
        if matches!(&self, Mask::AllFalse(_)) || matches!(mask.bit_buffer(), AllOr::None) {
            return Self::new_false(self.len());
        }
        if matches!(&self, Mask::AllTrue(_)) {
            return mask.clone();
        }
        if matches!(mask.bit_buffer(), AllOr::All) {
            return self;
        }

        let len = self.len();
        let out_true_count = mask.true_count();
        let mask_buffer = match mask.bit_buffer() {
            AllOr::Some(b) => b,
            _ => unreachable!(),
        };
        let mask_flat = build_mask_flat(mask_buffer);

        // Consume self to try for in-place operation
        let self_bit_buffer = self.into_bit_buffer();
        match self_bit_buffer.try_into_mut() {
            Ok(mut bit_buf_mut) => {
                if let Some((buf_ptr, num_full, remainder)) = try_inplace_ptr(&mut bit_buf_mut) {
                    #[cfg(target_arch = "x86_64")]
                    if std::arch::is_x86_feature_detected!("bmi2") {
                        unsafe {
                            dense_loop_bmi2_inplace(
                                buf_ptr, num_full, remainder, &mask_flat, len,
                            );
                        }
                        return Mask::from_buffer_with_true_count(
                            bit_buf_mut.freeze(),
                            out_true_count,
                        );
                    }

                    dense_loop_portable_inplace(buf_ptr, num_full, remainder, &mask_flat, len);
                    return Mask::from_buffer_with_true_count(
                        bit_buf_mut.freeze(),
                        out_true_count,
                    );
                }
                // Not aligned — freeze back and use allocating path
                let frozen = bit_buf_mut.freeze();
                Self::intersect_dense_alloc(frozen, &mask_flat, len, out_true_count)
            }
            Err(self_bit_buffer) => {
                // Shared buffer — use allocating path
                Self::intersect_dense_alloc(self_bit_buffer, &mask_flat, len, out_true_count)
            }
        }
    }

    /// Allocating dense intersection: extracts chunks from the BitBuffer and delegates
    /// to the appropriate loop (BMI2 or portable).
    fn intersect_dense_alloc(
        self_buffer: BitBuffer,
        mask_flat: &[u64],
        len: usize,
        out_true_count: usize,
    ) -> Mask {
        let (self_full, self_remainder) =
            if let Some((raw, rem)) = bit_buffer_raw_u64s(&self_buffer) {
                (std::borrow::Cow::Borrowed(raw), rem)
            } else {
                let chunks = self_buffer.chunks();
                let full: Vec<u64> = chunks.iter().collect();
                let rem = chunks.remainder_bits();
                (std::borrow::Cow::Owned(full), rem)
            };

        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("bmi2") {
            return unsafe {
                dense_loop_bmi2(&self_full, self_remainder, mask_flat, len, out_true_count)
            };
        }

        dense_loop_portable(&self_full, self_remainder, mask_flat, len, out_true_count)
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

    /// Verify `intersect_by_rank_owned` matches `intersect_by_rank` across densities.
    #[rstest]
    #[case::dense_50(0.5, 0.5)]
    #[case::dense_10(0.5, 0.1)]
    #[case::dense_90(0.5, 0.9)]
    #[case::sparse_base(0.1, 0.5)]
    fn test_intersect_by_rank_owned(#[case] base_density: f64, #[case] rank_density: f64) {
        let base_len = 1000;
        let step = (1.0 / base_density).ceil() as usize;
        let base_indices: Vec<usize> = (0..base_len).step_by(step).collect();
        let base = Mask::from_indices(base_len, base_indices);

        let rank_len = base.true_count();
        let rank = Mask::from_buffer(BitBuffer::from_iter((0..rank_len).map(|i| {
            let threshold = (rank_density * 1000.0) as usize;
            (i * 7 + 13) % 1000 < threshold
        })));

        let base2 = base.clone();
        let expected = base.intersect_by_rank(&rank);
        let result = base2.intersect_by_rank_owned(&rank);
        assert_eq!(result, expected);
    }

    /// Verify owned variant works with large masks that exercise multi-chunk paths.
    #[test]
    fn test_intersect_by_rank_owned_large() {
        let base_len = 10_000;
        let base = Mask::from_buffer(BitBuffer::from_iter((0..base_len).map(|i| i % 2 == 0)));
        let rank_len = base.true_count();
        let rank = Mask::from_buffer(BitBuffer::from_iter((0..rank_len).map(|i| i % 3 == 0)));

        let base2 = base.clone();
        let expected = base.intersect_by_rank(&rank);
        let result = base2.intersect_by_rank_owned(&rank);
        assert_eq!(result, expected);
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
