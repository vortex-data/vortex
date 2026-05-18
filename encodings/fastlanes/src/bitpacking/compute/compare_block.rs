// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Block-at-a-time decompress + compare + bit-pack kernel.
//!
//! For each 1024-element packed block:
//!  1. FastLanes `unchecked_unpack` into a reusable stack buffer (4 KB, fits in L1).
//!  2. Per 64 elements, build one `u64` bitmap word with the per-element comparison.
//!  3. Write the chunk's 16 u64 bitmap into the result.
//!
//! No `PrimitiveArray` allocation, no Arrow compare wrapper — the decompressed primitives
//! never leave the cache line they were unpacked into. Works at every supported bit
//! width via FastLanes' runtime-W unpack dispatcher; the `width` parameter goes straight
//! into the FastLanes packing/unpacking match.
//!
//! The bit-packing step (turning 1024 element compares into 16 u64 bitmap words) uses
//! AVX2 `cmpeq_epi32` + `movemask_ps` when available — same pattern Arrow's `cmp::eq`
//! emits — so the kernel reaches Arrow throughput while keeping the unpacked buffer
//! L1-resident. Scalar fallback retains the 8-element manual unroll.

use fastlanes::BitPacking;

const ELEMS_PER_CHUNK: usize = 1024;

/// Reusable stack buffer for one decoded chunk.
type Block = [u32; ELEMS_PER_CHUNK];

/// Unpack one 1024-element chunk into the reusable `block` buffer.
#[inline(always)]
fn unpack_block(packed_chunk: &[u32], w: usize, block: &mut Block) {
    debug_assert_eq!(packed_chunk.len(), 32 * w);
    // SAFETY: `packed_chunk` has the required `128 * w / size_of::<u32>()` length and
    // `block` is exactly 1024 elements.
    unsafe { BitPacking::unchecked_unpack(w, packed_chunk, block) };
}

/// Block-decompress Eq. Writes the chunk's 16 u64 element-order bitmap into
/// `chunk_bits`.
pub(crate) fn block_eq_u32(
    packed_chunk: &[u32],
    w: usize,
    c: u32,
    block: &mut Block,
    chunk_bits: &mut [u64; 16],
) {
    unpack_block(packed_chunk, w, block);
    pack_eq_bits(block, c, chunk_bits);
}

/// Block-decompress unsigned Lt.
pub(crate) fn block_lt_u32(
    packed_chunk: &[u32],
    w: usize,
    c: u32,
    block: &mut Block,
    chunk_bits: &mut [u64; 16],
) {
    unpack_block(packed_chunk, w, block);
    pack_lt_bits(block, c, chunk_bits);
}

/// Pack `block[i] == c` for `i ∈ 0..1024` into 16 u64 element-order words.
///
/// AVX2 path: per output u64, 8 `_mm256_cmpeq_epi32` + `_mm256_movemask_ps` cover 64
/// elements as 8 nibbles of 8 bits each. Scalar fallback unrolls 8 byte-groups manually.
#[inline]
fn pack_eq_bits(block: &Block, c: u32, chunk_bits: &mut [u64; 16]) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: target_feature=avx2 gates compilation here.
        unsafe {
            use std::arch::x86_64::*;
            let c_vec = _mm256_set1_epi32(c as i32);
            for u in 0..16 {
                let base = u * 64;
                let mut word = 0u64;
                for byte_i in 0..8 {
                    let off = base + byte_i * 8;
                    let ymm = _mm256_loadu_si256(block.as_ptr().add(off).cast::<__m256i>());
                    let cmp = _mm256_cmpeq_epi32(ymm, c_vec);
                    let mask =
                        u64::from(_mm256_movemask_ps(_mm256_castsi256_ps(cmp)) as u32 & 0xFF);
                    word |= mask << (byte_i * 8);
                }
                chunk_bits[u] = word;
            }
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        scalar_pack_eq_bits(block, c, chunk_bits);
    }
}

#[inline]
fn pack_lt_bits(block: &Block, c: u32, chunk_bits: &mut [u64; 16]) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // AVX2 unsigned u32 less-than via the `min_epu32` identity:
        //   a <= c iff min(a, c) == a
        //   a <  c iff (a <= c) AND NOT (a == c)
        unsafe {
            use std::arch::x86_64::*;
            let c_vec = _mm256_set1_epi32(c as i32);
            for u in 0..16 {
                let base = u * 64;
                let mut word = 0u64;
                for byte_i in 0..8 {
                    let off = base + byte_i * 8;
                    let ymm = _mm256_loadu_si256(block.as_ptr().add(off).cast::<__m256i>());
                    let min = _mm256_min_epu32(ymm, c_vec);
                    let le = _mm256_cmpeq_epi32(min, ymm);
                    let eq = _mm256_cmpeq_epi32(ymm, c_vec);
                    let lt = _mm256_andnot_si256(eq, le);
                    let mask = u64::from(_mm256_movemask_ps(_mm256_castsi256_ps(lt)) as u32 & 0xFF);
                    word |= mask << (byte_i * 8);
                }
                chunk_bits[u] = word;
            }
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        scalar_pack_lt_bits(block, c, chunk_bits);
    }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
#[inline]
fn scalar_pack_eq_bits(block: &Block, c: u32, chunk_bits: &mut [u64; 16]) {
    for u in 0..16 {
        let base = u * 64;
        let mut bytes = [0u8; 8];
        for byte_i in 0..8 {
            let off = base + byte_i * 8;
            let b0 = u8::from(block[off] == c);
            let b1 = u8::from(block[off + 1] == c);
            let b2 = u8::from(block[off + 2] == c);
            let b3 = u8::from(block[off + 3] == c);
            let b4 = u8::from(block[off + 4] == c);
            let b5 = u8::from(block[off + 5] == c);
            let b6 = u8::from(block[off + 6] == c);
            let b7 = u8::from(block[off + 7] == c);
            bytes[byte_i] = b0
                | (b1 << 1)
                | (b2 << 2)
                | (b3 << 3)
                | (b4 << 4)
                | (b5 << 5)
                | (b6 << 6)
                | (b7 << 7);
        }
        chunk_bits[u] = u64::from_le_bytes(bytes);
    }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
#[inline]
fn scalar_pack_lt_bits(block: &Block, c: u32, chunk_bits: &mut [u64; 16]) {
    for u in 0..16 {
        let base = u * 64;
        let mut bytes = [0u8; 8];
        for byte_i in 0..8 {
            let off = base + byte_i * 8;
            let b0 = u8::from(block[off] < c);
            let b1 = u8::from(block[off + 1] < c);
            let b2 = u8::from(block[off + 2] < c);
            let b3 = u8::from(block[off + 3] < c);
            let b4 = u8::from(block[off + 4] < c);
            let b5 = u8::from(block[off + 5] < c);
            let b6 = u8::from(block[off + 6] < c);
            let b7 = u8::from(block[off + 7] < c);
            bytes[byte_i] = b0
                | (b1 << 1)
                | (b2 << 2)
                | (b3 << 3)
                | (b4 << 4)
                | (b5 << 5)
                | (b6 << 6)
                | (b7 << 7);
        }
        chunk_bits[u] = u64::from_le_bytes(bytes);
    }
}

/// Allocate the reusable block buffer once. Stack-resident, 4 KB.
#[inline(always)]
pub(crate) fn new_block() -> Block {
    [0u32; ELEMS_PER_CHUNK]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pack_u32(values: &[u32; 1024], w: usize) -> Vec<u32> {
        let mut out = vec![0u32; 32 * w];
        unsafe {
            BitPacking::unchecked_pack(w, values, &mut out);
        }
        out
    }

    fn bit(chunk_bits: &[u64; 16], i: usize) -> bool {
        (chunk_bits[i / 64] >> (i % 64)) & 1 != 0
    }

    #[test]
    fn block_eq_w7_matches_naive() {
        let w = 7;
        let mask = (1u32 << w) - 1;
        let mut values = [0u32; 1024];
        for (i, v) in values.iter_mut().enumerate() {
            *v = ((i as u32).wrapping_mul(31).wrapping_add(7)) & mask;
        }
        let packed = pack_u32(&values, w);
        let mut block = new_block();
        for &c in &[0u32, 1, 5, 42, 100, 127] {
            let mut got = [0u64; 16];
            block_eq_u32(&packed, w, c, &mut block, &mut got);
            for i in 0..1024 {
                assert_eq!(bit(&got, i), values[i] == c, "i={i}, c={c}");
            }
        }
    }

    #[test]
    fn block_lt_w11_matches_naive() {
        let w = 11;
        let mask = (1u32 << w) - 1;
        let mut values = [0u32; 1024];
        for (i, v) in values.iter_mut().enumerate() {
            *v = ((i as u32).wrapping_mul(17).wrapping_add(3)) & mask;
        }
        let packed = pack_u32(&values, w);
        let mut block = new_block();
        for &c in &[0u32, 1, 10, 100, 1000, 2047] {
            let mut got = [0u64; 16];
            block_lt_u32(&packed, w, c, &mut block, &mut got);
            for i in 0..1024 {
                assert_eq!(bit(&got, i), values[i] < c, "i={i}, c={c}");
            }
        }
    }
}
