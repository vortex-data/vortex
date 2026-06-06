// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! SIMD packing of "truthy" bytes into an LSB-first bitmask.
//!
//! Packs `value.len()` bytes into bits where bit `i` is set iff `value[i] != 0`.
//! This is the contiguous-slice fast path behind `BitBufferMut::from(&[u8])` and
//! `BitBufferMut::from(&[bool])`.
//!
//! Expressed as a vector compare into an opmask, this lowers to one
//! `vptestmb` + `kmovq` per 64 bytes on AVX-512BW. The scalar
//! `packed |= (b != 0) << i` reduction loop does not: LLVM's SLP vectorizer
//! rewrites it into a `vpsllvq` shift-OR reduction instead, which is ~10-20x
//! slower for cache-resident inputs.

/// Pack `value.len()` truthy bytes (`b != 0`) into `words`, LSB-first, 64 bits
/// per `u64`. `words` must have at least `value.len().div_ceil(64)` entries.
#[inline]
pub fn pack_nonzero_bytes(words: &mut [u64], value: &[u8]) {
    let num_words = value.len().div_ceil(64);
    assert!(
        words.len() >= num_words,
        "words slice has {} entries, need at least {num_words}",
        words.len(),
    );

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
            // SAFETY: guarded by the runtime feature checks above.
            return unsafe { pack_nonzero_bytes_avx512(words, value) };
        }
        if is_x86_feature_detected!("avx2") {
            // SAFETY: guarded by the runtime feature check above.
            return unsafe { pack_nonzero_bytes_avx2(words, value) };
        }
    }

    pack_nonzero_bytes_scalar(words, value);
}

/// Portable fallback used directly on non-x86 targets and as the tail handler.
#[inline]
fn pack_nonzero_bytes_scalar(words: &mut [u64], value: &[u8]) {
    let full = value.len() / 64;
    for (word, chunk) in words.iter_mut().zip(value.chunks_exact(64)) {
        let mut bits = 0u64;
        for (i, &b) in chunk.iter().enumerate() {
            bits |= ((b != 0) as u64) << i;
        }
        *word = bits;
    }
    if !value.len().is_multiple_of(64) {
        let base = full * 64;
        let mut bits = 0u64;
        for (i, &b) in value[base..].iter().enumerate() {
            bits |= ((b != 0) as u64) << i;
        }
        words[full] = bits;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn pack_nonzero_bytes_avx512(words: &mut [u64], value: &[u8]) {
    use std::arch::x86_64::__m512i;
    use std::arch::x86_64::_mm512_loadu_si512;
    use std::arch::x86_64::_mm512_test_epi8_mask;

    let full = value.len() / 64;
    let ptr = value.as_ptr();
    for (i, word) in words.iter_mut().take(full).enumerate() {
        // SAFETY: i < full so the 64-byte load stays in bounds.
        let v = unsafe { _mm512_loadu_si512(ptr.add(i * 64) as *const __m512i) };
        // vptestmb: per-byte (v & v) != 0 -> 64-bit opmask; kmovq stores it.
        *word = _mm512_test_epi8_mask(v, v);
    }
    if !value.len().is_multiple_of(64) {
        pack_nonzero_bytes_scalar(&mut words[full..], &value[full * 64..]);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn pack_nonzero_bytes_avx2(words: &mut [u64], value: &[u8]) {
    use std::arch::x86_64::__m256i;
    use std::arch::x86_64::_mm256_cmpeq_epi8;
    use std::arch::x86_64::_mm256_loadu_si256;
    use std::arch::x86_64::_mm256_movemask_epi8;
    use std::arch::x86_64::_mm256_setzero_si256;

    let full = value.len() / 64;
    let ptr = value.as_ptr();
    let zero = _mm256_setzero_si256();
    for (i, word) in words.iter_mut().take(full).enumerate() {
        // Two 32-byte halves; movemask gives 1 bit per byte. cmpeq vs zero
        // marks zero bytes, so invert to mark non-zero bytes.
        // SAFETY: i < full so both 32-byte loads stay in bounds.
        let lo = unsafe { _mm256_loadu_si256(ptr.add(i * 64) as *const __m256i) };
        // SAFETY: i < full so the second 32-byte load stays in bounds.
        let hi = unsafe { _mm256_loadu_si256(ptr.add(i * 64 + 32) as *const __m256i) };
        let lo_zero = _mm256_movemask_epi8(_mm256_cmpeq_epi8(lo, zero)) as u32;
        let hi_zero = _mm256_movemask_epi8(_mm256_cmpeq_epi8(hi, zero)) as u32;
        let bits = (!lo_zero as u64) | ((!hi_zero as u64) << 32);
        *word = bits;
    }
    if !value.len().is_multiple_of(64) {
        pack_nonzero_bytes_scalar(&mut words[full..], &value[full * 64..]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(value: &[u8]) -> Vec<u64> {
        let mut w = vec![0u64; value.len().div_ceil(64)];
        for (i, &b) in value.iter().enumerate() {
            if b != 0 {
                w[i / 64] |= 1u64 << (i % 64);
            }
        }
        w
    }

    #[test]
    fn matches_reference() {
        for &n in &[0usize, 1, 7, 63, 64, 65, 127, 128, 200, 1000, 4096] {
            // Mix of zero and varied non-zero bytes to exercise the != 0 test.
            let data: Vec<u8> = (0..n)
                .map(|i| {
                    if i.is_multiple_of(5) {
                        0
                    } else {
                        u8::try_from(i % 200 + 1).unwrap()
                    }
                })
                .collect();
            let mut got = vec![0u64; n.div_ceil(64)];
            pack_nonzero_bytes(&mut got, &data);
            assert_eq!(got, reference(&data), "mismatch at n={n}");
        }
    }
}
