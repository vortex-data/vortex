// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use super::QuantizedQuery;
use super::split_code;

/// Integer part of IVF scoring. Both inputs contain signed bytes, with the stored vector
/// borrowed directly from its on-disk representation.
///
/// Callers must supply equal lengths and satisfy the kernel's CPU and dimension requirements.
trait SignedDotProduct {
    unsafe fn dot(query: &[i8], values: &[u8]) -> i32;
}

struct Portable;

impl SignedDotProduct for Portable {
    unsafe fn dot(query: &[i8], values: &[u8]) -> i32 {
        let mut lanes = [0i32; 16];
        let (q_chunks, q_rest) = query.as_chunks::<16>();
        let (c_chunks, c_rest) = values.as_chunks::<16>();
        for (q, c) in q_chunks.iter().zip(c_chunks) {
            for lane in 0..16 {
                lanes[lane] += i32::from(q[lane]) * i32::from(c[lane] as i8);
            }
        }
        let rest: i32 = q_rest
            .iter()
            .zip(c_rest)
            .map(|(&q, &c)| i32::from(q) * i32::from(c as i8))
            .sum();
        lanes.iter().sum::<i32>() + rest
    }
}

#[cfg(target_arch = "aarch64")]
struct Arm128;

#[cfg(target_arch = "aarch64")]
impl SignedDotProduct for Arm128 {
    #[inline]
    #[target_feature(enable = "neon,dotprod")]
    unsafe fn dot(query: &[i8], values: &[u8]) -> i32 {
        query[..128]
            .iter()
            .zip(&values[..128])
            .map(|(&q, &v)| i32::from(q) * i32::from(v as i8))
            .sum()
    }
}

#[cfg(target_arch = "x86_64")]
struct Avx2;

#[cfg(target_arch = "x86_64")]
impl SignedDotProduct for Avx2 {
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn dot(query: &[i8], values: &[u8]) -> i32 {
        use std::arch::x86_64::*;
        let wide = query.len() / 16 * 16;
        let mut lanes = _mm256_setzero_si256();
        for start in (0..wide).step_by(16) {
            // SAFETY: `start + 16 <= wide`, and the query and the code's values both hold
            // `query.len() >= wide` bytes, so each unaligned 16-byte load is in bounds.
            let (q, c) = unsafe {
                (
                    _mm_loadu_si128(query.as_ptr().add(start).cast()),
                    _mm_loadu_si128(values.as_ptr().add(start).cast()),
                )
            };
            let products = _mm256_madd_epi16(_mm256_cvtepi8_epi16(q), _mm256_cvtepi8_epi16(c));
            lanes = _mm256_add_epi32(lanes, products);
        }
        let half = _mm_add_epi32(
            _mm256_castsi256_si128(lanes),
            _mm256_extracti128_si256::<1>(lanes),
        );
        let half = _mm_add_epi32(half, _mm_shuffle_epi32::<0b0100_1110>(half));
        let half = _mm_add_epi32(half, _mm_shuffle_epi32::<0b1011_0001>(half));
        let rest: i32 = query[wide..]
            .iter()
            .zip(&values[wide..])
            .map(|(&q, &c)| i32::from(q) * i32::from(c as i8))
            .sum();
        _mm_cvtsi128_si32(half) + rest
    }
}

impl QuantizedQuery {
    pub(super) fn dot_all(&self, codes: &[u8], out: &mut Vec<f32>) {
        out.clear();
        out.reserve(codes.len() / (4 + self.values.len()));
        #[cfg(target_arch = "aarch64")]
        if self.values.len() == 128 && std::arch::is_aarch64_feature_detected!("dotprod") {
            // SAFETY: the CPU supports SDOT and the query has 128 dimensions.
            return unsafe { self.arm128_scores(codes, out) };
        }
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: the CPU supports AVX2.
            return unsafe { self.avx2_scores(codes, out) };
        }
        // SAFETY: the portable kernel supports all CPUs and dimensions.
        unsafe { self.scores::<Portable>(codes, 4 + self.values.len(), out) }
    }

    // Compile the entire scoring loop for the chosen ISA so the per-row dot can inline.
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon,dotprod")]
    unsafe fn arm128_scores(&self, codes: &[u8], out: &mut Vec<f32>) {
        // SAFETY: dot_all checked the dimension and CPU feature.
        unsafe { self.scores::<Arm128>(codes, 132, out) }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_scores(&self, codes: &[u8], out: &mut Vec<f32>) {
        // SAFETY: this function requires AVX2.
        unsafe { self.scores::<Avx2>(codes, 4 + self.values.len(), out) }
    }

    #[expect(
        clippy::inline_always,
        reason = "inline the loop into its target-feature caller"
    )]
    #[inline(always)]
    unsafe fn scores<K: SignedDotProduct>(
        &self,
        codes: &[u8],
        code_len: usize,
        out: &mut Vec<f32>,
    ) {
        for code in codes.chunks_exact(code_len) {
            let (scale, values) = split_code(code);
            // SAFETY: the caller selected a supported kernel and each code contains
            // exactly one byte per query dimension after its scale.
            let sum = unsafe { K::dot(&self.values, values) };
            #[expect(clippy::cast_precision_loss, reason = "IVF scores are f32")]
            out.push(sum as f32 * scale * self.scale);
        }
    }
}
