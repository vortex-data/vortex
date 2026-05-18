//! Reference delta-undo for 32-lane × 32-step `u8` blocks in a transposed,
//! step-major layout.
//!
//! # Layout
//!
//! A "block" is 1024 `u8` values arranged as 32 lanes × 32 steps. Memory is
//! **step-major**: step `s`'s 32 lane values are the contiguous bytes
//! `input[s*32 .. s*32 + 32]`. Equivalently `input[s * 32 + lane]`.
//!
//! This mirrors the contiguity rule from fastlanes-rs's `iterate!` macro,
//! where varying the lane index for a fixed `row` produces consecutive
//! memory positions. We pick 32 lanes (the AVX2 byte width) rather than
//! the 128 lanes that fastlanes-rs uses for u8, because the stencil-JIT
//! sibling code targets a single ymm register at a time. With 32 steps
//! this still yields a 1024-element block, matching the FastLanes block
//! size convention.
//!
//! Delta-undo is independent per lane:
//!
//! ```text
//! for lane in 0..32 {
//!     let mut prev = base[lane];
//!     for step in 0..32 {
//!         let next = input[step * 32 + lane].wrapping_add(prev);
//!         output[step * 32 + lane] = next;
//!         prev = next;
//!     }
//! }
//! ```
//!
//! Because lanes are independent, the SIMD shape is "one register-wide
//! `vpaddb` per step across all 32 lanes simultaneously":
//!
//! ```text
//! ymm_prev := vmovdqu base           ; 32 lane-prevs in one register
//! for step in 0..32 {
//!     ymm_cur  := vmovdqu input + step*32
//!     ymm_prev := vpaddb ymm_prev, ymm_cur
//!     vmovdqu  output + step*32, ymm_prev
//! }
//! ```
//!
//! The placeholder slot for the eventual stencil-JIT'd version is
//! [`undelta_jit_placeholder`], which currently just forwards to
//! [`undelta_avx2`].

// The scalar reference deliberately mirrors fastlanes-rs's lane/step
// loop structure, where indexing through `base[lane]` and the indexed
// input/output access is part of the pattern we want to read alongside
// the upstream code.
#![allow(clippy::needless_range_loop)]

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{__m256i, _mm256_add_epi8, _mm256_loadu_si256, _mm256_storeu_si256};

/// Number of lanes processed in parallel by the AVX2 inner loop.
pub const LANES: usize = 32;

/// Number of delta steps per lane (block size / `LANES`).
pub const STEPS: usize = 32;

/// Total elements in a single 1024-byte block.
pub const BLOCK: usize = LANES * STEPS;

/// Scalar reference implementation of delta-undo.
///
/// Walks the 32 lanes independently, carrying each lane's running sum
/// in `prev`. Matches the structure of `Delta::undelta` in
/// `fastlanes-rs/src/delta.rs`, specialized to the 32-lane × 32-step
/// step-major layout described in the module docs.
pub fn undelta_scalar(input: &[u8; BLOCK], base: &[u8; LANES], output: &mut [u8; BLOCK]) {
    for lane in 0..LANES {
        let mut prev = base[lane];
        for step in 0..STEPS {
            let idx = step * LANES + lane;
            let next = input[idx].wrapping_add(prev);
            output[idx] = next;
            prev = next;
        }
    }
}

/// Forward delta (companion to `undelta_scalar`), used only by the tests.
///
/// Produces the encoded form: each value becomes `input[i] - prev`,
/// where `prev` is `base[lane]` for the first step and the previous
/// raw input for subsequent steps within the same lane.
pub fn delta_scalar(input: &[u8; BLOCK], base: &[u8; LANES], output: &mut [u8; BLOCK]) {
    for lane in 0..LANES {
        let mut prev = base[lane];
        for step in 0..STEPS {
            let idx = step * LANES + lane;
            let next = input[idx];
            output[idx] = next.wrapping_sub(prev);
            prev = next;
        }
    }
}

/// AVX2 intrinsics implementation of delta-undo.
///
/// One `vpaddb` per step across all 32 lanes. Contiguous 32-byte loads
/// and stores rely on the step-major layout described in the module
/// docs.
///
/// # Safety
///
/// The caller must ensure the host CPU supports AVX2. Buffer references
/// already guarantee 1024 readable input bytes and 1024 writable output
/// bytes; no alignment beyond `u8` is required.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn undelta_avx2(input: &[u8; BLOCK], base: &[u8; LANES], output: &mut [u8; BLOCK]) {
    // SAFETY: `base` is 32 readable bytes; an unaligned load is sound.
    let mut prev = unsafe { _mm256_loadu_si256(base.as_ptr() as *const __m256i) };
    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();
    for step in 0..STEPS {
        let off = step * LANES;
        // SAFETY: `off + 32 <= BLOCK` for every step in 0..STEPS; both
        // pointers are derived from `&[u8; BLOCK]` so they are valid for
        // a 32-byte unaligned access.
        let cur = unsafe { _mm256_loadu_si256(in_ptr.add(off) as *const __m256i) };
        prev = _mm256_add_epi8(prev, cur);
        // SAFETY: same range as the load.
        unsafe { _mm256_storeu_si256(out_ptr.add(off) as *mut __m256i, prev) };
    }
}

/// Placeholder for the eventual stencil-JIT'd delta-undo kernel.
///
/// The slot exists so the benchmark example can time a "JIT" entry that
/// a follow-up session will replace with a real copy-and-patch kernel.
/// Today it just forwards to [`undelta_avx2`].
///
/// # Safety
///
/// Same contract as [`undelta_avx2`]: the caller must ensure the host
/// CPU supports AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn undelta_jit_placeholder(
    input: &[u8; BLOCK],
    base: &[u8; LANES],
    output: &mut [u8; BLOCK],
) {
    // SAFETY: forwarded to `undelta_avx2`; same AVX2 contract.
    unsafe { undelta_avx2(input, base, output) }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny xorshift32 so the tests have no external rand dep.
    struct XorShift32(u32);
    impl XorShift32 {
        fn new(seed: u32) -> Self {
            Self(if seed == 0 { 0x9E37_79B9 } else { seed })
        }
        fn next_u32(&mut self) -> u32 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            self.0 = x;
            x
        }
        fn fill(&mut self, dst: &mut [u8]) {
            for chunk in dst.chunks_mut(4) {
                let v = self.next_u32().to_le_bytes();
                for (d, s) in chunk.iter_mut().zip(v.iter()) {
                    *d = *s;
                }
            }
        }
    }

    fn empty_block() -> Box<[u8; BLOCK]> {
        vec![0u8; BLOCK].into_boxed_slice().try_into().unwrap()
    }

    #[test]
    fn round_trip_scalar() {
        let mut rng = XorShift32::new(0xC0FFEE);
        let mut values = empty_block();
        let mut base = [0u8; LANES];
        rng.fill(values.as_mut_slice());
        rng.fill(&mut base);

        let mut deltas = empty_block();
        delta_scalar(&values, &base, &mut deltas);

        let mut restored = empty_block();
        undelta_scalar(&deltas, &base, &mut restored);

        assert_eq!(values, restored);
    }

    #[test]
    fn round_trip_avx2() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let mut rng = XorShift32::new(0xDEAD_BEEF);
        let mut values = empty_block();
        let mut base = [0u8; LANES];
        rng.fill(values.as_mut_slice());
        rng.fill(&mut base);

        let mut deltas = empty_block();
        delta_scalar(&values, &base, &mut deltas);

        let mut restored = empty_block();
        // SAFETY: avx2 verified above.
        unsafe { undelta_avx2(&deltas, &base, &mut restored) };

        assert_eq!(values, restored);
    }

    #[test]
    fn scalar_and_avx2_agree_on_random_inputs() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let mut rng = XorShift32::new(0x1234_5678);
        for _ in 0..64 {
            let mut input = empty_block();
            let mut base = [0u8; LANES];
            rng.fill(input.as_mut_slice());
            rng.fill(&mut base);

            let mut out_scalar = empty_block();
            let mut out_avx2 = empty_block();
            undelta_scalar(&input, &base, &mut out_scalar);
            // SAFETY: avx2 verified above.
            unsafe { undelta_avx2(&input, &base, &mut out_avx2) };

            assert_eq!(out_scalar, out_avx2);
        }
    }

    #[test]
    fn placeholder_matches_avx2() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let mut rng = XorShift32::new(0xABCD);
        let mut input = empty_block();
        let mut base = [0u8; LANES];
        rng.fill(input.as_mut_slice());
        rng.fill(&mut base);

        let mut out_avx2 = empty_block();
        let mut out_placeholder = empty_block();
        // SAFETY: avx2 verified above.
        unsafe {
            undelta_avx2(&input, &base, &mut out_avx2);
            undelta_jit_placeholder(&input, &base, &mut out_placeholder);
        }
        assert_eq!(out_avx2, out_placeholder);
    }

    #[test]
    fn zero_base_zero_input_yields_zero() {
        let input = empty_block();
        let base = [0u8; LANES];
        let mut output = empty_block();
        undelta_scalar(&input, &base, &mut output);
        assert!(output.iter().all(|&b| b == 0));
    }

    #[test]
    fn constant_delta_one_produces_arithmetic_progression() {
        // input is "1" everywhere, base is "0" everywhere => each lane
        // counts 1..=32.
        let mut input = empty_block();
        for byte in input.iter_mut() {
            *byte = 1;
        }
        let base = [0u8; LANES];
        let mut output = empty_block();
        undelta_scalar(&input, &base, &mut output);

        for lane in 0..LANES {
            for step in 0..STEPS {
                let got = output[step * LANES + lane];
                let want = (step as u8).wrapping_add(1);
                assert_eq!(got, want, "lane {lane} step {step}");
            }
        }
    }
}
