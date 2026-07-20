// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Kernels for packing boolean values into bitmap words.
//!
//! `collect_bool` materializes each full 64-bit chunk as a `[bool; 64]` (a loop the
//! auto-vectorizer turns into vector stores for simple predicates) and then packs the 64 bytes
//! into a single `u64` with a byte→bit kernel:
//!
//! - x86-64 AVX-512BW: one `vptestmb` produces the full 64-bit mask.
//! - x86-64 AVX2: two `vpmovmskb` halves.
//! - x86-64 SSE2 (baseline): four `pmovmskb` quarters.
//! - aarch64 NEON (baseline): per-lane `ushl` by the bit position, then an `addp` reduction tree.
//! - elsewhere (and under Miri): a branch-free SWAR multiply.
//!
//! There are two tiers. The default ([`collect_bool_words_inline`]) compiles the loop once
//! with the widest *statically-enabled* kernel (SSE2 / NEON / SWAR on stock targets) and
//! inlines fully into the caller — safe for arbitrary predicates. The opt-in tier
//! ([`collect_bool_words_multiversioned`]) compiles the loop — with `f` inside — once per CPU
//! feature level and selects a clone by runtime detection; only for predicates small and
//! simple enough that the per-level duplication and its `#[target_feature]` call boundary pay
//! off.
//!
//! The bit-at-a-time loop lives on as [`collect_bool_word_scalar`], used for tail chunks and as
//! the reference implementation for tests and benchmarks.

/// Packs up to 64 boolean values into a little-endian `u64` word one bit at a time.
///
/// This is the scalar reference implementation behind
/// [`collect_bool_word`](crate::bit::collect_bool_word); prefer calling that entry point, which
/// takes the SIMD fast path for full 64-bit words.
#[inline]
pub fn collect_bool_word_scalar<F>(len: usize, mut f: F) -> u64
where
    F: FnMut(usize) -> bool,
{
    assert!(len <= 64, "cannot pack {len} bits into a u64 word");

    let mut packed = 0;
    for bit_idx in 0..len {
        packed |= (f(bit_idx) as u64) << bit_idx;
    }
    packed
}

/// Body of [`collect_bool_words`](crate::bit::collect_bool_words) (and, via a one-word slice,
/// of [`collect_bool_word`](crate::bit::collect_bool_word)): the word loop with the widest
/// pack kernel enabled *at compile time* — SSE2 on stock x86-64 (AVX2/AVX-512BW when built
/// with e.g. `-C target-cpu=native`), NEON on aarch64, SWAR elsewhere and under Miri.
///
/// Statically-enabled kernels are part of every function's feature set, so this loop
/// (predicate, the `[bool; 64]` materialization, and the pack) inlines fully into the caller
/// with no `#[target_feature]` boundary. That boundary is why *runtime*-detected wider kernels
/// are not used here: hiding an expensive, non-vectorizable predicate (e.g. FSST's per-string
/// DFA scan) behind a non-inlinable AVX-512 loop copy costs far more (~30% end to end) than
/// the wider pack saves — and an indirect call per word is worse still (~4x on cheap
/// predicates), since an opaque call target blocks fill/pack fusion regardless of how cheap
/// the kernel *selection* is. For provably cheap predicates, use [`collect_bool_words_multiversioned`].
#[inline(always)]
pub(crate) fn collect_bool_words_inline<F>(words: &mut [u64], len: usize, f: F)
where
    F: FnMut(usize) -> bool,
{
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw",
        not(miri)
    ))]
    {
        // SAFETY: AVX-512F/BW are statically enabled for this build (e.g. -C
        // target-cpu=native), so they are in every function's feature set and the kernel
        // inlines here like any other function.
        collect_bool_words_with(words, len, f, |bools| unsafe {
            pack_bool_word_avx512(bools)
        })
    }
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        not(all(target_feature = "avx512f", target_feature = "avx512bw")),
        not(miri)
    ))]
    {
        // SAFETY: AVX2 is statically enabled for this build.
        collect_bool_words_with(words, len, f, |bools| unsafe { pack_bool_word_avx2(bools) })
    }
    #[cfg(all(target_arch = "x86_64", not(target_feature = "avx2"), not(miri)))]
    {
        // SAFETY: SSE2 is part of the x86-64 baseline instruction set.
        collect_bool_words_with(words, len, f, |bools| unsafe { pack_bool_word_sse2(bools) })
    }
    #[cfg(all(target_arch = "aarch64", not(miri)))]
    {
        // SAFETY: NEON is part of the aarch64 baseline instruction set.
        collect_bool_words_with(words, len, f, |bools| unsafe { pack_bool_word_neon(bools) })
    }
    #[cfg(any(not(any(target_arch = "x86_64", target_arch = "aarch64")), miri))]
    collect_bool_words_with(words, len, f, pack_bool_word_swar)
}

/// Word loop with the *widest* pack kernel the CPU offers (AVX-512BW, then AVX2, then the
/// baseline), for predicates known to be cheap.
///
/// The wide loop copies live behind a `#[target_feature]` function boundary that cannot inline
/// into the caller, which deoptimizes expensive predicates (see the module docs and
/// `collect_bool_words_inline`). Only route a predicate here when its evaluation is trivial
/// — e.g. the bounds-check-free slice gathers in the `From<&[bool]>` / `From<&[u8]>`
/// conversions, or unchecked slice comparisons like the `between` kernels — where the fused
/// AVX-512 loop is worth another ~2x over the baseline kernel.
///
/// `words` must hold at least `len.div_ceil(64)` entries and `f` is invoked with `0..len`,
/// exactly once per index in ascending order.
///
/// Panics if `words` is too short.
#[inline]
pub fn collect_bool_words_multiversioned<F>(words: &mut [u64], len: usize, f: F)
where
    F: FnMut(usize) -> bool,
{
    let num_words = len.div_ceil(64);
    assert!(
        words.len() >= num_words,
        "words slice has {} entries, need at least {num_words}",
        words.len(),
    );

    // Without a full 64-bit word only the scalar tail would run; skip feature detection and
    // the `#[target_feature]` call boundary entirely.
    if len < 64 {
        return collect_bool_words_inline(words, len, f);
    }

    #[cfg(all(target_arch = "x86_64", not(miri)))]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
            // SAFETY: runtime detection guarantees the required target features.
            return unsafe { collect_bool_words_avx512(words, len, f) };
        }
        if is_x86_feature_detected!("avx2") {
            // SAFETY: runtime detection guarantees the required target features.
            return unsafe { collect_bool_words_avx2(words, len, f) };
        }
    }
    collect_bool_words_inline(words, len, f)
}

/// Shared word loop: materialize each full 64-bit chunk as a `[bool; 64]` and pack it with
/// `pack`; the tail chunk goes through the scalar loop.
///
/// Marked `#[inline(always)]` so each `#[target_feature]` wrapper gets its own fully-inlined
/// copy compiled with that feature set.
#[inline(always)]
fn collect_bool_words_with<F, P>(words: &mut [u64], len: usize, mut f: F, pack: P)
where
    F: FnMut(usize) -> bool,
    P: Fn(&[bool; 64]) -> u64,
{
    let full = len / 64;
    let remainder = len % 64;

    for (word_idx, word) in words[..full].iter_mut().enumerate() {
        let offset = word_idx * 64;
        let mut bools = [false; 64];
        for (bit_idx, b) in bools.iter_mut().enumerate() {
            *b = f(offset + bit_idx);
        }
        *word = pack(&bools);
    }

    if remainder != 0 {
        let offset = full * 64;
        words[full] = collect_bool_word_scalar(remainder, |bit_idx| f(offset + bit_idx));
    }
}

/// SSE2 copy of the [`collect_bool_words`](crate::bit::collect_bool_words) word loop.
///
/// `words` must hold at least `len.div_ceil(64)` entries and `f` is invoked with `0..len`.
///
/// # Safety
///
/// The caller must ensure the CPU supports SSE2 (always true on x86-64).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
pub unsafe fn collect_bool_words_sse2<F: FnMut(usize) -> bool>(
    words: &mut [u64],
    len: usize,
    f: F,
) {
    // SAFETY: the caller guarantees SSE2 support.
    collect_bool_words_with(words, len, f, |bools| unsafe { pack_bool_word_sse2(bools) })
}

/// AVX2 copy of the [`collect_bool_words`](crate::bit::collect_bool_words) word loop.
///
/// `words` must hold at least `len.div_ceil(64)` entries and `f` is invoked with `0..len`.
///
/// # Safety
///
/// The caller must ensure the CPU supports AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn collect_bool_words_avx2<F: FnMut(usize) -> bool>(
    words: &mut [u64],
    len: usize,
    f: F,
) {
    // SAFETY: the caller guarantees AVX2 support.
    collect_bool_words_with(words, len, f, |bools| unsafe { pack_bool_word_avx2(bools) })
}

/// AVX-512BW copy of the [`collect_bool_words`](crate::bit::collect_bool_words) word loop.
///
/// `words` must hold at least `len.div_ceil(64)` entries and `f` is invoked with `0..len`.
///
/// # Safety
///
/// The caller must ensure the CPU supports AVX-512F and AVX-512BW.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn collect_bool_words_avx512<F: FnMut(usize) -> bool>(
    words: &mut [u64],
    len: usize,
    f: F,
) {
    // SAFETY: the caller guarantees AVX-512F and AVX-512BW support.
    collect_bool_words_with(words, len, f, |bools| unsafe {
        pack_bool_word_avx512(bools)
    })
}

/// NEON copy of the [`collect_bool_words`](crate::bit::collect_bool_words) word loop.
///
/// `words` must hold at least `len.div_ceil(64)` entries and `f` is invoked with `0..len`.
///
/// # Safety
///
/// The caller must ensure the CPU supports NEON (always true on aarch64).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn collect_bool_words_neon<F: FnMut(usize) -> bool>(
    words: &mut [u64],
    len: usize,
    f: F,
) {
    // SAFETY: the caller guarantees NEON support.
    collect_bool_words_with(words, len, f, |bools| unsafe { pack_bool_word_neon(bools) })
}

/// Portable branch-free byte→bit pack kernel, used when no SIMD kernel is available.
///
/// Reads the bools eight at a time as a `u64` and gathers the eight `0x00`/`0x01` bytes into
/// eight contiguous bits with a single multiply: byte `i` contributes `2^(8i)`, and the magic
/// constant `0x0102_0408_1020_4080 = Σ 2^(56-7i)` shifts each contribution to bit `56 + i`
/// without any cross-term collisions, so the mask falls out of the top byte of the product.
#[inline]
pub fn pack_bool_word_swar(bools: &[bool; 64]) -> u64 {
    const MAGIC: u64 = 0x0102_0408_1020_4080;

    let (chunks, rest) = bools.as_chunks::<8>();
    debug_assert!(rest.is_empty());

    let mut packed = 0u64;
    for (chunk_idx, chunk) in chunks.iter().enumerate() {
        let word = u64::from_le_bytes(chunk.map(|b| b as u8));
        packed |= (word.wrapping_mul(MAGIC) >> 56) << (8 * chunk_idx);
    }
    packed
}

/// SSE2 byte→bit pack kernel: four 16-byte `pcmpeqb`-against-zero + `pmovmskb` rounds.
///
/// # Safety
///
/// The caller must ensure the CPU supports SSE2 (always true on x86-64).
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn pack_bool_word_sse2(bools: &[bool; 64]) -> u64 {
    use std::arch::x86_64::__m128i;
    use std::arch::x86_64::_mm_cmpeq_epi8;
    use std::arch::x86_64::_mm_loadu_si128;
    use std::arch::x86_64::_mm_movemask_epi8;
    use std::arch::x86_64::_mm_setzero_si128;

    let ptr = bools.as_ptr().cast::<u8>();
    let zero = _mm_setzero_si128();

    let mut packed = 0u64;
    for lane in 0..4 {
        // SAFETY: `lane * 16 + 16 <= 64`, so the 16-byte load is in bounds.
        let chunk = unsafe { _mm_loadu_si128(ptr.add(lane * 16).cast::<__m128i>()) };
        // `cmpeq` against zero sets 0xFF for *false* bytes; invert to get the truthy mask.
        let zero_mask = _mm_movemask_epi8(_mm_cmpeq_epi8(chunk, zero)) as u32 as u64;
        packed |= (!zero_mask & 0xFFFF) << (16 * lane);
    }
    packed
}

/// AVX2 byte→bit pack kernel: two 32-byte `vpcmpeqb`-against-zero + `vpmovmskb` rounds.
///
/// # Safety
///
/// The caller must ensure the CPU supports AVX2.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn pack_bool_word_avx2(bools: &[bool; 64]) -> u64 {
    use std::arch::x86_64::__m256i;
    use std::arch::x86_64::_mm256_cmpeq_epi8;
    use std::arch::x86_64::_mm256_loadu_si256;
    use std::arch::x86_64::_mm256_movemask_epi8;
    use std::arch::x86_64::_mm256_setzero_si256;

    let ptr = bools.as_ptr().cast::<u8>();
    let zero = _mm256_setzero_si256();

    // SAFETY: both 32-byte loads are within the 64-byte array.
    let lo = unsafe { _mm256_loadu_si256(ptr.cast::<__m256i>()) };
    // SAFETY: see above.
    let hi = unsafe { _mm256_loadu_si256(ptr.add(32).cast::<__m256i>()) };

    // `cmpeq` against zero sets 0xFF for *false* bytes; invert to get the truthy mask.
    let lo_mask = !(_mm256_movemask_epi8(_mm256_cmpeq_epi8(lo, zero)) as u32) as u64;
    let hi_mask = !(_mm256_movemask_epi8(_mm256_cmpeq_epi8(hi, zero)) as u32) as u64;
    lo_mask | (hi_mask << 32)
}

/// AVX-512BW byte→bit pack kernel: a single 64-byte `vptestmb` produces the whole word.
///
/// # Safety
///
/// The caller must ensure the CPU supports AVX-512F and AVX-512BW.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn pack_bool_word_avx512(bools: &[bool; 64]) -> u64 {
    use std::arch::x86_64::__m512i;
    use std::arch::x86_64::_mm512_loadu_si512;
    use std::arch::x86_64::_mm512_test_epi8_mask;

    // SAFETY: the 64-byte load covers exactly the `[bool; 64]` array.
    let chunk = unsafe { _mm512_loadu_si512(bools.as_ptr().cast::<__m512i>()) };
    // Mask bit `i` is set iff byte `i` AND byte `i` is nonzero, i.e. iff `bools[i]`.
    _mm512_test_epi8_mask(chunk, chunk)
}

/// NEON byte→bit pack kernel: shift each `0x00`/`0x01` byte left by its bit position
/// (`ushl`), then fold the four vectors into one `u64` with a pairwise-add (`addp`) tree.
///
/// # Safety
///
/// The caller must ensure the CPU supports NEON (always true on aarch64).
#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
pub unsafe fn pack_bool_word_neon(bools: &[bool; 64]) -> u64 {
    use std::arch::aarch64::vgetq_lane_u64;
    use std::arch::aarch64::vld1q_s8;
    use std::arch::aarch64::vld1q_u8;
    use std::arch::aarch64::vpaddq_u8;
    use std::arch::aarch64::vreinterpretq_u64_u8;
    use std::arch::aarch64::vshlq_u8;

    const BIT_SHIFTS: [i8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7];

    let ptr = bools.as_ptr().cast::<u8>();
    // SAFETY: loading 16 constant bytes from `BIT_SHIFTS`; the four 16-byte data loads below are
    // all within the 64-byte array.
    unsafe {
        let shifts = vld1q_s8(BIT_SHIFTS.as_ptr());

        // Byte j of each vector becomes `bools[16v + j] << (j % 8)`.
        let m0 = vshlq_u8(vld1q_u8(ptr), shifts);
        let m1 = vshlq_u8(vld1q_u8(ptr.add(16)), shifts);
        let m2 = vshlq_u8(vld1q_u8(ptr.add(32)), shifts);
        let m3 = vshlq_u8(vld1q_u8(ptr.add(48)), shifts);

        // Three rounds of pairwise adds sum each group of 8 weighted bytes into one mask byte,
        // yielding the 8 mask bytes in order in the low half of the final vector.
        let sum01 = vpaddq_u8(m0, m1);
        let sum23 = vpaddq_u8(m2, m3);
        let sum = vpaddq_u8(sum01, sum23);
        let sum = vpaddq_u8(sum, sum);
        vgetq_lane_u64::<0>(vreinterpretq_u64_u8(sum))
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::collect_bool_word_scalar;
    use super::pack_bool_word_swar;

    fn patterns() -> Vec<[bool; 64]> {
        let mut patterns = vec![
            [false; 64],
            [true; 64],
            std::array::from_fn(|i| i % 2 == 0),
            std::array::from_fn(|i| i % 3 == 0),
            std::array::from_fn(|i| i < 32),
            std::array::from_fn(|i| i == 0 || i == 63),
        ];
        // A few deterministic pseudo-random patterns.
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        for _ in 0..8 {
            patterns.push(std::array::from_fn(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (state >> 33) & 1 == 1
            }));
        }
        patterns
    }

    fn reference(bools: &[bool; 64]) -> u64 {
        collect_bool_word_scalar(64, |i| bools[i])
    }

    #[test]
    fn swar_matches_scalar() {
        for bools in patterns() {
            assert_eq!(pack_bool_word_swar(&bools), reference(&bools));
        }
    }

    #[test]
    fn dispatch_matches_scalar() {
        for bools in patterns() {
            assert_eq!(
                crate::bit::collect_bool_word(64, |i| bools[i]),
                reference(&bools)
            );
        }
    }

    #[cfg(all(target_arch = "x86_64", not(miri)))]
    #[test]
    fn sse2_matches_scalar() {
        for bools in patterns() {
            // SAFETY: SSE2 is part of the x86-64 baseline instruction set.
            assert_eq!(
                unsafe { super::pack_bool_word_sse2(&bools) },
                reference(&bools)
            );
        }
    }

    #[cfg(all(target_arch = "x86_64", not(miri)))]
    #[test]
    fn avx2_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        for bools in patterns() {
            // SAFETY: runtime detection guarantees AVX2.
            assert_eq!(
                unsafe { super::pack_bool_word_avx2(&bools) },
                reference(&bools)
            );
        }
    }

    #[cfg(all(target_arch = "x86_64", not(miri)))]
    #[test]
    fn avx512_matches_scalar() {
        if !(is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw")) {
            return;
        }
        for bools in patterns() {
            // SAFETY: runtime detection guarantees AVX-512F and AVX-512BW.
            assert_eq!(
                unsafe { super::pack_bool_word_avx512(&bools) },
                reference(&bools)
            );
        }
    }

    #[cfg(all(target_arch = "aarch64", not(miri)))]
    #[test]
    fn neon_matches_scalar() {
        for bools in patterns() {
            // SAFETY: NEON is part of the aarch64 baseline instruction set.
            assert_eq!(
                unsafe { super::pack_bool_word_neon(&bools) },
                reference(&bools)
            );
        }
    }

    #[rstest]
    #[case(0)]
    #[case(1)]
    #[case(63)]
    #[case(64)]
    #[case(65)]
    #[case(200)]
    fn multiversioned_matches_inline(#[case] len: usize) {
        let pattern = |i: usize| i.is_multiple_of(3) || i.is_multiple_of(7);
        let num_words = len.div_ceil(64);
        let mut multiversioned = vec![0u64; num_words];
        super::collect_bool_words_multiversioned(&mut multiversioned, len, pattern);
        let mut inline = vec![0u64; num_words];
        super::collect_bool_words_inline(&mut inline, len, pattern);
        assert_eq!(multiversioned, inline);
    }

    #[rstest]
    #[case(0)]
    #[case(1)]
    #[case(5)]
    #[case(63)]
    #[case(64)]
    fn collect_bool_word_partial_lens_match(#[case] len: usize) {
        let expected = collect_bool_word_scalar(len, |i| i % 3 == 0);
        assert_eq!(crate::bit::collect_bool_word(len, |i| i % 3 == 0), expected);
    }
}
