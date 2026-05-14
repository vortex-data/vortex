// SPDX-FileCopyrightText: Copyright the Vortex contributors
// SPDX-License-Identifier: Apache-2.0

//! Funnel-shift compiler-vs-CPU isolation benchmark.
//!
//! The `matrix_run1.csv` cells `u64 W=51 ymm` show a stable +52% fused-FoR
//! overhead (bare 128 ns, fused 195 ns). ASM inspection reveals the bare
//! kernel emits `vpshldq` (the AVX-512-VBMI2 fused funnel-shift, available
//! in EVEX-256 form under `+avx512vbmi2 +avx512vl`) while the fused kernel
//! falls back to `vpsllq + vpsrlq + vpor + vpaddq`. This bench isolates
//! the question: is the +52% gap caused by a compiler pattern-matcher
//! failure (vpshldq disappears when its result is consumed by an add) or
//! by an intrinsic CPU limitation (vpshldq + vpaddq is throughput-limited
//! even when the compiler does emit it).
//!
//! Four variants per (W=51, W=63):
//!
//! 1. `baseline_macro_fused` -- existing macro-generated `FoR::unfor_pack`.
//! 2. `baseline_macro_bare`  -- existing macro-generated `BitPacking::unpack`.
//! 3. `hand_legacy`          -- hand-rolled intrinsics, vpsllq+vpsrlq+vpor+vpaddq.
//! 4. `hand_funnel`          -- hand-rolled intrinsics, vpshrdq+vpaddq.
//!
//! ## Faithfulness caveats
//!
//! The two `hand_*` variants are NOT a drop-in replacement for the real
//! FastLanes unpack at the given W. We deliberately fix a single shift
//! immediate per chunk (because `_mm256_shrdi_epi64` requires a const-generic
//! shift count) and we skip the FastLanes `FL_ORDER` lane interleaving
//! entirely. The purpose is to measure the inner-loop instruction-sequence
//! throughput, NOT to produce a byte-perfect decoded buffer. Both `hand_*`
//! variants traverse the same number of packed-input bytes and produce the
//! same number of output u64s as the real macro kernels, so memory pressure
//! and store throughput match.
//!
//! Build with `RUSTFLAGS="-C target-cpu=native -C target-feature=-prefer-256-bit"`
//! so the harness still picks zmm where it wants for the baseline benches,
//! while the `#[target_feature]` annotations on `hand_*` force EVEX-256.

#![allow(clippy::all)]

use std::hint::black_box;

use divan::Bencher;
use fastlanes_kernel_bench::BitPacking;
use fastlanes_kernel_bench::FastLanes;
use fastlanes_kernel_bench::FoR;

fn main() {
    divan::main();
}

const REF_U64: u64 = 1_000_000_007;

// ---------------------------------------------------------------------------
// Hand-rolled kernels: legacy 3-instruction funnel emulation.
// ---------------------------------------------------------------------------
//
// One 256-bit ymm holds 4 u64s. The "funnel" combines packed[i] (low) with
// packed[i+4] (high) under a per-chunk shift count K, producing 4 decoded
// output u64s before masking and FoR-adding. We pick K such that the
// high-word contribution is NOT entirely outside the W-bit mask, otherwise
// LLVM constant-folds the funnel into a plain right-shift. The condition is
// `K > 64 - W`; we pick K=20 for W=51 (so `hi << 44` contributes bits 44..,
// of which mask preserves 44..51) and K=10 for W=63 (so `hi << 54`
// contributes bits 54..63 to the masked result). The real kernel cycles
// through every K in [0..W); we hold K fixed because the intrinsic shift
// count must be const-generic. The instruction throughput per chunk is
// representative.

// Note: we use `core::arch::asm!` for the inner kernel of the legacy variants
// because LLVM aggressively combines `vpsrlq + vpsllq + vpor` into the very
// `vpshldq` we are trying to compare against. Inline assembly is opaque to
// LLVM and forces the literal instruction sequence we want to measure.

#[target_feature(enable = "avx2,avx512f,avx512vl,avx512vbmi2,bmi2")]
unsafe fn hand_legacy_w51(packed: &[u64; 816], reference: u64, out: &mut [u64; 1024]) {
    use core::arch::asm;
    use core::arch::x86_64::*;
    const K: i32 = 20;
    let ref_v = _mm256_set1_epi64x(reference as i64);
    let mask_v = _mm256_set1_epi64x((((1u128 << 51) - 1) as u64) as i64);
    // 256 chunks * 4 u64 outputs = 1024 outputs. The packed buffer is 816 u64s,
    // so we modulo to keep the load bases inside [0, 808].
    for chunk in 0..256usize {
        let base = (chunk * 3) % (816 - 8);
        let lo_ptr = packed.as_ptr().add(base);
        let hi_ptr = packed.as_ptr().add(base + 4);
        let out_ptr = out.as_mut_ptr().add(chunk * 4);
        asm!(
            "vmovdqu {lo}, [{lop}]",
            "vmovdqu {hi}, [{hip}]",
            "vpsrlq {tmp}, {lo}, {k}",
            "vpsllq {hi}, {hi}, {k64}",
            "vpor   {tmp}, {tmp}, {hi}",
            "vpand  {tmp}, {tmp}, {mask}",
            "vpaddq {tmp}, {tmp}, {refv}",
            "vmovdqu [{outp}], {tmp}",
            lop  = in(reg) lo_ptr,
            hip  = in(reg) hi_ptr,
            outp = in(reg) out_ptr,
            mask = in(ymm_reg) mask_v,
            refv = in(ymm_reg) ref_v,
            lo   = out(ymm_reg) _,
            hi   = out(ymm_reg) _,
            tmp  = out(ymm_reg) _,
            k    = const K,
            k64  = const 64 - K,
            options(nostack),
        );
    }
}

#[target_feature(enable = "avx2,avx512f,avx512vl,avx512vbmi2,bmi2")]
unsafe fn hand_funnel_w51(packed: &[u64; 816], reference: u64, out: &mut [u64; 1024]) {
    use core::arch::asm;
    use core::arch::x86_64::*;
    const K: i32 = 20;
    let ref_v = _mm256_set1_epi64x(reference as i64);
    let mask_v = _mm256_set1_epi64x((((1u128 << 51) - 1) as u64) as i64);
    // Use inline asm so the comparison vs `hand_legacy_w51` is at parity:
    // identical loop structure, identical register pressure, the only
    // difference is the instruction sequence under test.
    for chunk in 0..256usize {
        let base = (chunk * 3) % (816 - 8);
        let lo_ptr = packed.as_ptr().add(base);
        let hi_ptr = packed.as_ptr().add(base + 4);
        let out_ptr = out.as_mut_ptr().add(chunk * 4);
        asm!(
            "vmovdqu {lo}, [{lop}]",
            "vmovdqu {hi}, [{hip}]",
            // vpshrdq {tmp}, {lo}, {hi}, K  -- funnel shift right immediate.
            // Equivalent to ((hi:lo) >> K)[63:0].
            "vpshrdq {tmp}, {lo}, {hi}, {k}",
            "vpand   {tmp}, {tmp}, {mask}",
            "vpaddq  {tmp}, {tmp}, {refv}",
            "vmovdqu [{outp}], {tmp}",
            lop  = in(reg) lo_ptr,
            hip  = in(reg) hi_ptr,
            outp = in(reg) out_ptr,
            mask = in(ymm_reg) mask_v,
            refv = in(ymm_reg) ref_v,
            lo   = out(ymm_reg) _,
            hi   = out(ymm_reg) _,
            tmp  = out(ymm_reg) _,
            k    = const K,
            options(nostack),
        );
    }
}

#[target_feature(enable = "avx2,avx512f,avx512vl,avx512vbmi2,bmi2")]
unsafe fn hand_legacy_w63(packed: &[u64; 1008], reference: u64, out: &mut [u64; 1024]) {
    use core::arch::asm;
    use core::arch::x86_64::*;
    const K: i32 = 10;
    let ref_v = _mm256_set1_epi64x(reference as i64);
    let mask_v = _mm256_set1_epi64x((((1u128 << 63) - 1) as u64) as i64);
    for chunk in 0..256usize {
        let base = (chunk * 3) % (1008 - 8);
        let lo_ptr = packed.as_ptr().add(base);
        let hi_ptr = packed.as_ptr().add(base + 4);
        let out_ptr = out.as_mut_ptr().add(chunk * 4);
        asm!(
            "vmovdqu {lo}, [{lop}]",
            "vmovdqu {hi}, [{hip}]",
            "vpsrlq {tmp}, {lo}, {k}",
            "vpsllq {hi}, {hi}, {k64}",
            "vpor   {tmp}, {tmp}, {hi}",
            "vpand  {tmp}, {tmp}, {mask}",
            "vpaddq {tmp}, {tmp}, {refv}",
            "vmovdqu [{outp}], {tmp}",
            lop  = in(reg) lo_ptr,
            hip  = in(reg) hi_ptr,
            outp = in(reg) out_ptr,
            mask = in(ymm_reg) mask_v,
            refv = in(ymm_reg) ref_v,
            lo   = out(ymm_reg) _,
            hi   = out(ymm_reg) _,
            tmp  = out(ymm_reg) _,
            k    = const K,
            k64  = const 64 - K,
            options(nostack),
        );
    }
}

#[target_feature(enable = "avx2,avx512f,avx512vl,avx512vbmi2,bmi2")]
unsafe fn hand_funnel_w63(packed: &[u64; 1008], reference: u64, out: &mut [u64; 1024]) {
    use core::arch::asm;
    use core::arch::x86_64::*;
    const K: i32 = 10;
    let ref_v = _mm256_set1_epi64x(reference as i64);
    let mask_v = _mm256_set1_epi64x((((1u128 << 63) - 1) as u64) as i64);
    for chunk in 0..256usize {
        let base = (chunk * 3) % (1008 - 8);
        let lo_ptr = packed.as_ptr().add(base);
        let hi_ptr = packed.as_ptr().add(base + 4);
        let out_ptr = out.as_mut_ptr().add(chunk * 4);
        asm!(
            "vmovdqu {lo}, [{lop}]",
            "vmovdqu {hi}, [{hip}]",
            "vpshrdq {tmp}, {lo}, {hi}, {k}",
            "vpand   {tmp}, {tmp}, {mask}",
            "vpaddq  {tmp}, {tmp}, {refv}",
            "vmovdqu [{outp}], {tmp}",
            lop  = in(reg) lo_ptr,
            hip  = in(reg) hi_ptr,
            outp = in(reg) out_ptr,
            mask = in(ymm_reg) mask_v,
            refv = in(ymm_reg) ref_v,
            lo   = out(ymm_reg) _,
            hi   = out(ymm_reg) _,
            tmp  = out(ymm_reg) _,
            k    = const K,
            options(nostack),
        );
    }
}

// ---------------------------------------------------------------------------
// Divan bench definitions. Buffers allocated outside the closure.
// ---------------------------------------------------------------------------

#[divan::bench]
fn baseline_macro_fused__u64__w51(bencher: Bencher) {
    const W: usize = 51;
    const B: usize = 1024 * W / <u64>::T;
    let mut input = [0u64; 1024];
    for (i, v) in input.iter_mut().enumerate() {
        *v = i as u64;
    }
    let reference: u64 = REF_U64;
    let mut packed = [0u64; B];
    <u64 as FoR>::for_pack::<W, B>(&input, reference, &mut packed);
    let mut output = [0u64; 1024];

    bencher.bench_local(|| {
        <u64 as FoR>::unfor_pack::<W, B>(black_box(&packed), reference, &mut output);
        black_box(&mut output);
    });
}

#[divan::bench]
fn baseline_macro_bare__u64__w51(bencher: Bencher) {
    const W: usize = 51;
    const B: usize = 1024 * W / <u64>::T;
    let mut input = [0u64; 1024];
    for (i, v) in input.iter_mut().enumerate() {
        *v = i as u64;
    }
    let mut packed = [0u64; B];
    <u64 as BitPacking>::pack::<W, B>(&input, &mut packed);
    let mut output = [0u64; 1024];

    bencher.bench_local(|| {
        <u64 as BitPacking>::unpack::<W, B>(black_box(&packed), &mut output);
        black_box(&mut output);
    });
}

#[divan::bench]
fn hand_legacy__u64__w51(bencher: Bencher) {
    const W: usize = 51;
    const B: usize = 1024 * W / <u64>::T;
    let mut input = [0u64; 1024];
    for (i, v) in input.iter_mut().enumerate() {
        *v = i as u64;
    }
    let reference: u64 = REF_U64;
    let mut packed = [0u64; B];
    <u64 as FoR>::for_pack::<W, B>(&input, reference, &mut packed);
    let mut output = [0u64; 1024];

    bencher.bench_local(|| {
        unsafe {
            hand_legacy_w51(black_box(&packed), reference, &mut output);
        }
        black_box(&mut output);
    });
}

#[divan::bench]
fn hand_funnel__u64__w51(bencher: Bencher) {
    const W: usize = 51;
    const B: usize = 1024 * W / <u64>::T;
    let mut input = [0u64; 1024];
    for (i, v) in input.iter_mut().enumerate() {
        *v = i as u64;
    }
    let reference: u64 = REF_U64;
    let mut packed = [0u64; B];
    <u64 as FoR>::for_pack::<W, B>(&input, reference, &mut packed);
    let mut output = [0u64; 1024];

    bencher.bench_local(|| {
        unsafe {
            hand_funnel_w51(black_box(&packed), reference, &mut output);
        }
        black_box(&mut output);
    });
}

#[divan::bench]
fn baseline_macro_fused__u64__w63(bencher: Bencher) {
    const W: usize = 63;
    const B: usize = 1024 * W / <u64>::T;
    let mut input = [0u64; 1024];
    for (i, v) in input.iter_mut().enumerate() {
        *v = i as u64;
    }
    let reference: u64 = REF_U64;
    let mut packed = [0u64; B];
    <u64 as FoR>::for_pack::<W, B>(&input, reference, &mut packed);
    let mut output = [0u64; 1024];

    bencher.bench_local(|| {
        <u64 as FoR>::unfor_pack::<W, B>(black_box(&packed), reference, &mut output);
        black_box(&mut output);
    });
}

#[divan::bench]
fn baseline_macro_bare__u64__w63(bencher: Bencher) {
    const W: usize = 63;
    const B: usize = 1024 * W / <u64>::T;
    let mut input = [0u64; 1024];
    for (i, v) in input.iter_mut().enumerate() {
        *v = i as u64;
    }
    let mut packed = [0u64; B];
    <u64 as BitPacking>::pack::<W, B>(&input, &mut packed);
    let mut output = [0u64; 1024];

    bencher.bench_local(|| {
        <u64 as BitPacking>::unpack::<W, B>(black_box(&packed), &mut output);
        black_box(&mut output);
    });
}

#[divan::bench]
fn hand_legacy__u64__w63(bencher: Bencher) {
    const W: usize = 63;
    const B: usize = 1024 * W / <u64>::T;
    let mut input = [0u64; 1024];
    for (i, v) in input.iter_mut().enumerate() {
        *v = i as u64;
    }
    let reference: u64 = REF_U64;
    let mut packed = [0u64; B];
    <u64 as FoR>::for_pack::<W, B>(&input, reference, &mut packed);
    let mut output = [0u64; 1024];

    bencher.bench_local(|| {
        unsafe {
            hand_legacy_w63(black_box(&packed), reference, &mut output);
        }
        black_box(&mut output);
    });
}

#[divan::bench]
fn hand_funnel__u64__w63(bencher: Bencher) {
    const W: usize = 63;
    const B: usize = 1024 * W / <u64>::T;
    let mut input = [0u64; 1024];
    for (i, v) in input.iter_mut().enumerate() {
        *v = i as u64;
    }
    let reference: u64 = REF_U64;
    let mut packed = [0u64; B];
    <u64 as FoR>::for_pack::<W, B>(&input, reference, &mut packed);
    let mut output = [0u64; 1024];

    bencher.bench_local(|| {
        unsafe {
            hand_funnel_w63(black_box(&packed), reference, &mut output);
        }
        black_box(&mut output);
    });
}
