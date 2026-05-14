// SPDX-FileCopyrightText: Copyright the Vortex contributors
// SPDX-License-Identifier: Apache-2.0

//! Source-pattern experiment: which plain-safe-Rust phrasing of the FoR-fused
//! unpack body causes LLVM to emit `vpshldq + vpand + vpaddq` (the funnel-shift
//! sequence) instead of the legacy `vpsllq + vpsrlq + vpor + vpand + vpaddq`?
//!
//! All variants decode the same fixed cell -- u64, W=51 -- producing 1024
//! u64 outputs from an 816 u64 packed input, with a `wrapping_add(reference)`
//! folded into the inner loop. No `unsafe`, no `core::arch` intrinsics.
//!
//! Build with `RUSTFLAGS="-C target-cpu=native -C target-feature=-prefer-256-bit"`
//! to make `vpshldq` (AVX-512-VBMI2, EVEX-256) available without forcing zmm.

#![allow(clippy::all)]

use std::hint::black_box;

use divan::Bencher;

fn main() {
    divan::main();
}

const W: usize = 51;
const MASK: u64 = (1u64 << W) - 1;
const PACKED_LEN: usize = 1024 * W / 64; // 816
const REF_U64: u64 = 1_000_000_007;

// ---------------------------------------------------------------------------
// Variant 1: pat_macro_shape -- baseline matching the upstream macro.
// Mask THEN combine. This is the IR LLVM currently sees and fails on.
// ---------------------------------------------------------------------------
#[inline(never)]
fn pat_macro_shape(packed: &[u64; PACKED_LEN], reference: u64, out: &mut [u64; 1024]) {
    for i in 0..1024 {
        let bit_pos = i * W;
        let word = bit_pos / 64;
        let shift = (bit_pos % 64) as u32;
        let take_lo = (64 - shift).min(W as u32);
        let lo = (packed[word] >> shift) & ((1u64 << take_lo) - 1);
        let val = if take_lo as usize == W {
            lo
        } else {
            let hi = packed[word + 1] & ((1u64 << (W as u32 - take_lo)) - 1);
            lo | (hi << take_lo)
        };
        out[i] = val.wrapping_add(reference);
    }
}

// ---------------------------------------------------------------------------
// Variant 2: pat_combine_then_mask -- combine first, mask after.
// ---------------------------------------------------------------------------
#[inline(never)]
fn pat_combine_then_mask(packed: &[u64; PACKED_LEN], reference: u64, out: &mut [u64; 1024]) {
    for i in 0..1024 {
        let bit_pos = i * W;
        let word = bit_pos / 64;
        let shift = (bit_pos % 64) as u32;
        let val = if shift as usize + W <= 64 {
            (packed[word] >> shift) & MASK
        } else {
            let lo = packed[word] >> shift;
            let hi = packed[word + 1] << (64 - shift);
            (lo | hi) & MASK
        };
        out[i] = val.wrapping_add(reference);
    }
}

// ---------------------------------------------------------------------------
// Variant 3: pat_branchless_funnel -- always do the funnel-shift, no branch.
// NOTE: contains a known correctness bug at `shift == 0`, where
// `wrapping_shl(64)` is identity rather than zero. For W=51 across i in
// 0..1024 the offsets `i*51 % 64` produce shift==0 only at i==0, so a single
// element will be wrong; that does not affect the codegen-shape question.
// ---------------------------------------------------------------------------
#[inline(never)]
fn pat_branchless_funnel(packed: &[u64; PACKED_LEN], reference: u64, out: &mut [u64; 1024]) {
    for i in 0..1024 {
        let bit_pos = i * W;
        let word = bit_pos / 64;
        let next = (word + 1).min(PACKED_LEN - 1);
        let shift = (bit_pos % 64) as u32;
        let lo = packed[word] >> shift;
        let hi = packed[next].wrapping_shl(64u32.wrapping_sub(shift));
        let val = (lo | hi) & MASK;
        out[i] = val.wrapping_add(reference);
    }
}

// ---------------------------------------------------------------------------
// Variant 4: pat_u128_cat -- catenate via u128, shift, truncate.
// ---------------------------------------------------------------------------
#[inline(never)]
fn pat_u128_cat(packed: &[u64; PACKED_LEN], reference: u64, out: &mut [u64; 1024]) {
    for i in 0..1024 {
        let bit_pos = i * W;
        let word = bit_pos / 64;
        let shift = (bit_pos % 64) as u32;
        let next = (word + 1).min(PACKED_LEN - 1);
        let combined = (packed[word] as u128) | ((packed[next] as u128) << 64);
        let val = ((combined >> shift) as u64) & MASK;
        out[i] = val.wrapping_add(reference);
    }
}

// ---------------------------------------------------------------------------
// Variant 5: pat_u128_cat_unrolled4 -- as variant 4, manually unrolled 4x.
// ---------------------------------------------------------------------------
#[inline(never)]
fn pat_u128_cat_unrolled4(packed: &[u64; PACKED_LEN], reference: u64, out: &mut [u64; 1024]) {
    let mut i = 0;
    while i + 3 < 1024 {
        for k in 0..4 {
            let bit_pos = (i + k) * W;
            let word = bit_pos / 64;
            let shift = (bit_pos % 64) as u32;
            let next = (word + 1).min(PACKED_LEN - 1);
            let combined = (packed[word] as u128) | ((packed[next] as u128) << 64);
            let val = ((combined >> shift) as u64) & MASK;
            out[i + k] = val.wrapping_add(reference);
        }
        i += 4;
    }
    // Tail
    while i < 1024 {
        let bit_pos = i * W;
        let word = bit_pos / 64;
        let shift = (bit_pos % 64) as u32;
        let next = (word + 1).min(PACKED_LEN - 1);
        let combined = (packed[word] as u128) | ((packed[next] as u128) << 64);
        let val = ((combined >> shift) as u64) & MASK;
        out[i] = val.wrapping_add(reference);
        i += 1;
    }
}

// ---------------------------------------------------------------------------
// Variant 6: pat_chunked_aligned -- process in groups where shift is constant.
// Mimics the FastLanes lane-major layout: 16 lanes x 64 rows.
// ---------------------------------------------------------------------------
#[inline(never)]
fn pat_chunked_aligned(packed: &[u64; PACKED_LEN], reference: u64, out: &mut [u64; 1024]) {
    const LANES: usize = 16; // 1024 / 64
    for lane in 0..LANES {
        for row in 0..64 {
            let bit_pos = (row * LANES + lane) * W;
            let word = bit_pos / 64;
            let shift = (bit_pos % 64) as u32;
            let next = (word + 1).min(PACKED_LEN - 1);
            let lo = packed[word] >> shift;
            let hi = packed[next] << (64 - shift);
            let val = (lo | hi) & MASK;
            out[row * LANES + lane] = val.wrapping_add(reference);
        }
    }
}

// ---------------------------------------------------------------------------
// Bench harness: allocate buffers outside the closure (matching the existing
// style in funnel_shift_fix.rs and multi_block.rs).
// ---------------------------------------------------------------------------

fn make_packed() -> [u64; PACKED_LEN] {
    // Fill with non-trivial bit pattern so LLVM cannot constant-fold loads.
    let mut p = [0u64; PACKED_LEN];
    for (i, v) in p.iter_mut().enumerate() {
        *v = (i as u64).wrapping_mul(0x9E3779B97F4A7C15);
    }
    p
}

#[divan::bench]
fn pat_macro_shape__u64__w51(bencher: Bencher) {
    let packed = make_packed();
    let reference = REF_U64;
    let mut output = [0u64; 1024];
    bencher.bench_local(|| {
        pat_macro_shape(black_box(&packed), reference, &mut output);
        black_box(&mut output);
    });
}

#[divan::bench]
fn pat_combine_then_mask__u64__w51(bencher: Bencher) {
    let packed = make_packed();
    let reference = REF_U64;
    let mut output = [0u64; 1024];
    bencher.bench_local(|| {
        pat_combine_then_mask(black_box(&packed), reference, &mut output);
        black_box(&mut output);
    });
}

#[divan::bench]
fn pat_branchless_funnel__u64__w51(bencher: Bencher) {
    let packed = make_packed();
    let reference = REF_U64;
    let mut output = [0u64; 1024];
    bencher.bench_local(|| {
        pat_branchless_funnel(black_box(&packed), reference, &mut output);
        black_box(&mut output);
    });
}

#[divan::bench]
fn pat_u128_cat__u64__w51(bencher: Bencher) {
    let packed = make_packed();
    let reference = REF_U64;
    let mut output = [0u64; 1024];
    bencher.bench_local(|| {
        pat_u128_cat(black_box(&packed), reference, &mut output);
        black_box(&mut output);
    });
}

#[divan::bench]
fn pat_u128_cat_unrolled4__u64__w51(bencher: Bencher) {
    let packed = make_packed();
    let reference = REF_U64;
    let mut output = [0u64; 1024];
    bencher.bench_local(|| {
        pat_u128_cat_unrolled4(black_box(&packed), reference, &mut output);
        black_box(&mut output);
    });
}

#[divan::bench]
fn pat_chunked_aligned__u64__w51(bencher: Bencher) {
    let packed = make_packed();
    let reference = REF_U64;
    let mut output = [0u64; 1024];
    bencher.bench_local(|| {
        pat_chunked_aligned(black_box(&packed), reference, &mut output);
        black_box(&mut output);
    });
}
