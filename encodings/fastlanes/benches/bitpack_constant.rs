// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compare strategies for FastLanes-bitpacking a constant value across one or more
//! 1024-element blocks, plus strategies for `bitpacked_block == constant` eq compare.
//!
//! The output layout for `BitPacking::pack` is `[row_0 x LANES, row_1 x LANES, ..., row_{W-1} x LANES]`,
//! and for a constant input every value within a single row is identical. That gives
//! several ways to produce the packed bytes:
//!
//!   1. `heap_buf_pack` - allocate `vec![value; 1024]`, run `unchecked_pack`.
//!      (Matches what `bitpack_primitive` does today when fed a constant.)
//!   2. `stack_buf_pack` - place a `[value; 1024]` on the stack and run `unchecked_pack`.
//!   3. `stack_buf_pack_const` - same as 2 but with `pack::<W, B>` so the width is known
//!      at compile time.
//!   4. `compute_rows_splat` - compute the `W` row words directly with shift/or, then
//!      splat each row across `LANES` positions. No 1024-element intermediate.
//!
//! For `eq <constant>` over a packed 1024-block we benchmark:
//!
//!   A. `block_eq_no_unpack` - keep the `W` constant-row words; for each lane, XOR
//!      packed-array words against the matching row word in registers, extract each
//!      element's W-bit field from the XOR and emit `bit = (W bits == 0)`. No
//!      `u32` value is ever materialized to memory.
//!   B. `heap_unpack_collect` - unpack the block into a heap `Vec<u32>(1024)` via
//!      `BitPacking::unchecked_unpack`, then `BitBufferMut::collect_bool`.
//!   C. `stack_unpack_collect` - unpack into a stack `[u32; 1024]` via the
//!      compile-time `BitPacking::pack`/`unpack::<W, B>`, then `collect_bool`.
//!   D. `fl_unpack_cmp_*` - use fastlanes' own `BitPackingCompare::unpack_cmp` which
//!      is the same `unpack!` macro driving a fused eq+bool-write kernel.
//!      Apples-to-apples codegen vs `unpack` (no extra target_feature, no intrinsics).
//!   E. `block_batch` / `block_batch_avx2` / `block_batch_avx512_hand` - SIMD eq
//!      kernels writing 16 `u64`s per block. The `*_avx512_hand` path uses explicit
//!      `_mm512_cmpeq_epu32_mask`; not apples-to-apples with `unpack`, kept as a
//!      ceiling reference.
//!
//! ## Findings (Intel Xeon, AVX-512 available, `-C target-cpu=native`)
//!
//! Throughput (median Gitem/s on u32, 64 blocks per call):
//! ```text
//!   W:                              1     4    12    16    24
//!   block_batch_avx512_v4         24.5 22.1 13.2 18.8 10.2  <- best (vpternlogd+precomp)
//!   block_batch_avx512_v5         23.0 21.7 13.1 18.8 10.2
//!   block_batch_avx512_v3         24.1 18.2 13.2 14.1  9.8
//!   block_batch_avx512_v2         19.4 16.4 13.2 14.3 10.8
//!   block_batch_avx512_hand       20.3 18.0 12.5 11.7  9.5
//!   block_unpack_avx512_hand      12.3 11.8 10.8 10.3  9.5
//!   stack_unpack_avx512            7.3  7.7  7.4  7.3  7.1
//!   stack_unpack_per1k             5.84 6.44 6.27 6.22 6.02 <- best auto-vec
//!   fl_unpack_cmp_per1k            5.60 6.04 6.01 4.91 6.07
//!   fl_unpack_cmp_batch            2.93 2.88 2.92 2.86 2.86
//!   fl_unpack_cmp_collect          2.58 2.59 2.49 2.66 2.52
//!   stack_unpack_collect           2.66 2.67 2.59 2.62 2.55
//!   block_batch (mine, auto-vec)   1.72 1.37 1.67 1.68 1.48
//! ```
//!
//! AVX-512 progression on splat-cmp algorithm (W=4 column):
//!   v1 hand:     18.0 Gitem/s  -- load + xor + and + cmpeq per chunk
//!   v2 testn:    16.4           -- testn fuses and+cmpeq (slower: testn lat 4 vs cmpeq 3)
//!   v3 precomp:  18.2           -- pre-broadcast c & w as per-row target
//!   v4 ternlog:  22.1 (+22%)    -- vpternlogd fuses xor+and into 1 op
//!   v5 OR-bdry:  21.7           -- v4 + OR-fused boundary (no help, more dep latency)
//!
//! v4 is the best: 24.5 Gitem/s = 98 GB/s of u32 input at W=1, ~22 Gitem/s for
//! W where W divides T=32 (1, 4, 16). Boundary-heavy widths (12, 24) are
//! load-throughput limited at ~13 / ~10 Gitem/s.
//!
//! Algorithm-vs-algorithm at the same SIMD level (both AVX-512 hand-tuned):
//! the **splat algorithm wins** by 1.4-1.6x. Both emit the same 4 ops per
//! lane per row (load + xor/srl + and + cmpeq), but `vpsrld` with a variable
//! `__m128i` shift count has worse port pressure than `vpxorq`, and the splat
//! algo keeps the W-bit constant pre-positioned in `const_row` words so the
//! "extract" step disappears.
//!
//! At SSE2 auto-vec the two algorithms FLIP: unpack-then-compare beats splat
//! (6.4 vs 1.4 Gitem/s) because every row of the splat algo pays a
//! `packssdw+packsswb+pmovmskb` horizontal bit-mask chain. With AVX-512's
//! `vpcmpeqd -> kreg` that chain collapses to one op and splat re-takes the lead.
//!
//! Fusion matters: `stack_unpack_avx512` uses the same AVX-512 cmpeq+kmask
//! compare but writes/re-reads the unpacked block, losing ~40% to the extra
//! buffer traffic.
//!
//! The `*_per1k` (no intrinsics) strategies are 2-2.4x faster than the same
//! kernel with per-block `BitBuffer` heap allocs because the stack scratch
//! stays in L1 between the unpack write and the inline collect_bool read.
//!
//! Code size per (W, T) specialization (bytes, u32, 32 W values):
//! ```text
//!   BitPacking::pack                 ~1025  ->  32.8 KB / T   (baseline)
//!   BitPacking::unpack                ~797  ->  25.5 KB / T   (baseline)
//!   BitPackingCompare::unpack_cmp    ~1049  ->  33.6 KB / T   (same instrs as unpack)
//!   block_eq_batch_avx512_hand        ~868  ->  27.8 KB / T   (smaller than unpack!)
//!   block_eq_collect_u64 (auto-vec)  ~3368  -> 107.8 KB / T   (4x bloat, SSE2 chain)
//! ```
//!
//! Key insight: my auto-vec `block_eq_collect_u64` is BOTH 4x bigger AND 2-3x slower
//! than `fl_unpack_cmp`, because SSE2 needs a `packssdw + packsswb + pmovmskb` chain
//! per 8 lanes to produce horizontal bit-mask output. AVX-512's `vpcmpeqd -> kreg`
//! collapses that into one instruction -- simultaneously 4x smaller and 6x faster.
//! Code size correlates with how cleanly the target ISA expresses the operation,
//! not with whether you used intrinsics.
//!
//! Run with `cargo bench -p vortex-fastlanes --bench bitpack_constant`.

#![expect(clippy::unwrap_used)]

use std::hint::black_box;

use divan::Bencher;
use divan::counter::ItemsCount;
use fastlanes::BitPacking;
use fastlanes::BitPackingCompare;
use fastlanes::FastLanes;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;

fn main() {
    validate_strategies();
    divan::main();
}

/// Run before benches: every strategy must produce identical packed bytes.
/// Panics if a strategy disagrees with `unchecked_pack` on a `[value; 1024]` input.
fn validate_strategies() {
    macro_rules! check {
        ($W:literal, $value:expr) => {{
            const W: usize = $W;
            const B: usize = 32 * W;
            let v: u32 = $value;

            let input = [v; 1024];
            let mut expected = vec![0u32; B];
            // SAFETY: fixed lengths satisfy `unchecked_pack` contract.
            unsafe { BitPacking::unchecked_pack(W, &input, &mut expected) };

            let mut a = vec![0u32; B];
            heap_buf_pack(v, W, &mut a);
            assert_eq!(a, expected, "heap_buf_pack mismatch for W={W}");

            let mut s = vec![0u32; B];
            stack_buf_pack(v, W, &mut s);
            assert_eq!(s, expected, "stack_buf_pack mismatch for W={W}");

            let mut c = [0u32; B];
            stack_buf_pack_const::<u32, W, B>(v, &mut c);
            assert_eq!(c.as_slice(), expected.as_slice(), "stack_buf_pack_const mismatch for W={W}");

            let mut d = [0u32; B];
            compute_rows_splat::<u32, W, B>(v, &mut d);
            assert_eq!(d.as_slice(), expected.as_slice(), "compute_rows_splat mismatch for W={W}");
        }};
    }

    check!(1, 1);
    check!(1, 0);
    check!(4, 0xA);
    check!(4, 0xF);
    check!(12, 0xABC);
    check!(16, 0xBEEF);
    check!(24, 0x12_3456);

    validate_eq_strategies();
}

/// Verify that all three `eq` strategies produce identical 1024-bit results on
/// a mixed (50% match) packed block. Panics on disagreement.
fn validate_eq_strategies() {
    macro_rules! check_eq {
        ($W:literal) => {{
            const W: usize = $W;
            const B: usize = 32 * W;
            let constant: u32 = 0b1010_1010_1010_1010 & if W == 32 { u32::MAX } else { (1u32 << W) - 1 };

            // Build a mixed input: half the elements equal `constant`, the rest
            // walk through other in-range values.
            let mut input = [0u32; 1024];
            for i in 0..1024 {
                input[i] = if i % 2 == 0 {
                    constant
                } else {
                    (i as u32) & if W == 32 { u32::MAX } else { (1u32 << W) - 1 }
                };
            }
            let mut packed = [0u32; B];
            BitPacking::pack::<W, B>(&input, &mut packed);

            // Reference: unpack + element compare.
            let mut unpacked = [0u32; 1024];
            BitPacking::unpack::<W, B>(&packed, &mut unpacked);
            let expected: Vec<bool> = (0..1024).map(|i| unpacked[i] == constant).collect();

            let const_rows = const_row_words_u32::<W>(constant);
            let a = block_eq_no_unpack_collect::<W, B>(&packed, &const_rows);
            let a2 = block_eq_simd_collect::<W, B>(&packed, &const_rows);
            let b = heap_unpack_collect(&packed, W, constant);
            let c = stack_unpack_collect::<W, B>(&packed, constant);

            // Batched variant: produces 16 u64s == 1024 bits per block.
            let mut batched = [0u64; 16];
            block_eq_collect_u64::<W, B>(&packed, &const_rows, &mut batched);

            // Per-1k shared-output variants: also 16 u64s per block.
            let mut fl_per1k = vec![0u64; 16];
            fastlanes_unpack_cmp_per1k_shared::<W, B>(
                std::slice::from_ref(&packed),
                constant,
                &mut fl_per1k,
            );
            let mut stack_per1k = vec![0u64; 16];
            stack_unpack_per1k_shared::<W, B>(
                std::slice::from_ref(&packed),
                constant,
                &mut stack_per1k,
            );

            for i in 0..1024 {
                assert_eq!(a.value(i), expected[i], "block_eq_no_unpack mismatch at i={i}, W={W}");
                assert_eq!(a2.value(i), expected[i], "block_eq_simd mismatch at i={i}, W={W}");
                assert_eq!(b.value(i), expected[i], "heap_unpack_collect mismatch at i={i}, W={W}");
                assert_eq!(c.value(i), expected[i], "stack_unpack_collect mismatch at i={i}, W={W}");
                let bit = (batched[i / 64] >> (i % 64)) & 1 != 0;
                assert_eq!(bit, expected[i], "block_eq_batch mismatch at i={i}, W={W}");
                let bit_fl = (fl_per1k[i / 64] >> (i % 64)) & 1 != 0;
                assert_eq!(bit_fl, expected[i], "fl_unpack_cmp_per1k mismatch at i={i}, W={W}");
                let bit_su = (stack_per1k[i / 64] >> (i % 64)) & 1 != 0;
                assert_eq!(bit_su, expected[i], "stack_unpack_per1k mismatch at i={i}, W={W}");
            }

            // AVX-512 hand-rolled path: only check when the running CPU supports
            // it (the validate step is also gated by `is_x86_feature_detected!`
            // since this runs on the host CPU).
            #[cfg(target_arch = "x86_64")]
            if is_x86_feature_detected!("avx512f")
                && is_x86_feature_detected!("avx512bw")
                && is_x86_feature_detected!("avx512dq")
            {
                let mut hand = vec![0u64; 16];
                // SAFETY: feature-gated by `is_x86_feature_detected!`.
                unsafe {
                    block_eq_batch_avx512_hand::<W, B>(
                        std::slice::from_ref(&packed),
                        &const_rows,
                        &mut hand,
                    )
                };
                for i in 0..1024 {
                    let bit = (hand[i / 64] >> (i % 64)) & 1 != 0;
                    assert_eq!(bit, expected[i], "avx512_hand mismatch at i={i}, W={W}");
                }

                let mut hand_v2 = vec![0u64; 16];
                unsafe {
                    block_eq_batch_avx512_v2::<W, B>(
                        std::slice::from_ref(&packed),
                        &const_rows,
                        &mut hand_v2,
                    )
                };
                for i in 0..1024 {
                    let bit = (hand_v2[i / 64] >> (i % 64)) & 1 != 0;
                    assert_eq!(bit, expected[i], "avx512_v2 mismatch at i={i}, W={W}");
                }

                let mut hand_v3 = vec![0u64; 16];
                unsafe {
                    block_eq_batch_avx512_v3::<W, B>(
                        std::slice::from_ref(&packed),
                        &const_rows,
                        &mut hand_v3,
                    )
                };
                for i in 0..1024 {
                    let bit = (hand_v3[i / 64] >> (i % 64)) & 1 != 0;
                    assert_eq!(bit, expected[i], "avx512_v3 mismatch at i={i}, W={W}");
                }

                let mut hand_v4 = vec![0u64; 16];
                unsafe {
                    block_eq_batch_avx512_v4::<W, B>(
                        std::slice::from_ref(&packed),
                        &const_rows,
                        &mut hand_v4,
                    )
                };
                for i in 0..1024 {
                    let bit = (hand_v4[i / 64] >> (i % 64)) & 1 != 0;
                    assert_eq!(bit, expected[i], "avx512_v4 mismatch at i={i}, W={W}");
                }

                let mut hand_v5 = vec![0u64; 16];
                unsafe {
                    block_eq_batch_avx512_v5::<W, B>(
                        std::slice::from_ref(&packed),
                        &const_rows,
                        &mut hand_v5,
                    )
                };
                for i in 0..1024 {
                    let bit = (hand_v5[i / 64] >> (i % 64)) & 1 != 0;
                    assert_eq!(bit, expected[i], "avx512_v5 mismatch at i={i}, W={W}");
                }
            }

            // AVX2 path (separate feature gate so it runs on more CPUs).
            #[cfg(target_arch = "x86_64")]
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("bmi2") {
                let mut hand_av2 = vec![0u64; 16];
                unsafe {
                    block_eq_batch_avx2_v4_u32::<W, B>(
                        std::slice::from_ref(&packed),
                        &const_rows,
                        &mut hand_av2,
                    )
                };
                for i in 0..1024 {
                    let bit = (hand_av2[i / 64] >> (i % 64)) & 1 != 0;
                    assert_eq!(bit, expected[i], "avx2_v4 mismatch at i={i}, W={W}");
                }

                let mut hand_un = vec![0u64; 16];
                unsafe {
                    block_eq_unpack_avx512_hand::<W, B>(
                        std::slice::from_ref(&packed),
                        constant,
                        &mut hand_un,
                    )
                };
                for i in 0..1024 {
                    let bit = (hand_un[i / 64] >> (i % 64)) & 1 != 0;
                    assert_eq!(bit, expected[i], "avx512_unpack mismatch at i={i}, W={W}");
                }

                let mut su_av = vec![0u64; 16];
                unsafe {
                    stack_unpack_avx512_collect::<W, B>(
                        std::slice::from_ref(&packed),
                        constant,
                        &mut su_av,
                    )
                };
                for i in 0..1024 {
                    let bit = (su_av[i / 64] >> (i % 64)) & 1 != 0;
                    assert_eq!(bit, expected[i], "stack_unpack_avx512 mismatch at i={i}, W={W}");
                }
            }
        }};
    }

    check_eq!(1);
    check_eq!(2);
    check_eq!(3);
    check_eq!(4);
    check_eq!(5);
    check_eq!(7);
    check_eq!(8);
    check_eq!(11);
    check_eq!(12);
    check_eq!(16);
    check_eq!(17);
    check_eq!(23);
    check_eq!(24);
    check_eq!(29);

    validate_v6();
}

/// Validate v6 W=8 and W=16 specializations against unpack reference.
#[cfg(target_arch = "x86_64")]
fn validate_v6() {
    if !is_x86_feature_detected!("avx512f")
        || !is_x86_feature_detected!("avx512bw")
        || !is_x86_feature_detected!("bmi2")
    {
        return;
    }

    // W=8
    {
        const W: usize = 8;
        const B: usize = 32 * W;
        let constant: u32 = 0xA5;
        let mut input = [0u32; 1024];
        for i in 0..1024 {
            input[i] = if i % 3 == 0 { constant } else { i as u32 & 0xFF };
        }
        let mut packed = [0u32; B];
        BitPacking::pack::<W, B>(&input, &mut packed);
        let mut expected = [0u32; 1024];
        BitPacking::unpack::<W, B>(&packed, &mut expected);
        let mut out = vec![0u64; 16];
        // SAFETY: feature-gated.
        unsafe {
            block_eq_batch_avx512_v6_w8::<B>(std::slice::from_ref(&packed), constant, &mut out);
        }
        for i in 0..1024 {
            let bit = (out[i / 64] >> (i % 64)) & 1 != 0;
            assert_eq!(bit, expected[i] == constant, "v6_w8 mismatch at i={i}");
        }
    }

    // W=16
    {
        const W: usize = 16;
        const B: usize = 32 * W;
        let constant: u32 = 0xBEEF;
        let mut input = [0u32; 1024];
        for i in 0..1024 {
            input[i] = if i % 3 == 0 { constant } else { i as u32 & 0xFFFF };
        }
        let mut packed = [0u32; B];
        BitPacking::pack::<W, B>(&input, &mut packed);
        let mut expected = [0u32; 1024];
        BitPacking::unpack::<W, B>(&packed, &mut expected);
        let mut out = vec![0u64; 16];
        unsafe {
            block_eq_batch_avx512_v6_w16::<B>(std::slice::from_ref(&packed), constant, &mut out);
        }
        for i in 0..1024 {
            let bit = (out[i / 64] >> (i % 64)) & 1 != 0;
            assert_eq!(bit, expected[i] == constant, "v6_w16 mismatch at i={i}");
        }
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn validate_v6() {}

// Number of 1024-element FastLanes blocks per benchmark iteration. 64 blocks ~= 64Ki
// constants per call which is large enough to make per-call overhead negligible.
const BLOCKS: usize = 64;

// ---------------------------------------------------------------------------
// Strategies
// ---------------------------------------------------------------------------

/// Strategy 1: allocate a fresh `vec![value; 1024]` and call `unchecked_pack`.
///
/// This is what the current `bitpack_primitive` path executes when handed a constant
/// slice -- the slice itself is already materialized in memory.
#[inline]
fn heap_buf_pack<T: BitPacking + Copy>(value: T, width: usize, out: &mut [T]) {
    let input: Vec<T> = vec![value; 1024];
    // SAFETY: `input` has length 1024 and `out` has length `128 * width / size_of::<T>()`.
    unsafe { BitPacking::unchecked_pack(width, &input, out) };
}

/// Strategy 2: stack-resident input array, runtime-known width.
#[inline]
fn stack_buf_pack<T: BitPacking + Copy>(value: T, width: usize, out: &mut [T]) {
    let input: [T; 1024] = [value; 1024];
    // SAFETY: see `heap_buf_pack`.
    unsafe { BitPacking::unchecked_pack(width, &input, out) };
}

/// Strategy 3: stack-resident input array, compile-time-known width.
#[inline]
fn stack_buf_pack_const<T: BitPacking + Copy, const W: usize, const B: usize>(
    value: T,
    out: &mut [T; B],
) {
    let input: [T; 1024] = [value; 1024];
    BitPacking::pack::<W, B>(&input, out);
}

/// Strategy 4: directly compute the `W` row words by reproducing the shift/or kernel
/// once, then splat each row across `LANES` positions. No 1024-element intermediate.
#[inline]
fn compute_rows_splat<T, const W: usize, const B: usize>(value: T, out: &mut [T; B])
where
    T: FastLanes + Copy,
{
    let t_bits = T::T;
    let lanes = T::LANES;
    debug_assert!(W > 0);
    debug_assert!(W <= t_bits);
    debug_assert_eq!(B, lanes * W);

    let mask = (T::one() << W) - T::one();
    let src = value & mask;

    // Row words for a single lane (every lane produces the same W words for a
    // constant input). Worst case `W == t_bits == 64`, so a fixed-size buffer is fine.
    let mut row_words = [T::zero(); 64];
    let mut tmp = src;

    for row in 0..t_bits {
        if row != 0 {
            tmp = tmp | (src << ((row * W) % t_bits));
        }
        let curr_word = (row * W) / t_bits;
        let next_word = ((row + 1) * W) / t_bits;
        if next_word > curr_word {
            row_words[curr_word] = tmp;
            let remaining_bits = ((row + 1) * W) % t_bits;
            // `W - remaining_bits` is in [0, T-1] except when remaining_bits == 0,
            // in which case we're writing the last word for this lane and the
            // carry is unused. Guard the shift to avoid UB when `W == T` and
            // `remaining_bits == 0`.
            tmp = if remaining_bits == 0 {
                T::zero()
            } else {
                src >> (W - remaining_bits)
            };
        }
    }

    for row in 0..W {
        out[row * lanes..(row + 1) * lanes].fill(row_words[row]);
    }
}

// ---------------------------------------------------------------------------
// Eq-compare strategies: 1024 bitpacked u32 values vs a scalar constant.
// Each strategy emits 1024 bits in *logical* input order via `collect_bool`,
// so callers can append into a larger `BitBufferMut`.
// ---------------------------------------------------------------------------

/// FastLanes transposed-input index function for `T = u32`. Maps a (row, lane)
/// in the packed-walk order back to a logical input position in `0..1024`.
#[inline(always)]
fn idx_u32(row: usize, lane: usize) -> usize {
    const FL_ORDER: [usize; 8] = [0, 4, 2, 6, 1, 5, 3, 7];
    let o = row / 8;
    let s = row % 8;
    FL_ORDER[o] * 16 + s * 128 + lane
}

/// Compute the `W` packed row words for a single lane of a constant `value`.
/// (`compute_rows_splat` does this then fans out across `LANES`; the eq path
/// only needs the per-lane row words since every lane shares the same ones.)
#[inline(always)]
fn const_row_words_u32<const W: usize>(value: u32) -> [u32; 32] {
    let mut row_words = [0u32; 32];
    debug_assert!(W > 0 && W <= 32);
    let mask: u32 = if W == 32 { u32::MAX } else { (1u32 << W) - 1 };
    let src = value & mask;
    let mut tmp = src;
    for row in 0..32usize {
        if row != 0 {
            tmp |= src << ((row * W) % 32);
        }
        let curr_word = (row * W) / 32;
        let next_word = ((row + 1) * W) / 32;
        if next_word > curr_word {
            row_words[curr_word] = tmp;
            let remaining_bits = ((row + 1) * W) % 32;
            tmp = if remaining_bits == 0 {
                0
            } else {
                src >> (W - remaining_bits)
            };
        }
    }
    row_words
}

/// Strategy A: block-block eq with no value materialization.
///
/// For each lane, XOR each packed-array word with the matching constant row
/// word in a register, extract every element's W-bit field from the XOR and
/// emit `1` iff those W bits are all zero. Output is written into a logical-
/// order `[u8; 1024]` scratch so the final `collect_bool` is a straight scan.
#[inline]
fn block_eq_no_unpack<const W: usize, const B: usize>(
    packed: &[u32; B],
    const_rows: &[u32; 32],
    out: &mut [u8; 1024],
) {
    const T: usize = 32;
    const LANES: usize = 32;
    debug_assert!(W > 0 && W <= T);
    debug_assert_eq!(B, LANES * W);

    let mask: u32 = if W == 32 { u32::MAX } else { (1u32 << W) - 1 };

    for lane in 0..LANES {
        let mut src_xor: u32 = packed[lane] ^ const_rows[0];

        for row in 0..T {
            let curr_word = (row * W) / T;
            let next_word = ((row + 1) * W) / T;
            let shift = (row * W) % T;

            let diff: u32 = if next_word > curr_word {
                let remaining_bits = ((row + 1) * W) % T;
                let current_bits = W - remaining_bits;
                let cur_mask = if current_bits == 0 { 0 } else { (1u32 << current_bits) - 1 };
                let lo = (src_xor >> shift) & cur_mask;
                if next_word < W {
                    src_xor = packed[LANES * next_word + lane] ^ const_rows[next_word];
                    let rem_mask = if remaining_bits == 0 { 0 } else { (1u32 << remaining_bits) - 1 };
                    lo | ((src_xor & rem_mask) << current_bits)
                } else {
                    lo
                }
            } else {
                (src_xor >> shift) & mask
            };

            out[idx_u32(row, lane)] = (diff == 0) as u8;
        }
    }
}

/// Strategy A wrapper: produce a `BitBuffer` of 1024 bits via `collect_bool`.
#[inline]
fn block_eq_no_unpack_collect<const W: usize, const B: usize>(
    packed: &[u32; B],
    const_rows: &[u32; 32],
) -> BitBuffer {
    let mut scratch = [0u8; 1024];
    block_eq_no_unpack::<W, B>(packed, const_rows, &mut scratch);
    BitBufferMut::collect_bool(1024, |i| scratch[i] != 0).freeze()
}

/// FastLanes output position k => row index, for u32. Each row's 32-bit mask
/// goes at base = k * 32 = `FL_ORDER[row/8] * 16 + (row%8) * 128`.
/// Two adjacent k values share one `u64` in the bit buffer (k/2 = u64 index).
const ROW_FOR_K_U32: [usize; 32] = [
    0, 16, 8, 24, 1, 17, 9, 25, 2, 18, 10, 26, 3, 19, 11, 27, 4, 20, 12, 28, 5, 21, 13, 29, 6, 22,
    14, 30, 7, 23, 15, 31,
];

/// Compress 32 u32 "is-zero" indicators into a single `u32` bitmask.
///
/// Split into 4 groups of 8 lanes; each 8-lane group matches `cmpeq + movemask_ps`
/// on x86-64, so the whole function should lower to ~4 AVX2 ops or 1 AVX-512 op.
/// `bit lane = 1` iff `zeros[lane] == 0`.
#[inline(always)]
fn movemask32(zeros: &[u32; 32]) -> u32 {
    let mut bytes = [0u8; 4];
    for c in 0..4 {
        let base = c * 8;
        let b: u8 = ((zeros[base] == 0) as u8)
            | (((zeros[base + 1] == 0) as u8) << 1)
            | (((zeros[base + 2] == 0) as u8) << 2)
            | (((zeros[base + 3] == 0) as u8) << 3)
            | (((zeros[base + 4] == 0) as u8) << 4)
            | (((zeros[base + 5] == 0) as u8) << 5)
            | (((zeros[base + 6] == 0) as u8) << 6)
            | (((zeros[base + 7] == 0) as u8) << 7);
        bytes[c] = b;
    }
    u32::from_le_bytes(bytes)
}

/// Compute the 32-bit "eq mask" for a single (compile-time-known W, runtime row)
/// of a packed u32 block. Bit `lane` is set iff the element at `(row, lane)`
/// equals the constant whose row words are `const_rows`.
///
/// The inner loop walks all 32 lanes at a fixed row, so shift and mask are
/// loop-invariant. With AVX2/AVX-512 enabled this lowers to broadcast-XOR +
/// broadcast-AND + per-lane `cmpeq` + movemask.
#[inline(always)]
fn compute_row_mask<const W: usize>(packed: &[u32], const_rows: &[u32; 32], row: usize) -> u32 {
    const T: usize = 32;
    const LANES: usize = 32;
    let elem_mask: u32 = if W == 32 { u32::MAX } else { (1u32 << W) - 1 };

    let curr_word = (row * W) / T;
    let next_word = ((row + 1) * W) / T;
    let shift = (row * W) % T;

    let mut zeros = [0u32; LANES];
    if next_word == curr_word {
        let window = elem_mask << shift;
        let const_cw = const_rows[curr_word];
        let cw_off = LANES * curr_word;
        for lane in 0..LANES {
            zeros[lane] = (packed[cw_off + lane] ^ const_cw) & window;
        }
    } else if next_word < W {
        let current_bits = T - shift;
        let remaining_bits = W - current_bits;
        let high_mask = u32::MAX << shift;
        let low_mask = (1u32 << remaining_bits) - 1;

        let const_cw = const_rows[curr_word];
        let const_nw = const_rows[next_word];
        let cw_off = LANES * curr_word;
        let nw_off = LANES * next_word;

        for lane in 0..LANES {
            let a = (packed[cw_off + lane] ^ const_cw) & high_mask;
            let b = (packed[nw_off + lane] ^ const_nw) & low_mask;
            zeros[lane] = a | b;
        }
    } else {
        let high_mask = u32::MAX << shift;
        let const_cw = const_rows[curr_word];
        let cw_off = LANES * curr_word;
        for lane in 0..LANES {
            zeros[lane] = (packed[cw_off + lane] ^ const_cw) & high_mask;
        }
    }

    movemask32(&zeros)
}

/// Collect 1024 result bits as 16 little-endian `u64`s, using the
/// `collect_bool` chunk-of-64 idea but with each chunk emitted by a SIMD
/// row-mask computation. Per `u64`: two rows that map to its low and high
/// halves are computed and concatenated -- one store per chunk, zero
/// per-block heap traffic.
#[inline]
fn block_eq_collect_u64<const W: usize, const B: usize>(
    packed: &[u32; B],
    const_rows: &[u32; 32],
    out: &mut [u64; 16],
) {
    for u in 0..16 {
        let row_lo = ROW_FOR_K_U32[2 * u];
        let row_hi = ROW_FOR_K_U32[2 * u + 1];
        let lo = compute_row_mask::<W>(packed, const_rows, row_lo) as u64;
        let hi = compute_row_mask::<W>(packed, const_rows, row_hi) as u64;
        out[u] = lo | (hi << 32);
    }
}

// ---------------------------------------------------------------------------
// Hand-written AVX-512 path -- explicit intrinsics, no auto-vec dependency.
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m128i, __m256i, __m512i, _kand_mask16, _mm256_and_si256, _mm256_castsi256_ps,
    _mm256_cmpeq_epi32, _mm256_loadu_si256, _mm256_movemask_ps, _mm256_or_si256,
    _mm256_set1_epi32, _mm256_setzero_si256, _mm256_xor_si256, _mm512_and_si512,
    _mm512_cmpeq_epi8_mask, _mm512_cmpeq_epi16_mask, _mm512_cmpeq_epu32_mask,
    _mm512_loadu_si512, _mm512_or_si512, _mm512_set1_epi32, _mm512_setzero_si512,
    _mm512_sll_epi32, _mm512_srl_epi32, _mm512_ternarylogic_epi32,
    _mm512_testn_epi32_mask, _mm512_xor_si512, _mm_cvtsi32_si128, _pext_u64,
};

/// Hand-rolled AVX-512 row mask: load 32 packed u32 lanes as two `__m512i`,
/// broadcast XOR + AND, then `vpcmpeqd` to two 16-bit mask registers, OR'd
/// into one 32-bit row mask. ~8 SIMD instructions per row.
///
/// # Safety
/// Caller must hold the AVX-512F target feature. The `packed` slice must have
/// at least 32 `u32`s starting at `cw_off`, and (when crossing a word boundary)
/// at `nw_off`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,bmi2")]
#[inline]
unsafe fn row_mask_avx512<const W: usize>(packed: &[u32], const_rows: &[u32; 32], row: usize) -> u32 {
    const T: usize = 32;
    const LANES: usize = 32;
    let elem_mask: u32 = if W == 32 { u32::MAX } else { (1u32 << W) - 1 };

    let curr_word = (row * W) / T;
    let next_word = ((row + 1) * W) / T;
    let shift = (row * W) % T;

    // SAFETY: this function has `#[target_feature(enable = "avx512f")]` so all
    // intrinsics below are callable; pointer loads come from `packed` which the
    // caller has guaranteed is at least `LANES * W` u32s long.
    unsafe {
        let zero = _mm512_setzero_si512();

        if next_word == curr_word {
            let window = elem_mask << shift;
            let const_v = _mm512_set1_epi32(const_rows[curr_word] as i32);
            let window_v = _mm512_set1_epi32(window as i32);
            let base = packed.as_ptr().wrapping_add(LANES * curr_word);
            let v0 = _mm512_loadu_si512(base as *const __m512i);
            let v1 = _mm512_loadu_si512(base.add(16) as *const __m512i);
            let z0 = _mm512_and_si512(_mm512_xor_si512(v0, const_v), window_v);
            let z1 = _mm512_and_si512(_mm512_xor_si512(v1, const_v), window_v);
            let m0 = _mm512_cmpeq_epu32_mask(z0, zero) as u32;
            let m1 = _mm512_cmpeq_epu32_mask(z1, zero) as u32;
            m0 | (m1 << 16)
        } else if next_word < W {
            let current_bits = T - shift;
            let remaining_bits = W - current_bits;
            let high_mask = u32::MAX << shift;
            let low_mask = (1u32 << remaining_bits) - 1;

            let const_cw = _mm512_set1_epi32(const_rows[curr_word] as i32);
            let const_nw = _mm512_set1_epi32(const_rows[next_word] as i32);
            let high_v = _mm512_set1_epi32(high_mask as i32);
            let low_v = _mm512_set1_epi32(low_mask as i32);

            let cw = packed.as_ptr().wrapping_add(LANES * curr_word);
            let nw = packed.as_ptr().wrapping_add(LANES * next_word);
            let a0 = _mm512_loadu_si512(cw as *const __m512i);
            let a1 = _mm512_loadu_si512(cw.add(16) as *const __m512i);
            let b0 = _mm512_loadu_si512(nw as *const __m512i);
            let b1 = _mm512_loadu_si512(nw.add(16) as *const __m512i);

            let za0 = _mm512_and_si512(_mm512_xor_si512(a0, const_cw), high_v);
            let za1 = _mm512_and_si512(_mm512_xor_si512(a1, const_cw), high_v);
            let zb0 = _mm512_and_si512(_mm512_xor_si512(b0, const_nw), low_v);
            let zb1 = _mm512_and_si512(_mm512_xor_si512(b1, const_nw), low_v);
            let z0 = _mm512_or_si512(za0, zb0);
            let z1 = _mm512_or_si512(za1, zb1);

            let m0 = _mm512_cmpeq_epu32_mask(z0, zero) as u32;
            let m1 = _mm512_cmpeq_epu32_mask(z1, zero) as u32;
            m0 | (m1 << 16)
        } else {
            let high_mask = u32::MAX << shift;
            let const_v = _mm512_set1_epi32(const_rows[curr_word] as i32);
            let high_v = _mm512_set1_epi32(high_mask as i32);
            let base = packed.as_ptr().wrapping_add(LANES * curr_word);
            let v0 = _mm512_loadu_si512(base as *const __m512i);
            let v1 = _mm512_loadu_si512(base.add(16) as *const __m512i);
            let z0 = _mm512_and_si512(_mm512_xor_si512(v0, const_v), high_v);
            let z1 = _mm512_and_si512(_mm512_xor_si512(v1, const_v), high_v);
            let m0 = _mm512_cmpeq_epu32_mask(z0, zero) as u32;
            let m1 = _mm512_cmpeq_epu32_mask(z1, zero) as u32;
            m0 | (m1 << 16)
        }
    }
}

/// AVX-512 batched eq using `row_mask_avx512` for the inner kernel.
///
/// # Safety
/// AVX-512F target feature required.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,bmi2")]
#[inline]
unsafe fn block_eq_batch_avx512_hand<const W: usize, const B: usize>(
    blocks: &[[u32; B]],
    const_rows: &[u32; 32],
    out: &mut [u64],
) {
    debug_assert_eq!(out.len(), 16 * blocks.len());
    for (b, blk) in blocks.iter().enumerate() {
        let chunk = &mut out[b * 16..b * 16 + 16];
        for u in 0..16 {
            let row_lo = ROW_FOR_K_U32[2 * u];
            let row_hi = ROW_FOR_K_U32[2 * u + 1];
            let lo = unsafe { row_mask_avx512::<W>(blk.as_slice(), const_rows, row_lo) } as u64;
            let hi = unsafe { row_mask_avx512::<W>(blk.as_slice(), const_rows, row_hi) } as u64;
            chunk[u] = lo | (hi << 32);
        }
    }
}

// ---------------------------------------------------------------------------
// AVX-512 v2: testn (fuse and+cmpeq into one) + per-block hoisted const_row
// broadcasts. Same algorithm as v1 (splat-cmp), tighter encoding.
//
// Op count per non-boundary row (per 16-lane chunk):
//   v1: load + xor + and + cmpeq         = 4 ops
//   v2: load + xor + testn               = 3 ops    (25% reduction)
// Per boundary row:
//   v1: 2*load + 2*xor + 2*and + or + cmpeq = 8 ops
//   v2: 2*load + 2*xor + 2*testn + kand     = 7 ops
//
// And the broadcasts of `const_rows[curr_word]` move outside the row loop
// (only W distinct values, each used by T/W rows).
// ---------------------------------------------------------------------------

/// AVX-512 v2 batched eq: testn fusion + hoisted const_row broadcasts.
///
/// # Safety
/// AVX-512F target feature required.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,bmi2")]
#[inline]
unsafe fn block_eq_batch_avx512_v2<const W: usize, const B: usize>(
    blocks: &[[u32; B]],
    const_rows: &[u32; 32],
    out: &mut [u64],
) {
    const T: usize = 32;
    const LANES: usize = 32;
    debug_assert_eq!(out.len(), 16 * blocks.len());

    // SAFETY: feature-gated.
    unsafe {
        // Pre-broadcast all W const_row values once; each is used by T/W rows.
        let mut const_v: [__m512i; 32] = [_mm512_setzero_si512(); 32];
        for i in 0..W {
            const_v[i] = _mm512_set1_epi32(const_rows[i] as i32);
        }

        let elem_mask: u32 = if W == 32 { u32::MAX } else { (1u32 << W) - 1 };

        for (b_idx, blk) in blocks.iter().enumerate() {
            let mut row_masks = [0u32; 32];
            for row in 0..T {
                let curr_word = (row * W) / T;
                let shift = (row * W) % T;
                // True spanning condition: the W-bit field actually straddles
                // a word boundary (not just "next row uses a new word").
                let spans = shift + W > T;
                let next_word = curr_word + 1;
                let base = blk.as_ptr().wrapping_add(LANES * curr_word);
                let v0 = _mm512_loadu_si512(base as *const __m512i);
                let v1 = _mm512_loadu_si512(base.add(16) as *const __m512i);
                let c_cw = const_v[curr_word];

                let m: u32 = if !spans {
                    // W bits live entirely in curr_word at positions [shift, shift+W).
                    let window = elem_mask << shift;
                    let win_v = _mm512_set1_epi32(window as i32);
                    let m0 = _mm512_testn_epi32_mask(
                        _mm512_xor_si512(v0, c_cw),
                        win_v,
                    ) as u32;
                    let m1 = _mm512_testn_epi32_mask(
                        _mm512_xor_si512(v1, c_cw),
                        win_v,
                    ) as u32;
                    m0 | (m1 << 16)
                } else {
                    // Spanning: top bits of curr_word + bottom bits of next_word.
                    let current_bits = T - shift;
                    let remaining_bits = W - current_bits;
                    let high_mask = u32::MAX << shift;
                    let low_mask = (1u32 << remaining_bits) - 1;
                    let high_v = _mm512_set1_epi32(high_mask as i32);
                    let low_v = _mm512_set1_epi32(low_mask as i32);
                    let c_nw = const_v[next_word];
                    let nw = blk.as_ptr().wrapping_add(LANES * next_word);
                    let n0 = _mm512_loadu_si512(nw as *const __m512i);
                    let n1 = _mm512_loadu_si512(nw.add(16) as *const __m512i);
                    // (cw ^ c_cw) & high == 0 AND (nw ^ c_nw) & low == 0
                    let m_a0 = _mm512_testn_epi32_mask(
                        _mm512_xor_si512(v0, c_cw),
                        high_v,
                    );
                    let m_a1 = _mm512_testn_epi32_mask(
                        _mm512_xor_si512(v1, c_cw),
                        high_v,
                    );
                    let m_b0 = _mm512_testn_epi32_mask(
                        _mm512_xor_si512(n0, c_nw),
                        low_v,
                    );
                    let m_b1 = _mm512_testn_epi32_mask(
                        _mm512_xor_si512(n1, c_nw),
                        low_v,
                    );
                    let m0 = _kand_mask16(m_a0, m_b0) as u32;
                    let m1 = _kand_mask16(m_a1, m_b1) as u32;
                    m0 | (m1 << 16)
                };

                row_masks[row] = m;
            }

            let chunk = &mut out[b_idx * 16..b_idx * 16 + 16];
            for u in 0..16 {
                let lo = row_masks[ROW_FOR_K_U32[2 * u]] as u64;
                let hi = row_masks[ROW_FOR_K_U32[2 * u + 1]] as u64;
                chunk[u] = lo | (hi << 32);
            }
        }
    }
}

/// AVX-512 v3 batched eq: precompute everything possible per row (broadcast
/// `c & w` once, broadcast `w` once), inner loop is just load + AND + cmpeq.
///
/// Op count per non-boundary row chunk: load + and + cmpeq = 3 ops.
/// Op count per boundary row chunk: 2*load + 2*and + 2*cmpeq + kand = 7 ops
/// (vs v2's 7 ops with testn -- but cmpeq has 3-cyc latency, testn 4-cyc).
///
/// # Safety
/// AVX-512F target feature required.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,bmi2")]
#[inline]
unsafe fn block_eq_batch_avx512_v3<const W: usize, const B: usize>(
    blocks: &[[u32; B]],
    const_rows: &[u32; 32],
    out: &mut [u64],
) {
    const T: usize = 32;
    const LANES: usize = 32;
    debug_assert_eq!(out.len(), 16 * blocks.len());

    // SAFETY: feature-gated.
    unsafe {
        let elem_mask: u32 = if W == 32 { u32::MAX } else { (1u32 << W) - 1 };
        let zero = _mm512_setzero_si512();

        // Pre-broadcast per-row constants (W rows max; we compute all 32 for
        // simplicity but only those that get used are touched).
        // For non-boundary rows: window_v[r], target_v[r] = (const_rows[cw] & window) broadcast.
        // For boundary rows: hi_v[r], lo_v[r], hi_tgt_v[r], lo_tgt_v[r].
        let mut window_v: [__m512i; 32] = [zero; 32];
        let mut target_v: [__m512i; 32] = [zero; 32];
        let mut hi_v: [__m512i; 32] = [zero; 32];
        let mut lo_v: [__m512i; 32] = [zero; 32];
        let mut hi_tgt_v: [__m512i; 32] = [zero; 32];
        let mut lo_tgt_v: [__m512i; 32] = [zero; 32];
        let mut spans = [false; 32];
        for row in 0..T {
            let curr_word = (row * W) / T;
            let shift = (row * W) % T;
            let span = shift + W > T;
            spans[row] = span;
            if !span {
                let window = elem_mask << shift;
                let target = const_rows[curr_word] & window;
                window_v[row] = _mm512_set1_epi32(window as i32);
                target_v[row] = _mm512_set1_epi32(target as i32);
            } else {
                let current_bits = T - shift;
                let remaining_bits = W - current_bits;
                let high_mask = u32::MAX << shift;
                let low_mask = (1u32 << remaining_bits) - 1;
                let next_word = curr_word + 1;
                let hi_tgt = const_rows[curr_word] & high_mask;
                let lo_tgt = const_rows[next_word] & low_mask;
                hi_v[row] = _mm512_set1_epi32(high_mask as i32);
                lo_v[row] = _mm512_set1_epi32(low_mask as i32);
                hi_tgt_v[row] = _mm512_set1_epi32(hi_tgt as i32);
                lo_tgt_v[row] = _mm512_set1_epi32(lo_tgt as i32);
            }
        }

        for (b_idx, blk) in blocks.iter().enumerate() {
            let mut row_masks = [0u32; 32];
            for row in 0..T {
                let curr_word = (row * W) / T;
                let base = blk.as_ptr().wrapping_add(LANES * curr_word);
                let v0 = _mm512_loadu_si512(base as *const __m512i);
                let v1 = _mm512_loadu_si512(base.add(16) as *const __m512i);

                let m: u32 = if !spans[row] {
                    let masked0 = _mm512_and_si512(v0, window_v[row]);
                    let masked1 = _mm512_and_si512(v1, window_v[row]);
                    let m0 = _mm512_cmpeq_epu32_mask(masked0, target_v[row]) as u32;
                    let m1 = _mm512_cmpeq_epu32_mask(masked1, target_v[row]) as u32;
                    m0 | (m1 << 16)
                } else {
                    let next_word = curr_word + 1;
                    let nw = blk.as_ptr().wrapping_add(LANES * next_word);
                    let n0 = _mm512_loadu_si512(nw as *const __m512i);
                    let n1 = _mm512_loadu_si512(nw.add(16) as *const __m512i);
                    let h0 = _mm512_and_si512(v0, hi_v[row]);
                    let h1 = _mm512_and_si512(v1, hi_v[row]);
                    let l0 = _mm512_and_si512(n0, lo_v[row]);
                    let l1 = _mm512_and_si512(n1, lo_v[row]);
                    let m_h0 = _mm512_cmpeq_epu32_mask(h0, hi_tgt_v[row]);
                    let m_h1 = _mm512_cmpeq_epu32_mask(h1, hi_tgt_v[row]);
                    let m_l0 = _mm512_cmpeq_epu32_mask(l0, lo_tgt_v[row]);
                    let m_l1 = _mm512_cmpeq_epu32_mask(l1, lo_tgt_v[row]);
                    let m0 = _kand_mask16(m_h0, m_l0) as u32;
                    let m1 = _kand_mask16(m_h1, m_l1) as u32;
                    m0 | (m1 << 16)
                };
                row_masks[row] = m;
            }
            let chunk = &mut out[b_idx * 16..b_idx * 16 + 16];
            for u in 0..16 {
                let lo = row_masks[ROW_FOR_K_U32[2 * u]] as u64;
                let hi = row_masks[ROW_FOR_K_U32[2 * u + 1]] as u64;
                chunk[u] = lo | (hi << 32);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// AVX2 hand-rolled splat-cmp (no AVX-512, no vpternlogd, no kregs).
//
// 256-bit ymm with 8 u32 lanes per chunk; 4 chunks per 32-lane row.
// Bit extraction via vpcmpeqd + vpmovmskps per 8-lane chunk (8 bits each).
// ---------------------------------------------------------------------------

/// AVX2 batched eq using splat-cmp algorithm.
///
/// # Safety
/// AVX2 target feature required.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi2")]
#[inline]
unsafe fn block_eq_batch_avx2_v4_u32<const W: usize, const B: usize>(
    blocks: &[[u32; B]],
    const_rows: &[u32; 32],
    out: &mut [u64],
) {
    const T: usize = 32;
    const LANES: usize = 32;
    debug_assert_eq!(out.len(), 16 * blocks.len());

    // SAFETY: feature-gated.
    unsafe {
        let elem_mask: u32 = if W == 32 { u32::MAX } else { (1u32 << W) - 1 };

        // Pre-broadcast W const_row values + per-row masks.
        let zero = _mm256_setzero_si256();
        let mut const_v: [__m256i; 32] = [zero; 32];
        for i in 0..W {
            const_v[i] = _mm256_set1_epi32(const_rows[i] as i32);
        }
        let mut window_v: [__m256i; 32] = [zero; 32];
        let mut hi_v: [__m256i; 32] = [zero; 32];
        let mut lo_v: [__m256i; 32] = [zero; 32];
        let mut spans = [false; 32];
        for row in 0..T {
            let shift = (row * W) % T;
            let span = shift + W > T;
            spans[row] = span;
            if !span {
                window_v[row] = _mm256_set1_epi32((elem_mask << shift) as i32);
            } else {
                let current_bits = T - shift;
                let remaining_bits = W - current_bits;
                hi_v[row] = _mm256_set1_epi32((u32::MAX << shift) as i32);
                lo_v[row] = _mm256_set1_epi32(((1u32 << remaining_bits) - 1) as i32);
            }
        }

        for (b_idx, blk) in blocks.iter().enumerate() {
            let mut row_masks = [0u32; 32];
            for row in 0..T {
                let curr_word = (row * W) / T;
                let base = blk.as_ptr().wrapping_add(LANES * curr_word);
                let c_cw = const_v[curr_word];

                // 4 chunks of 8 lanes via 256-bit ymm.
                let m: u32 = if !spans[row] {
                    let w = window_v[row];
                    let v0 = _mm256_loadu_si256(base as *const __m256i);
                    let v1 = _mm256_loadu_si256(base.add(8) as *const __m256i);
                    let v2 = _mm256_loadu_si256(base.add(16) as *const __m256i);
                    let v3 = _mm256_loadu_si256(base.add(24) as *const __m256i);
                    let z0 = _mm256_and_si256(_mm256_xor_si256(v0, c_cw), w);
                    let z1 = _mm256_and_si256(_mm256_xor_si256(v1, c_cw), w);
                    let z2 = _mm256_and_si256(_mm256_xor_si256(v2, c_cw), w);
                    let z3 = _mm256_and_si256(_mm256_xor_si256(v3, c_cw), w);
                    let zv = _mm256_setzero_si256();
                    let m0 = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(z0, zv))) as u32 & 0xFF;
                    let m1 = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(z1, zv))) as u32 & 0xFF;
                    let m2 = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(z2, zv))) as u32 & 0xFF;
                    let m3 = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(z3, zv))) as u32 & 0xFF;
                    m0 | (m1 << 8) | (m2 << 16) | (m3 << 24)
                } else {
                    let next_word = curr_word + 1;
                    let c_nw = const_v[next_word];
                    let hi = hi_v[row];
                    let lo = lo_v[row];
                    let nw = blk.as_ptr().wrapping_add(LANES * next_word);
                    let v0 = _mm256_loadu_si256(base as *const __m256i);
                    let v1 = _mm256_loadu_si256(base.add(8) as *const __m256i);
                    let v2 = _mm256_loadu_si256(base.add(16) as *const __m256i);
                    let v3 = _mm256_loadu_si256(base.add(24) as *const __m256i);
                    let n0 = _mm256_loadu_si256(nw as *const __m256i);
                    let n1 = _mm256_loadu_si256(nw.add(8) as *const __m256i);
                    let n2 = _mm256_loadu_si256(nw.add(16) as *const __m256i);
                    let n3 = _mm256_loadu_si256(nw.add(24) as *const __m256i);
                    let h0 = _mm256_and_si256(_mm256_xor_si256(v0, c_cw), hi);
                    let h1 = _mm256_and_si256(_mm256_xor_si256(v1, c_cw), hi);
                    let h2 = _mm256_and_si256(_mm256_xor_si256(v2, c_cw), hi);
                    let h3 = _mm256_and_si256(_mm256_xor_si256(v3, c_cw), hi);
                    let l0 = _mm256_and_si256(_mm256_xor_si256(n0, c_nw), lo);
                    let l1 = _mm256_and_si256(_mm256_xor_si256(n1, c_nw), lo);
                    let l2 = _mm256_and_si256(_mm256_xor_si256(n2, c_nw), lo);
                    let l3 = _mm256_and_si256(_mm256_xor_si256(n3, c_nw), lo);
                    let z0 = _mm256_or_si256(h0, l0);
                    let z1 = _mm256_or_si256(h1, l1);
                    let z2 = _mm256_or_si256(h2, l2);
                    let z3 = _mm256_or_si256(h3, l3);
                    let zv = _mm256_setzero_si256();
                    let m0 = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(z0, zv))) as u32 & 0xFF;
                    let m1 = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(z1, zv))) as u32 & 0xFF;
                    let m2 = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(z2, zv))) as u32 & 0xFF;
                    let m3 = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(z3, zv))) as u32 & 0xFF;
                    m0 | (m1 << 8) | (m2 << 16) | (m3 << 24)
                };
                row_masks[row] = m;
            }
            let chunk = &mut out[b_idx * 16..b_idx * 16 + 16];
            for u in 0..16 {
                let lo = row_masks[ROW_FOR_K_U32[2 * u]] as u64;
                let hi = row_masks[ROW_FOR_K_U32[2 * u + 1]] as u64;
                chunk[u] = lo | (hi << 32);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// v6: per-W natural-granularity cmpeq.
//
// Key insight: for W ∈ {8, 16, 32}, the packed layout has one W-bit element
// per SIMD lane at byte/word/dword granularity. We can use `vpcmpeqb`,
// `vpcmpeqw`, `vpcmpeqd` directly against a broadcast constant, getting a
// kmask with one bit PER ELEMENT — not per row.
//
// One zmm cmpeq → 64 / 32 / 16 element results.
// All 32 rows of a block collapse into the same SIMD work as 1-row in v4
// — there's no per-row outer loop.
//
// Per block at W=8:
//   - 8 zmm loads (covers 256 u32 = 1024 bytes = 1024 elements at 1 byte each)
//   - 8 vpxor + 8 vpcmpeqb_mask = 16 SIMD ops to produce 8 × 64-bit kmasks
//   - 4 PEXTs per kmask (extract per-row bits) + assembly = ~32 scalar ops
//   - One final ROW_FOR_K shuffle
// Per block at v4 W=8: ~6 ops × 32 rows = ~200 ops. v6 ≈ 5× fewer ops.
//
// For W ∉ {8, 16, 32}, fall back to v4 (or a specialized smear-then-extract
// kernel — left as TODO).
// ---------------------------------------------------------------------------

/// W=8 specialization: byte-granularity vpcmpeqb against broadcast constant.
///
/// Packed layout for W=8, u32: each u32 holds 4 elements at byte positions
/// [0..7], [8..15], [16..23], [24..31] = 4 rows worth of one lane.
/// Across 32 lanes there are 8 packed u32s = 32 bytes = 32 elements per "row group of 4".
/// curr_word covers 4 rows. Total: 8 curr_words × 4 rows = 32 rows × 32 lanes = 1024 elements.
///
/// Strategy: load 16 u32s (= 64 bytes = 16 lanes × 4 rows) per zmm; vpcmpeqb gives
/// a 64-bit kmask with one bit per element. Then PEXT 4 row masks (16 bits each)
/// out of that kmask. Two zmm loads cover all 32 lanes of one curr_word group.
///
/// # Safety
/// AVX-512BW + BMI2 required.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,bmi2")]
#[inline]
unsafe fn block_eq_batch_avx512_v6_w8<const B: usize>(
    blocks: &[[u32; B]],
    constant: u32,
    out: &mut [u64],
) {
    debug_assert_eq!(B, 32 * 8);
    debug_assert_eq!(out.len(), 16 * blocks.len());

    // SAFETY: feature-gated.
    unsafe {
        // Broadcast `constant` (low 8 bits) into every byte of a zmm.
        let cb = _mm512_set1_epi32(((constant & 0xFF) * 0x0101_0101) as i32);

        for (b_idx, blk) in blocks.iter().enumerate() {
            let mut row_masks = [0u32; 32];
            // 8 curr_words, each has 32 lanes × 4 rows of bytes.
            // Per curr_word: 2 zmm loads (16 lanes each × 4 bytes/lane = 64 bytes per zmm).
            for cw in 0..8 {
                let base = blk.as_ptr().add(32 * cw) as *const __m512i;
                let v0 = _mm512_loadu_si512(base);
                let v1 = _mm512_loadu_si512(base.add(1));
                // XOR + cmpeq against zero. (vpcmpeqb vs broadcast(C) would also work.)
                let z0 = _mm512_xor_si512(v0, cb);
                let z1 = _mm512_xor_si512(v1, cb);
                let zero = _mm512_setzero_si512();
                // 64-bit kmask: bit i = (byte i of z is zero) = (element i matches).
                let m0: u64 = _mm512_cmpeq_epi8_mask(z0, zero);
                let m1: u64 = _mm512_cmpeq_epi8_mask(z1, zero);
                // Byte b of zmm corresponds to element at (row = 4cw + b%4, lane = b/4).
                // So bits at positions {0, 4, 8, ..., 60} of m0 = row 4cw, lanes 0..15.
                //    bits at positions {1, 5, 9, ..., 61} of m0 = row 4cw+1, lanes 0..15.
                //    etc.
                // Extract via PEXT with mask 0x1111...1111 << r.
                for r in 0..4 {
                    let pat = 0x1111_1111_1111_1111u64 << r;
                    let lo = _pext_u64(m0, pat) as u32; // 16 bits, lanes 0..15
                    let hi = _pext_u64(m1, pat) as u32; // 16 bits, lanes 16..31
                    row_masks[4 * cw + r] = lo | (hi << 16);
                }
            }
            // Same final permutation as v4.
            let chunk = &mut out[b_idx * 16..b_idx * 16 + 16];
            for u in 0..16 {
                let lo = row_masks[ROW_FOR_K_U32[2 * u]] as u64;
                let hi = row_masks[ROW_FOR_K_U32[2 * u + 1]] as u64;
                chunk[u] = lo | (hi << 32);
            }
        }
    }
}

/// W=16 specialization: word-granularity vpcmpeqw.
///
/// Each packed u32 holds 2 elements (rows 2*cw, 2*cw+1) for one lane.
/// Per zmm load (16 u32s) we get 32 u16 elements = 16 lanes × 2 rows.
/// vpcmpeqw → 32-bit kmask. PEXT 2 row masks of 16 bits each.
/// 2 zmm loads per curr_word (lanes 0..15 + 16..31). 16 curr_words per block.
///
/// # Safety
/// AVX-512BW + BMI2 required.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,bmi2")]
#[inline]
unsafe fn block_eq_batch_avx512_v6_w16<const B: usize>(
    blocks: &[[u32; B]],
    constant: u32,
    out: &mut [u64],
) {
    debug_assert_eq!(B, 32 * 16);
    debug_assert_eq!(out.len(), 16 * blocks.len());

    // SAFETY: feature-gated.
    unsafe {
        // Broadcast constant (low 16 bits) into every word of a zmm.
        let lo16 = constant & 0xFFFF;
        let cb = _mm512_set1_epi32(((lo16 | (lo16 << 16)) as i32));

        for (b_idx, blk) in blocks.iter().enumerate() {
            let mut row_masks = [0u32; 32];
            // 16 curr_words, each has 32 lanes × 2 rows of words.
            for cw in 0..16 {
                let base = blk.as_ptr().add(32 * cw) as *const __m512i;
                let v0 = _mm512_loadu_si512(base);
                let v1 = _mm512_loadu_si512(base.add(1));
                let z0 = _mm512_xor_si512(v0, cb);
                let z1 = _mm512_xor_si512(v1, cb);
                let zero = _mm512_setzero_si512();
                let m0: u32 = _mm512_cmpeq_epi16_mask(z0, zero);
                let m1: u32 = _mm512_cmpeq_epi16_mask(z1, zero);
                // word b of zmm corresponds to element at (row = 2cw + b%2, lane = b/2).
                // Bits {0, 2, 4, ..., 30} → row 2cw, lanes 0..15.
                // Bits {1, 3, 5, ..., 31} → row 2cw+1, lanes 0..15.
                let pat_even = 0x5555_5555u64;
                let pat_odd = 0xAAAA_AAAAu64;
                let lo_even = _pext_u64(m0 as u64, pat_even) as u32;
                let lo_odd = _pext_u64(m0 as u64, pat_odd) as u32;
                let hi_even = _pext_u64(m1 as u64, pat_even) as u32;
                let hi_odd = _pext_u64(m1 as u64, pat_odd) as u32;
                row_masks[2 * cw] = lo_even | (hi_even << 16);
                row_masks[2 * cw + 1] = lo_odd | (hi_odd << 16);
            }
            let chunk = &mut out[b_idx * 16..b_idx * 16 + 16];
            for u in 0..16 {
                let lo = row_masks[ROW_FOR_K_U32[2 * u]] as u64;
                let hi = row_masks[ROW_FOR_K_U32[2 * u + 1]] as u64;
                chunk[u] = lo | (hi << 32);
            }
        }
    }
}

/// AVX-512 v5: v4 + fused boundary path. Combine high and low halves via
/// OR before the single cmpeq (saves 2 cmpeq + 2 kand per boundary chunk).
///
/// # Safety
/// AVX-512F target feature required.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,bmi2")]
#[inline]
unsafe fn block_eq_batch_avx512_v5<const W: usize, const B: usize>(
    blocks: &[[u32; B]],
    const_rows: &[u32; 32],
    out: &mut [u64],
) {
    const T: usize = 32;
    const LANES: usize = 32;
    debug_assert_eq!(out.len(), 16 * blocks.len());

    // SAFETY: feature-gated.
    unsafe {
        let elem_mask: u32 = if W == 32 { u32::MAX } else { (1u32 << W) - 1 };
        let zero = _mm512_setzero_si512();

        let mut const_v: [__m512i; 32] = [zero; 32];
        for i in 0..W {
            const_v[i] = _mm512_set1_epi32(const_rows[i] as i32);
        }
        let mut window_v: [__m512i; 32] = [zero; 32];
        let mut hi_v: [__m512i; 32] = [zero; 32];
        let mut lo_v: [__m512i; 32] = [zero; 32];
        let mut spans = [false; 32];
        for row in 0..T {
            let shift = (row * W) % T;
            let span = shift + W > T;
            spans[row] = span;
            if !span {
                window_v[row] = _mm512_set1_epi32((elem_mask << shift) as i32);
            } else {
                let current_bits = T - shift;
                let remaining_bits = W - current_bits;
                hi_v[row] = _mm512_set1_epi32((u32::MAX << shift) as i32);
                lo_v[row] = _mm512_set1_epi32(((1u32 << remaining_bits) - 1) as i32);
            }
        }

        for (b_idx, blk) in blocks.iter().enumerate() {
            let mut row_masks = [0u32; 32];
            for row in 0..T {
                let curr_word = (row * W) / T;
                let base = blk.as_ptr().wrapping_add(LANES * curr_word);
                let v0 = _mm512_loadu_si512(base as *const __m512i);
                let v1 = _mm512_loadu_si512(base.add(16) as *const __m512i);
                let c_cw = const_v[curr_word];

                let m: u32 = if !spans[row] {
                    let z0 = _mm512_ternarylogic_epi32::<0x28>(v0, c_cw, window_v[row]);
                    let z1 = _mm512_ternarylogic_epi32::<0x28>(v1, c_cw, window_v[row]);
                    let m0 = _mm512_cmpeq_epu32_mask(z0, zero) as u32;
                    let m1 = _mm512_cmpeq_epu32_mask(z1, zero) as u32;
                    m0 | (m1 << 16)
                } else {
                    let next_word = curr_word + 1;
                    let c_nw = const_v[next_word];
                    let nw = blk.as_ptr().wrapping_add(LANES * next_word);
                    let n0 = _mm512_loadu_si512(nw as *const __m512i);
                    let n1 = _mm512_loadu_si512(nw.add(16) as *const __m512i);
                    // h = (v ^ c_cw) & hi;  l = (n ^ c_nw) & lo;  test (h | l) == 0
                    let h0 = _mm512_ternarylogic_epi32::<0x28>(v0, c_cw, hi_v[row]);
                    let h1 = _mm512_ternarylogic_epi32::<0x28>(v1, c_cw, hi_v[row]);
                    let l0 = _mm512_ternarylogic_epi32::<0x28>(n0, c_nw, lo_v[row]);
                    let l1 = _mm512_ternarylogic_epi32::<0x28>(n1, c_nw, lo_v[row]);
                    let t0 = _mm512_or_si512(h0, l0);
                    let t1 = _mm512_or_si512(h1, l1);
                    let m0 = _mm512_cmpeq_epu32_mask(t0, zero) as u32;
                    let m1 = _mm512_cmpeq_epu32_mask(t1, zero) as u32;
                    m0 | (m1 << 16)
                };
                row_masks[row] = m;
            }
            let chunk = &mut out[b_idx * 16..b_idx * 16 + 16];
            for u in 0..16 {
                let lo = row_masks[ROW_FOR_K_U32[2 * u]] as u64;
                let hi = row_masks[ROW_FOR_K_U32[2 * u + 1]] as u64;
                chunk[u] = lo | (hi << 32);
            }
        }
    }
}

/// AVX-512 v4: same as v3 (pre-broadcast everything) but use ternarylogic
/// to fuse `(v ^ c) & w` into one instruction, then cmpeq with zero.
/// Op count per non-boundary chunk: load + ternlog + cmpeq = 3 ops -- same
/// as v3, but tests if ternlog has better latency than `and + cmpeq w/ tgt`.
///
/// # Safety
/// AVX-512F target feature required.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,bmi2")]
#[inline]
unsafe fn block_eq_batch_avx512_v4<const W: usize, const B: usize>(
    blocks: &[[u32; B]],
    const_rows: &[u32; 32],
    out: &mut [u64],
) {
    const T: usize = 32;
    const LANES: usize = 32;
    debug_assert_eq!(out.len(), 16 * blocks.len());

    // SAFETY: feature-gated.
    unsafe {
        let elem_mask: u32 = if W == 32 { u32::MAX } else { (1u32 << W) - 1 };
        let zero = _mm512_setzero_si512();

        let mut const_v: [__m512i; 32] = [zero; 32];
        for i in 0..W {
            const_v[i] = _mm512_set1_epi32(const_rows[i] as i32);
        }
        let mut window_v: [__m512i; 32] = [zero; 32];
        let mut hi_v: [__m512i; 32] = [zero; 32];
        let mut lo_v: [__m512i; 32] = [zero; 32];
        let mut spans = [false; 32];
        for row in 0..T {
            let shift = (row * W) % T;
            let span = shift + W > T;
            spans[row] = span;
            if !span {
                window_v[row] = _mm512_set1_epi32((elem_mask << shift) as i32);
            } else {
                let current_bits = T - shift;
                let remaining_bits = W - current_bits;
                hi_v[row] = _mm512_set1_epi32((u32::MAX << shift) as i32);
                lo_v[row] = _mm512_set1_epi32(((1u32 << remaining_bits) - 1) as i32);
            }
        }

        for (b_idx, blk) in blocks.iter().enumerate() {
            let mut row_masks = [0u32; 32];
            for row in 0..T {
                let curr_word = (row * W) / T;
                let base = blk.as_ptr().wrapping_add(LANES * curr_word);
                let v0 = _mm512_loadu_si512(base as *const __m512i);
                let v1 = _mm512_loadu_si512(base.add(16) as *const __m512i);
                let c_cw = const_v[curr_word];

                let m: u32 = if !spans[row] {
                    // (v ^ c) & w via ternarylogic (imm 0x28).
                    let z0 = _mm512_ternarylogic_epi32::<0x28>(v0, c_cw, window_v[row]);
                    let z1 = _mm512_ternarylogic_epi32::<0x28>(v1, c_cw, window_v[row]);
                    let m0 = _mm512_cmpeq_epu32_mask(z0, zero) as u32;
                    let m1 = _mm512_cmpeq_epu32_mask(z1, zero) as u32;
                    m0 | (m1 << 16)
                } else {
                    let next_word = curr_word + 1;
                    let c_nw = const_v[next_word];
                    let nw = blk.as_ptr().wrapping_add(LANES * next_word);
                    let n0 = _mm512_loadu_si512(nw as *const __m512i);
                    let n1 = _mm512_loadu_si512(nw.add(16) as *const __m512i);
                    let h0 = _mm512_ternarylogic_epi32::<0x28>(v0, c_cw, hi_v[row]);
                    let h1 = _mm512_ternarylogic_epi32::<0x28>(v1, c_cw, hi_v[row]);
                    let l0 = _mm512_ternarylogic_epi32::<0x28>(n0, c_nw, lo_v[row]);
                    let l1 = _mm512_ternarylogic_epi32::<0x28>(n1, c_nw, lo_v[row]);
                    let m_h0 = _mm512_cmpeq_epu32_mask(h0, zero);
                    let m_h1 = _mm512_cmpeq_epu32_mask(h1, zero);
                    let m_l0 = _mm512_cmpeq_epu32_mask(l0, zero);
                    let m_l1 = _mm512_cmpeq_epu32_mask(l1, zero);
                    let m0 = _kand_mask16(m_h0, m_l0) as u32;
                    let m1 = _kand_mask16(m_h1, m_l1) as u32;
                    m0 | (m1 << 16)
                };
                row_masks[row] = m;
            }
            let chunk = &mut out[b_idx * 16..b_idx * 16 + 16];
            for u in 0..16 {
                let lo = row_masks[ROW_FOR_K_U32[2 * u]] as u64;
                let hi = row_masks[ROW_FOR_K_U32[2 * u + 1]] as u64;
                chunk[u] = lo | (hi << 32);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// AVX-512 hand-rolled, unpack-then-compare algorithm.
//
// Same SIMD level as `row_mask_avx512` (the splat variant) but uses the
// alternative *algorithm*: shift+and to extract each lane's W-bit value, then
// `_mm512_cmpeq_epu32_mask` against a broadcast scalar constant. No
// precomputed const-row words; the constant stays as a single broadcast value.
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,bmi2")]
#[inline]
unsafe fn row_mask_avx512_unpack<const W: usize>(
    packed: &[u32],
    constant_v: __m512i,
    mask_v: __m512i,
    row: usize,
) -> u32 {
    const T: usize = 32;
    const LANES: usize = 32;
    let curr_word = (row * W) / T;
    let next_word = ((row + 1) * W) / T;
    let shift = ((row * W) % T) as i32;

    // SAFETY: feature-gated; pointer loads come from `packed` which the caller
    // has guaranteed is at least `LANES * W` u32s long.
    unsafe {
        let base = packed.as_ptr().wrapping_add(LANES * curr_word);
        let v0 = _mm512_loadu_si512(base as *const __m512i);
        let v1 = _mm512_loadu_si512(base.add(16) as *const __m512i);
        let shift_count: __m128i = _mm_cvtsi32_si128(shift);

        if next_word == curr_word {
            // W bits fit in one word per lane: shift right, mask, compare.
            let e0 = _mm512_and_si512(_mm512_srl_epi32(v0, shift_count), mask_v);
            let e1 = _mm512_and_si512(_mm512_srl_epi32(v1, shift_count), mask_v);
            let m0 = _mm512_cmpeq_epu32_mask(e0, constant_v) as u32;
            let m1 = _mm512_cmpeq_epu32_mask(e1, constant_v) as u32;
            m0 | (m1 << 16)
        } else if next_word < W {
            // Boundary: combine high bits of curr_word with low bits of next_word.
            let current_bits = (T as i32) - shift;
            let lcount: __m128i = _mm_cvtsi32_si128(current_bits);
            let nw = packed.as_ptr().wrapping_add(LANES * next_word);
            let n0 = _mm512_loadu_si512(nw as *const __m512i);
            let n1 = _mm512_loadu_si512(nw.add(16) as *const __m512i);
            let h0 = _mm512_srl_epi32(v0, shift_count);
            let h1 = _mm512_srl_epi32(v1, shift_count);
            let l0 = _mm512_sll_epi32(n0, lcount);
            let l1 = _mm512_sll_epi32(n1, lcount);
            let e0 = _mm512_and_si512(_mm512_or_si512(h0, l0), mask_v);
            let e1 = _mm512_and_si512(_mm512_or_si512(h1, l1), mask_v);
            let m0 = _mm512_cmpeq_epu32_mask(e0, constant_v) as u32;
            let m1 = _mm512_cmpeq_epu32_mask(e1, constant_v) as u32;
            m0 | (m1 << 16)
        } else {
            let e0 = _mm512_and_si512(_mm512_srl_epi32(v0, shift_count), mask_v);
            let e1 = _mm512_and_si512(_mm512_srl_epi32(v1, shift_count), mask_v);
            let m0 = _mm512_cmpeq_epu32_mask(e0, constant_v) as u32;
            let m1 = _mm512_cmpeq_epu32_mask(e1, constant_v) as u32;
            m0 | (m1 << 16)
        }
    }
}

/// AVX-512 batched eq using the unpack-then-compare algorithm.
///
/// # Safety
/// AVX-512F target feature required.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,bmi2")]
#[inline]
unsafe fn block_eq_unpack_avx512_hand<const W: usize, const B: usize>(
    blocks: &[[u32; B]],
    constant: u32,
    out: &mut [u64],
) {
    debug_assert_eq!(out.len(), 16 * blocks.len());
    // SAFETY: feature-gated.
    unsafe {
        let constant_v = _mm512_set1_epi32(constant as i32);
        let elem_mask: u32 = if W == 32 { u32::MAX } else { (1u32 << W) - 1 };
        let mask_v = _mm512_set1_epi32(elem_mask as i32);
        for (b, blk) in blocks.iter().enumerate() {
            let chunk = &mut out[b * 16..b * 16 + 16];
            for u in 0..16 {
                let row_lo = ROW_FOR_K_U32[2 * u];
                let row_hi = ROW_FOR_K_U32[2 * u + 1];
                let lo = row_mask_avx512_unpack::<W>(blk.as_slice(), constant_v, mask_v, row_lo)
                    as u64;
                let hi = row_mask_avx512_unpack::<W>(blk.as_slice(), constant_v, mask_v, row_hi)
                    as u64;
                chunk[u] = lo | (hi << 32);
            }
        }
    }
}

/// Decoupled AVX-512: use fastlanes `BitPacking::unpack` to a stack [u32; 1024],
/// then AVX-512 cmpeq+kmask for the compare phase. Different from
/// `block_eq_unpack_avx512_hand` (which fuses unpack and compare in registers)
/// — this one writes the full unpacked block to L1 then re-reads it.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,bmi2")]
#[inline]
unsafe fn stack_unpack_avx512_collect<const W: usize, const B: usize>(
    blocks: &[[u32; B]],
    constant: u32,
    out: &mut [u64],
) {
    debug_assert_eq!(out.len(), 16 * blocks.len());
    // SAFETY: feature-gated.
    unsafe {
        let constant_v = _mm512_set1_epi32(constant as i32);
        for (b, blk) in blocks.iter().enumerate() {
            let mut unpacked = [0u32; 1024];
            BitPacking::unpack::<W, B>(blk, &mut unpacked);
            let chunk = &mut out[b * 16..b * 16 + 16];
            // 1024 bits = 16 u64s; each u64 = 64 bits = 4 chunks of 16 u32s.
            for u in 0..16 {
                let base = unpacked.as_ptr().add(u * 64);
                let v0 = _mm512_loadu_si512(base as *const __m512i);
                let v1 = _mm512_loadu_si512(base.add(16) as *const __m512i);
                let v2 = _mm512_loadu_si512(base.add(32) as *const __m512i);
                let v3 = _mm512_loadu_si512(base.add(48) as *const __m512i);
                let m0 = _mm512_cmpeq_epu32_mask(v0, constant_v) as u64;
                let m1 = _mm512_cmpeq_epu32_mask(v1, constant_v) as u64;
                let m2 = _mm512_cmpeq_epu32_mask(v2, constant_v) as u64;
                let m3 = _mm512_cmpeq_epu32_mask(v3, constant_v) as u64;
                chunk[u] = m0 | (m1 << 16) | (m2 << 32) | (m3 << 48);
            }
        }
    }
}

/// Strategy A2 wrapper: drop the per-block `[u64; 16]` directly into a
/// `BitBuffer`, no `collect_bool` repack and no per-block alloc beyond the
/// shared output buffer (allocated once outside the timing loop).
#[inline]
fn block_eq_simd_collect<const W: usize, const B: usize>(
    packed: &[u32; B],
    const_rows: &[u32; 32],
) -> BitBuffer {
    let mut bits = [0u64; 16];
    block_eq_collect_u64::<W, B>(packed, const_rows, &mut bits);
    let mut bytes = vortex_buffer::BufferMut::<u8>::with_capacity(128);
    for w in bits.iter() {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    BitBuffer::new(bytes.freeze(), 1024)
}

/// Batched eq over many blocks, writing 16 `u64`s per block into `out`.
///
/// Output capacity must be `16 * blocks.len()`. Per-iteration overhead is just
/// the SIMD compute; the output buffer is shared so there are no per-block
/// heap allocations.
#[inline]
fn block_eq_batch<const W: usize, const B: usize>(
    blocks: &[[u32; B]],
    const_rows: &[u32; 32],
    out: &mut [u64],
) {
    debug_assert_eq!(out.len(), 16 * blocks.len());
    for (b, blk) in blocks.iter().enumerate() {
        let chunk: &mut [u64; 16] = (&mut out[b * 16..b * 16 + 16]).try_into().unwrap();
        block_eq_collect_u64::<W, B>(blk, const_rows, chunk);
    }
}

/// Batched eq, AVX2 variant.
///
/// # Safety
/// Caller must ensure AVX2 is available.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn block_eq_batch_avx2<const W: usize, const B: usize>(
    blocks: &[[u32; B]],
    const_rows: &[u32; 32],
    out: &mut [u64],
) {
    block_eq_batch::<W, B>(blocks, const_rows, out)
}

/// Batched eq, AVX-512 variant.
///
/// # Safety
/// Caller must ensure AVX-512F/BW/DQ are available.
#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,bmi2")]
#[inline]
unsafe fn block_eq_batch_avx512<const W: usize, const B: usize>(
    blocks: &[[u32; B]],
    const_rows: &[u32; 32],
    out: &mut [u64],
) {
    block_eq_batch::<W, B>(blocks, const_rows, out)
}

/// Strategy B: heap-allocated unpack, then `collect_bool` with `== constant`.
#[inline]
fn heap_unpack_collect(packed: &[u32], width: usize, constant: u32) -> BitBuffer {
    let mut unpacked: Vec<u32> = vec![0u32; 1024];
    // SAFETY: input length is `128 * width / 4 == 32 * width` u32s; output is 1024.
    unsafe { BitPacking::unchecked_unpack(width, packed, &mut unpacked) };
    BitBufferMut::collect_bool(1024, |i| unpacked[i] == constant).freeze()
}

/// Strategy C: stack-allocated unpack with compile-time `W`, then `collect_bool`.
#[inline]
fn stack_unpack_collect<const W: usize, const B: usize>(
    packed: &[u32; B],
    constant: u32,
) -> BitBuffer {
    let mut unpacked = [0u32; 1024];
    BitPacking::unpack::<W, B>(packed, &mut unpacked);
    BitBufferMut::collect_bool(1024, |i| unpacked[i] == constant).freeze()
}

/// Strategy D (apples-to-apples vs `unpack`): fastlanes' `BitPackingCompare::unpack_cmp`
/// uses the **same** `unpack!` macro under the hood -- so the codegen is identical to
/// `BitPacking::unpack` except the kernel writes a `bool` to `output[$idx]` instead of
/// the unpacked `u32`. Same instruction palette, same auto-vec level. The `collect_bool`
/// call then bit-packs.
#[inline]
fn fastlanes_unpack_cmp_collect<const W: usize, const B: usize>(
    packed: &[u32; B],
    constant: u32,
) -> BitBuffer {
    let mut bools = [false; 1024];
    BitPackingCompare::unpack_cmp::<W, B, u32, _>(packed, &mut bools, |a, b| a == b, constant);
    BitBufferMut::collect_bool(1024, |i| bools[i]).freeze()
}

/// Strategy D batched: write all blocks' bools to a single big slice, then
/// `collect_bool` over the whole thing at the end. Removes per-block heap alloc
/// but the bool buffer is too large to fit in L1, so the final `collect_bool`
/// reads cold -- consistently slower than the `*_per1k` variants. Kept for
/// reference.
#[inline]
fn fastlanes_unpack_cmp_batch<const W: usize, const B: usize>(
    blocks: &[[u32; B]],
    constant: u32,
    out_bools: &mut [bool],
) {
    debug_assert_eq!(out_bools.len(), 1024 * blocks.len());
    for (b, blk) in blocks.iter().enumerate() {
        let slot: &mut [bool; 1024] = (&mut out_bools[b * 1024..b * 1024 + 1024])
            .try_into()
            .unwrap();
        BitPackingCompare::unpack_cmp::<W, B, u32, _>(blk, slot, |a, b| a == b, constant);
    }
}

/// Strategy D variant: per-block unpack_cmp into a **stack** `[bool; 1024]`,
/// then inline-collect 16 `u64`s into a shared output buffer. The bools stay
/// hot in L1 between the unpack write and the collect_bool read; no per-block
/// heap alloc, no global cold bool buffer.
#[inline]
fn fastlanes_unpack_cmp_per1k_shared<const W: usize, const B: usize>(
    blocks: &[[u32; B]],
    constant: u32,
    out: &mut [u64],
) {
    debug_assert_eq!(out.len(), 16 * blocks.len());
    for (b, blk) in blocks.iter().enumerate() {
        let mut bools = [false; 1024];
        BitPackingCompare::unpack_cmp::<W, B, u32, _>(
            blk,
            &mut bools,
            |a, b| a == b,
            constant,
        );
        let dst = &mut out[b * 16..b * 16 + 16];
        // Inline `collect_bool` for exactly 1024 bits: 16 chunks of 64.
        for chunk in 0..16 {
            let mut packed: u64 = 0;
            for bit_idx in 0..64 {
                packed |= (bools[chunk * 64 + bit_idx] as u64) << bit_idx;
            }
            dst[chunk] = packed;
        }
    }
}

/// Strategy C variant: per-block `BitPacking::unpack` into a **stack**
/// `[u32; 1024]`, then inline-collect with `== constant` writing 16 `u64`s
/// to the shared output. Same L1-locality argument as the unpack_cmp variant.
#[inline]
fn stack_unpack_per1k_shared<const W: usize, const B: usize>(
    blocks: &[[u32; B]],
    constant: u32,
    out: &mut [u64],
) {
    debug_assert_eq!(out.len(), 16 * blocks.len());
    for (b, blk) in blocks.iter().enumerate() {
        let mut unpacked = [0u32; 1024];
        BitPacking::unpack::<W, B>(blk, &mut unpacked);
        let dst = &mut out[b * 16..b * 16 + 16];
        for chunk in 0..16 {
            let mut packed: u64 = 0;
            for bit_idx in 0..64 {
                packed |= ((unpacked[chunk * 64 + bit_idx] == constant) as u64) << bit_idx;
            }
            dst[chunk] = packed;
        }
    }
}

// ---------------------------------------------------------------------------
// Benchmark harness
// ---------------------------------------------------------------------------

const BIT_WIDTHS_U32: &[u8] = &[1, 4, 12, 16, 24];

/// `B` for `T = u32` and a given `W`: `1024 * W / 32 == 32 * W`.
const fn b_u32(w: usize) -> usize {
    32 * w
}

#[divan::bench(consts = BIT_WIDTHS_U32)]
fn u32_heap_buf_pack<const W: u8>(bencher: Bencher) {
    let value: u32 = ((1u64 << W) - 1) as u32; // densest in-range constant
    let mut out = vec![0u32; b_u32(W as usize) * BLOCKS];
    bencher
        .counter(ItemsCount::new(1024usize * BLOCKS))
        .bench_local(|| {
            for block in 0..BLOCKS {
                let start = block * b_u32(W as usize);
                heap_buf_pack(black_box(value), W as usize, &mut out[start..start + b_u32(W as usize)]);
            }
            black_box(&mut out);
        });
}

#[divan::bench(consts = BIT_WIDTHS_U32)]
fn u32_stack_buf_pack<const W: u8>(bencher: Bencher) {
    let value: u32 = ((1u64 << W) - 1) as u32;
    let mut out = vec![0u32; b_u32(W as usize) * BLOCKS];
    bencher
        .counter(ItemsCount::new(1024usize * BLOCKS))
        .bench_local(|| {
            for block in 0..BLOCKS {
                let start = block * b_u32(W as usize);
                stack_buf_pack(
                    black_box(value),
                    W as usize,
                    &mut out[start..start + b_u32(W as usize)],
                );
            }
            black_box(&mut out);
        });
}

macro_rules! u32_const_benches {
    ($($name:ident => $W:literal),* $(,)?) => {
        $(
            #[divan::bench]
            fn $name(bencher: Bencher) {
                const W: usize = $W;
                const B: usize = b_u32(W);
                let value: u32 = ((1u64 << W) - 1) as u32;
                let mut out = vec![[0u32; B]; BLOCKS];
                bencher
                    .counter(ItemsCount::new(1024usize * BLOCKS))
                    .bench_local(|| {
                        for o in out.iter_mut() {
                            stack_buf_pack_const::<u32, W, B>(black_box(value), o);
                        }
                        black_box(&mut out);
                    });
            }
        )*
    };
}

u32_const_benches!(
    u32_stack_buf_pack_const_w1 => 1,
    u32_stack_buf_pack_const_w4 => 4,
    u32_stack_buf_pack_const_w12 => 12,
    u32_stack_buf_pack_const_w16 => 16,
    u32_stack_buf_pack_const_w24 => 24,
);

macro_rules! u32_compute_rows_benches {
    ($($name:ident => $W:literal),* $(,)?) => {
        $(
            #[divan::bench]
            fn $name(bencher: Bencher) {
                const W: usize = $W;
                const B: usize = b_u32(W);
                let value: u32 = ((1u64 << W) - 1) as u32;
                let mut out = vec![[0u32; B]; BLOCKS];
                bencher
                    .counter(ItemsCount::new(1024usize * BLOCKS))
                    .bench_local(|| {
                        for o in out.iter_mut() {
                            compute_rows_splat::<u32, W, B>(black_box(value), o);
                        }
                        black_box(&mut out);
                    });
            }
        )*
    };
}

u32_compute_rows_benches!(
    u32_compute_rows_splat_w1 => 1,
    u32_compute_rows_splat_w4 => 4,
    u32_compute_rows_splat_w12 => 12,
    u32_compute_rows_splat_w16 => 16,
    u32_compute_rows_splat_w24 => 24,
);

// ---------------------------------------------------------------------------
// Eq benches: 64 packed blocks vs a scalar constant.
//
// Each iteration consumes 64 * (32 * W) packed u32 words and produces a
// `BitBuffer` of length 64 * 1024 bits. The constant is chosen in-range so
// some blocks have a high match rate (we put the same constant in the array
// so half the blocks match exactly) -- branch behavior of the strategies
// should not affect timing materially since all of them are branch-free in
// the inner loop.
// ---------------------------------------------------------------------------

const EQ_BLOCKS: usize = 64;

/// Build `EQ_BLOCKS` packed blocks where every element is `value`.
/// We use the validated `compute_rows_splat` since it's far faster than
/// `unchecked_pack` for constants -- this is just setup, but speed matters
/// because divan still spends time on `with_inputs`.
fn make_constant_packed_blocks<const W: usize, const B: usize>(value: u32) -> Vec<[u32; B]> {
    let mut blocks = vec![[0u32; B]; EQ_BLOCKS];
    for b in blocks.iter_mut() {
        compute_rows_splat::<u32, W, B>(value, b);
    }
    blocks
}

macro_rules! u32_eq_benches {
    ($prefix:ident => $W:literal) => {
        paste::paste! {
            #[divan::bench]
            fn [<$prefix _block_no_unpack_collect_w $W>](bencher: Bencher) {
                const W: usize = $W;
                const B: usize = 32 * W;
                let value: u32 = ((1u64 << W) - 1) as u32;
                let blocks = make_constant_packed_blocks::<W, B>(value);
                let const_rows = const_row_words_u32::<W>(value);

                bencher
                    .counter(ItemsCount::new(1024usize * EQ_BLOCKS))
                    .bench_local(|| {
                        let mut out = BitBufferMut::with_capacity(1024 * EQ_BLOCKS);
                        for blk in blocks.iter() {
                            let bb = block_eq_no_unpack_collect::<W, B>(blk, &const_rows);
                            out.append_buffer(&bb);
                        }
                        black_box(out)
                    });
            }

            #[divan::bench]
            fn [<$prefix _block_simd_collect_w $W>](bencher: Bencher) {
                const W: usize = $W;
                const B: usize = 32 * W;
                let value: u32 = ((1u64 << W) - 1) as u32;
                let blocks = make_constant_packed_blocks::<W, B>(value);
                let const_rows = const_row_words_u32::<W>(value);

                bencher
                    .counter(ItemsCount::new(1024usize * EQ_BLOCKS))
                    .bench_local(|| {
                        let mut out = BitBufferMut::with_capacity(1024 * EQ_BLOCKS);
                        for blk in blocks.iter() {
                            let bb = block_eq_simd_collect::<W, B>(blk, &const_rows);
                            out.append_buffer(&bb);
                        }
                        black_box(out)
                    });
            }

            // Batched variants share one output `Vec<u64>` across blocks, so
            // the only work in the timed loop is SIMD compute -- no per-block
            // alloc, no BitBuffer wrapping until the end.
            #[divan::bench]
            fn [<$prefix _block_batch_w $W>](bencher: Bencher) {
                const W: usize = $W;
                const B: usize = 32 * W;
                let value: u32 = ((1u64 << W) - 1) as u32;
                let blocks = make_constant_packed_blocks::<W, B>(value);
                let const_rows = const_row_words_u32::<W>(value);
                let mut out: Vec<u64> = vec![0u64; 16 * EQ_BLOCKS];

                bencher
                    .counter(ItemsCount::new(1024usize * EQ_BLOCKS))
                    .bench_local(|| {
                        block_eq_batch::<W, B>(&blocks, &const_rows, &mut out);
                        black_box(&mut out);
                    });
            }

            #[divan::bench]
            fn [<$prefix _block_batch_avx2_w $W>](bencher: Bencher) {
                if !is_x86_feature_detected!("avx2") {
                    return;
                }
                const W: usize = $W;
                const B: usize = 32 * W;
                let value: u32 = ((1u64 << W) - 1) as u32;
                let blocks = make_constant_packed_blocks::<W, B>(value);
                let const_rows = const_row_words_u32::<W>(value);
                let mut out: Vec<u64> = vec![0u64; 16 * EQ_BLOCKS];

                bencher
                    .counter(ItemsCount::new(1024usize * EQ_BLOCKS))
                    .bench_local(|| {
                        // SAFETY: feature-gated above.
                        unsafe { block_eq_batch_avx2::<W, B>(&blocks, &const_rows, &mut out) };
                        black_box(&mut out);
                    });
            }

            #[divan::bench]
            fn [<$prefix _block_batch_avx512_w $W>](bencher: Bencher) {
                if !is_x86_feature_detected!("avx512f")
                    || !is_x86_feature_detected!("avx512bw")
                    || !is_x86_feature_detected!("avx512dq")
                {
                    return;
                }
                const W: usize = $W;
                const B: usize = 32 * W;
                let value: u32 = ((1u64 << W) - 1) as u32;
                let blocks = make_constant_packed_blocks::<W, B>(value);
                let const_rows = const_row_words_u32::<W>(value);
                let mut out: Vec<u64> = vec![0u64; 16 * EQ_BLOCKS];

                bencher
                    .counter(ItemsCount::new(1024usize * EQ_BLOCKS))
                    .bench_local(|| {
                        // SAFETY: feature-gated above.
                        unsafe { block_eq_batch_avx512::<W, B>(&blocks, &const_rows, &mut out) };
                        black_box(&mut out);
                    });
            }

            #[divan::bench]
            fn [<$prefix _block_batch_avx512_hand_w $W>](bencher: Bencher) {
                #[cfg(not(target_arch = "x86_64"))]
                {
                    return;
                }
                #[cfg(target_arch = "x86_64")]
                {
                    if !is_x86_feature_detected!("avx512f")
                        || !is_x86_feature_detected!("avx512bw")
                        || !is_x86_feature_detected!("avx512dq")
                    {
                        return;
                    }
                    const W: usize = $W;
                    const B: usize = 32 * W;
                    let value: u32 = ((1u64 << W) - 1) as u32;
                    let blocks = make_constant_packed_blocks::<W, B>(value);
                    let const_rows = const_row_words_u32::<W>(value);
                    let mut out: Vec<u64> = vec![0u64; 16 * EQ_BLOCKS];

                    bencher
                        .counter(ItemsCount::new(1024usize * EQ_BLOCKS))
                        .bench_local(|| {
                            // SAFETY: feature-gated above.
                            unsafe { block_eq_batch_avx512_hand::<W, B>(&blocks, &const_rows, &mut out) };
                            black_box(&mut out);
                        });
                }
            }

            // AVX2 v4 (u32): same algorithm as v4 but with 256-bit ymm,
            // vpxor + vpand (no vpternlogd), and movemask_ps for bit extraction.
            #[divan::bench]
            fn [<$prefix _block_batch_avx2_v4_w $W>](bencher: Bencher) {
                #[cfg(not(target_arch = "x86_64"))]
                {
                    return;
                }
                #[cfg(target_arch = "x86_64")]
                {
                    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("bmi2") {
                        return;
                    }
                    const W: usize = $W;
                    const B: usize = 32 * W;
                    let value: u32 = ((1u64 << W) - 1) as u32;
                    let blocks = make_constant_packed_blocks::<W, B>(value);
                    let const_rows = const_row_words_u32::<W>(value);
                    let mut out: Vec<u64> = vec![0u64; 16 * EQ_BLOCKS];

                    bencher
                        .counter(ItemsCount::new(1024usize * EQ_BLOCKS))
                        .bench_local(|| {
                            // SAFETY: feature-gated above.
                            unsafe { block_eq_batch_avx2_v4_u32::<W, B>(&blocks, &const_rows, &mut out) };
                            black_box(&mut out);
                        });
                }
            }

            // AVX-512 v3: precompute c & w as per-row target, inner loop is
            // load + and + cmpeq(target).
            #[divan::bench]
            fn [<$prefix _block_batch_avx512_v3_w $W>](bencher: Bencher) {
                #[cfg(not(target_arch = "x86_64"))]
                {
                    return;
                }
                #[cfg(target_arch = "x86_64")]
                {
                    if !is_x86_feature_detected!("avx512f")
                        || !is_x86_feature_detected!("avx512bw")
                        || !is_x86_feature_detected!("avx512dq")
                    {
                        return;
                    }
                    const W: usize = $W;
                    const B: usize = 32 * W;
                    let value: u32 = ((1u64 << W) - 1) as u32;
                    let blocks = make_constant_packed_blocks::<W, B>(value);
                    let const_rows = const_row_words_u32::<W>(value);
                    let mut out: Vec<u64> = vec![0u64; 16 * EQ_BLOCKS];

                    bencher
                        .counter(ItemsCount::new(1024usize * EQ_BLOCKS))
                        .bench_local(|| {
                            // SAFETY: feature-gated above.
                            unsafe { block_eq_batch_avx512_v3::<W, B>(&blocks, &const_rows, &mut out) };
                            black_box(&mut out);
                        });
                }
            }

            // AVX-512 v5: v4 + fused boundary path (OR-then-cmpeq).
            #[divan::bench]
            fn [<$prefix _block_batch_avx512_v5_w $W>](bencher: Bencher) {
                #[cfg(not(target_arch = "x86_64"))]
                {
                    return;
                }
                #[cfg(target_arch = "x86_64")]
                {
                    if !is_x86_feature_detected!("avx512f")
                        || !is_x86_feature_detected!("avx512bw")
                        || !is_x86_feature_detected!("avx512dq")
                    {
                        return;
                    }
                    const W: usize = $W;
                    const B: usize = 32 * W;
                    let value: u32 = ((1u64 << W) - 1) as u32;
                    let blocks = make_constant_packed_blocks::<W, B>(value);
                    let const_rows = const_row_words_u32::<W>(value);
                    let mut out: Vec<u64> = vec![0u64; 16 * EQ_BLOCKS];

                    bencher
                        .counter(ItemsCount::new(1024usize * EQ_BLOCKS))
                        .bench_local(|| {
                            // SAFETY: feature-gated above.
                            unsafe { block_eq_batch_avx512_v5::<W, B>(&blocks, &const_rows, &mut out) };
                            black_box(&mut out);
                        });
                }
            }

            // AVX-512 v4: pre-broadcast const_rows + window masks, inner uses
            // vpternlogd to fuse (v ^ c) & w into one op.
            #[divan::bench]
            fn [<$prefix _block_batch_avx512_v4_w $W>](bencher: Bencher) {
                #[cfg(not(target_arch = "x86_64"))]
                {
                    return;
                }
                #[cfg(target_arch = "x86_64")]
                {
                    if !is_x86_feature_detected!("avx512f")
                        || !is_x86_feature_detected!("avx512bw")
                        || !is_x86_feature_detected!("avx512dq")
                    {
                        return;
                    }
                    const W: usize = $W;
                    const B: usize = 32 * W;
                    let value: u32 = ((1u64 << W) - 1) as u32;
                    let blocks = make_constant_packed_blocks::<W, B>(value);
                    let const_rows = const_row_words_u32::<W>(value);
                    let mut out: Vec<u64> = vec![0u64; 16 * EQ_BLOCKS];

                    bencher
                        .counter(ItemsCount::new(1024usize * EQ_BLOCKS))
                        .bench_local(|| {
                            // SAFETY: feature-gated above.
                            unsafe { block_eq_batch_avx512_v4::<W, B>(&blocks, &const_rows, &mut out) };
                            black_box(&mut out);
                        });
                }
            }

            // AVX-512 v2 of splat-cmp: testn (fuse and+cmpeq) + hoisted
            // const_row broadcasts.
            #[divan::bench]
            fn [<$prefix _block_batch_avx512_v2_w $W>](bencher: Bencher) {
                #[cfg(not(target_arch = "x86_64"))]
                {
                    return;
                }
                #[cfg(target_arch = "x86_64")]
                {
                    if !is_x86_feature_detected!("avx512f")
                        || !is_x86_feature_detected!("avx512bw")
                        || !is_x86_feature_detected!("avx512dq")
                    {
                        return;
                    }
                    const W: usize = $W;
                    const B: usize = 32 * W;
                    let value: u32 = ((1u64 << W) - 1) as u32;
                    let blocks = make_constant_packed_blocks::<W, B>(value);
                    let const_rows = const_row_words_u32::<W>(value);
                    let mut out: Vec<u64> = vec![0u64; 16 * EQ_BLOCKS];

                    bencher
                        .counter(ItemsCount::new(1024usize * EQ_BLOCKS))
                        .bench_local(|| {
                            // SAFETY: feature-gated above.
                            unsafe { block_eq_batch_avx512_v2::<W, B>(&blocks, &const_rows, &mut out) };
                            black_box(&mut out);
                        });
                }
            }

            // AVX-512 hand, but using the UNPACK-then-compare algorithm
            // (shift+and+cmpeq against broadcast constant), no precomputed
            // const-row words. Apples-to-apples vs `block_batch_avx512_hand`
            // at the same SIMD level: which ALGORITHM wins?
            #[divan::bench]
            fn [<$prefix _block_unpack_avx512_hand_w $W>](bencher: Bencher) {
                #[cfg(not(target_arch = "x86_64"))]
                {
                    return;
                }
                #[cfg(target_arch = "x86_64")]
                {
                    if !is_x86_feature_detected!("avx512f")
                        || !is_x86_feature_detected!("avx512bw")
                        || !is_x86_feature_detected!("avx512dq")
                    {
                        return;
                    }
                    const W: usize = $W;
                    const B: usize = 32 * W;
                    let value: u32 = ((1u64 << W) - 1) as u32;
                    let blocks = make_constant_packed_blocks::<W, B>(value);
                    let mut out: Vec<u64> = vec![0u64; 16 * EQ_BLOCKS];

                    bencher
                        .counter(ItemsCount::new(1024usize * EQ_BLOCKS))
                        .bench_local(|| {
                            // SAFETY: feature-gated above.
                            unsafe {
                                block_eq_unpack_avx512_hand::<W, B>(&blocks, black_box(value), &mut out)
                            };
                            black_box(&mut out);
                        });
                }
            }

            // Decoupled: fastlanes BitPacking::unpack into stack [u32; 1024],
            // then AVX-512 compare phase. Same final cmpeq+kmask compute as
            // the fused unpack variant but with an extra write+reread of the
            // unpacked block.
            #[divan::bench]
            fn [<$prefix _stack_unpack_avx512_w $W>](bencher: Bencher) {
                #[cfg(not(target_arch = "x86_64"))]
                {
                    return;
                }
                #[cfg(target_arch = "x86_64")]
                {
                    if !is_x86_feature_detected!("avx512f")
                        || !is_x86_feature_detected!("avx512bw")
                        || !is_x86_feature_detected!("avx512dq")
                    {
                        return;
                    }
                    const W: usize = $W;
                    const B: usize = 32 * W;
                    let value: u32 = ((1u64 << W) - 1) as u32;
                    let blocks = make_constant_packed_blocks::<W, B>(value);
                    let mut out: Vec<u64> = vec![0u64; 16 * EQ_BLOCKS];

                    bencher
                        .counter(ItemsCount::new(1024usize * EQ_BLOCKS))
                        .bench_local(|| {
                            // SAFETY: feature-gated above.
                            unsafe {
                                stack_unpack_avx512_collect::<W, B>(&blocks, black_box(value), &mut out)
                            };
                            black_box(&mut out);
                        });
                }
            }

            #[divan::bench]
            fn [<$prefix _heap_unpack_collect_w $W>](bencher: Bencher) {
                const W: usize = $W;
                const B: usize = 32 * W;
                let value: u32 = ((1u64 << W) - 1) as u32;
                let blocks = make_constant_packed_blocks::<W, B>(value);

                bencher
                    .counter(ItemsCount::new(1024usize * EQ_BLOCKS))
                    .bench_local(|| {
                        let mut out = BitBufferMut::with_capacity(1024 * EQ_BLOCKS);
                        for blk in blocks.iter() {
                            let bb = heap_unpack_collect(blk.as_slice(), W, black_box(value));
                            out.append_buffer(&bb);
                        }
                        black_box(out)
                    });
            }

            #[divan::bench]
            fn [<$prefix _stack_unpack_collect_w $W>](bencher: Bencher) {
                const W: usize = $W;
                const B: usize = 32 * W;
                let value: u32 = ((1u64 << W) - 1) as u32;
                let blocks = make_constant_packed_blocks::<W, B>(value);

                bencher
                    .counter(ItemsCount::new(1024usize * EQ_BLOCKS))
                    .bench_local(|| {
                        let mut out = BitBufferMut::with_capacity(1024 * EQ_BLOCKS);
                        for blk in blocks.iter() {
                            let bb = stack_unpack_collect::<W, B>(blk, black_box(value));
                            out.append_buffer(&bb);
                        }
                        black_box(out)
                    });
            }

            // Apples-to-apples: fastlanes' own fused unpack+cmp+bools, then collect_bool.
            // Uses the exact same `unpack!` macro as `BitPacking::unpack`, so the codegen
            // is the same instruction palette (no extra target_feature, no intrinsics).
            #[divan::bench]
            fn [<$prefix _fl_unpack_cmp_collect_w $W>](bencher: Bencher) {
                const W: usize = $W;
                const B: usize = 32 * W;
                let value: u32 = ((1u64 << W) - 1) as u32;
                let blocks = make_constant_packed_blocks::<W, B>(value);

                bencher
                    .counter(ItemsCount::new(1024usize * EQ_BLOCKS))
                    .bench_local(|| {
                        let mut out = BitBufferMut::with_capacity(1024 * EQ_BLOCKS);
                        for blk in blocks.iter() {
                            let bb = fastlanes_unpack_cmp_collect::<W, B>(blk, black_box(value));
                            out.append_buffer(&bb);
                        }
                        black_box(out)
                    });
            }

            // Batched: share a single 64*1024-byte bools buffer + one final collect_bool.
            #[divan::bench]
            fn [<$prefix _fl_unpack_cmp_batch_w $W>](bencher: Bencher) {
                const W: usize = $W;
                const B: usize = 32 * W;
                let value: u32 = ((1u64 << W) - 1) as u32;
                let blocks = make_constant_packed_blocks::<W, B>(value);
                let mut bools = vec![false; 1024 * EQ_BLOCKS];

                bencher
                    .counter(ItemsCount::new(1024usize * EQ_BLOCKS))
                    .bench_local(|| {
                        fastlanes_unpack_cmp_batch::<W, B>(
                            &blocks,
                            black_box(value),
                            &mut bools,
                        );
                        let bb = BitBufferMut::collect_bool(bools.len(), |i| bools[i]).freeze();
                        black_box(bb)
                    });
            }

            // Per-1k collect: stack [bool; 1024] hot in L1 between unpack_cmp and
            // the inline 16-u64 pack; output written to a shared u64 buffer.
            #[divan::bench]
            fn [<$prefix _fl_unpack_cmp_per1k_w $W>](bencher: Bencher) {
                const W: usize = $W;
                const B: usize = 32 * W;
                let value: u32 = ((1u64 << W) - 1) as u32;
                let blocks = make_constant_packed_blocks::<W, B>(value);
                let mut out: Vec<u64> = vec![0u64; 16 * EQ_BLOCKS];

                bencher
                    .counter(ItemsCount::new(1024usize * EQ_BLOCKS))
                    .bench_local(|| {
                        fastlanes_unpack_cmp_per1k_shared::<W, B>(
                            &blocks,
                            black_box(value),
                            &mut out,
                        );
                        black_box(&mut out);
                    });
            }

            // Per-1k collect using BitPacking::unpack into stack [u32; 1024].
            #[divan::bench]
            fn [<$prefix _stack_unpack_per1k_w $W>](bencher: Bencher) {
                const W: usize = $W;
                const B: usize = 32 * W;
                let value: u32 = ((1u64 << W) - 1) as u32;
                let blocks = make_constant_packed_blocks::<W, B>(value);
                let mut out: Vec<u64> = vec![0u64; 16 * EQ_BLOCKS];

                bencher
                    .counter(ItemsCount::new(1024usize * EQ_BLOCKS))
                    .bench_local(|| {
                        stack_unpack_per1k_shared::<W, B>(
                            &blocks,
                            black_box(value),
                            &mut out,
                        );
                        black_box(&mut out);
                    });
            }
        }
    };
}

u32_eq_benches!(u32_eq => 1);
u32_eq_benches!(u32_eq => 2);
u32_eq_benches!(u32_eq => 3);
u32_eq_benches!(u32_eq => 4);
u32_eq_benches!(u32_eq => 5);
u32_eq_benches!(u32_eq => 7);
u32_eq_benches!(u32_eq => 8);
u32_eq_benches!(u32_eq => 11);
u32_eq_benches!(u32_eq => 12);
u32_eq_benches!(u32_eq => 16);
u32_eq_benches!(u32_eq => 17);
u32_eq_benches!(u32_eq => 23);
u32_eq_benches!(u32_eq => 24);
u32_eq_benches!(u32_eq => 29);

// v6 specializations (W=8 byte-cmp, W=16 word-cmp) — explicit benches.
#[divan::bench]
fn u32_eq_block_batch_avx512_v6_w8(bencher: Bencher) {
    #[cfg(target_arch = "x86_64")]
    {
        if !is_x86_feature_detected!("avx512f")
            || !is_x86_feature_detected!("avx512bw")
            || !is_x86_feature_detected!("bmi2")
        {
            return;
        }
        const W: usize = 8;
        const B: usize = 32 * W;
        let value: u32 = 0xA5;
        let blocks = make_constant_packed_blocks::<W, B>(value);
        let mut out: Vec<u64> = vec![0u64; 16 * EQ_BLOCKS];
        bencher
            .counter(ItemsCount::new(1024usize * EQ_BLOCKS))
            .bench_local(|| {
                // SAFETY: feature-gated.
                unsafe { block_eq_batch_avx512_v6_w8::<B>(&blocks, black_box(value), &mut out) };
                black_box(&mut out);
            });
    }
}

#[divan::bench]
fn u32_eq_block_batch_avx512_v6_w16(bencher: Bencher) {
    #[cfg(target_arch = "x86_64")]
    {
        if !is_x86_feature_detected!("avx512f")
            || !is_x86_feature_detected!("avx512bw")
            || !is_x86_feature_detected!("bmi2")
        {
            return;
        }
        const W: usize = 16;
        const B: usize = 32 * W;
        let value: u32 = 0xBEEF;
        let blocks = make_constant_packed_blocks::<W, B>(value);
        let mut out: Vec<u64> = vec![0u64; 16 * EQ_BLOCKS];
        bencher
            .counter(ItemsCount::new(1024usize * EQ_BLOCKS))
            .bench_local(|| {
                // SAFETY: feature-gated.
                unsafe { block_eq_batch_avx512_v6_w16::<B>(&blocks, black_box(value), &mut out) };
                black_box(&mut out);
            });
    }
}
