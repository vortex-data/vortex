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
//!   block_batch_avx512_hand       19.3 18.0 14.8 13.1 12.5  <- splat algo, hand AVX-512
//!   block_unpack_avx512_hand      12.3 11.8 10.8 10.3  9.5  <- unpack algo, hand AVX-512
//!   stack_unpack_avx512            7.3  7.7  7.4  7.3  7.1  <- decoupled (no fusion)
//!   stack_unpack_per1k             5.84 6.44 6.27 6.22 6.02 <- best auto-vec
//!   fl_unpack_cmp_per1k            5.60 6.04 6.01 4.91 6.07
//!   fl_unpack_cmp_batch            2.93 2.88 2.92 2.86 2.86
//!   fl_unpack_cmp_collect          2.58 2.59 2.49 2.66 2.52
//!   stack_unpack_collect           2.66 2.67 2.59 2.62 2.55
//!   block_batch (mine, auto-vec)   1.72 1.37 1.67 1.68 1.48
//! ```
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
    check_eq!(4);
    check_eq!(12);
    check_eq!(16);
    check_eq!(24);
}

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
    __m128i, __m512i, _mm512_and_si512, _mm512_cmpeq_epu32_mask, _mm512_loadu_si512,
    _mm512_or_si512, _mm512_set1_epi32, _mm512_setzero_si512, _mm512_sll_epi32, _mm512_srl_epi32,
    _mm512_xor_si512, _mm_cvtsi32_si128,
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
u32_eq_benches!(u32_eq => 4);
u32_eq_benches!(u32_eq => 12);
u32_eq_benches!(u32_eq => 16);
u32_eq_benches!(u32_eq => 24);
