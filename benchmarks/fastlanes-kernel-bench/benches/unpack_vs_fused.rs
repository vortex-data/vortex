// SPDX-FileCopyrightText: Copyright the Vortex contributors
// SPDX-License-Identifier: Apache-2.0

//! Microbenchmarks for the FastLanes 1024-element unpack kernel.
//!
//! Three variants are measured for every (type, bit width) pair:
//!
//! 1. `bare_unpack` -- `BitPacking::unpack` only (no FoR step at all).
//! 2. `unfused_for` -- `BitPacking::unpack` followed by a separate
//!    `for i in 0..1024 { out[i] = out[i].wrapping_add(reference) }` pass.
//! 3. `fused_for`   -- `FoR::unfor_pack` (FoR reference application fused into
//!    the unpack kernel via the upstream macro).
//!
//! Every input/output buffer is pre-allocated outside the timed closure, so the
//! divan measurement covers nothing but the kernel itself: no Vortex array
//! plumbing, no `Buffer` allocation, no validity handling, no patches.
//!
//! Bit width 0 is skipped (degenerate -- packed array is zero bytes).
//! Bit width T is the identity case and is included for completeness.

#![allow(clippy::all)]

use std::hint::black_box;

use divan::Bencher;
use fastlanes_kernel_bench::BitPacking;
use fastlanes_kernel_bench::FastLanes;
use fastlanes_kernel_bench::FoR;

fn main() {
    divan::main();
}

// One reference value per type. Picked to be non-trivial so that LLVM cannot
// fold the `wrapping_add` into a no-op.
const REF_U8: u8 = 7;
const REF_U16: u16 = 1_009;
const REF_U32: u32 = 1_000_003;
const REF_U64: u64 = 1_000_000_007;

/// Generate `BARE`, `UNFUSED`, and `FUSED` benchmarks for a single
/// `(type, bit-width)` pair. Each benchmark uses fixed-size stack buffers so
/// that nothing is allocated inside the timed closure.
macro_rules! gen_bench {
    ($T:ty, $W:literal, $REF:expr) => {
        paste::paste! {
            #[divan::bench]
            #[allow(non_snake_case)]
            fn [<bare_unpack__ $T __w $W>](bencher: Bencher) {
                const W: usize = $W;
                const B: usize = 1024 * W / <$T>::T;
                let mut input = [0 as $T; 1024];
                for (i, v) in input.iter_mut().enumerate() {
                    *v = i as $T;
                }
                let mut packed = [0 as $T; B];
                <$T as BitPacking>::pack::<W, B>(&input, &mut packed);
                let mut output = [0 as $T; 1024];

                bencher.bench_local(|| {
                    <$T as BitPacking>::unpack::<W, B>(black_box(&packed), &mut output);
                    black_box(&mut output);
                });
            }

            #[divan::bench]
            #[allow(non_snake_case)]
            fn [<unfused_for__ $T __w $W>](bencher: Bencher) {
                const W: usize = $W;
                const B: usize = 1024 * W / <$T>::T;
                let mut input = [0 as $T; 1024];
                for (i, v) in input.iter_mut().enumerate() {
                    *v = i as $T;
                }
                let mut packed = [0 as $T; B];
                <$T as BitPacking>::pack::<W, B>(&input, &mut packed);
                let mut output = [0 as $T; 1024];
                let reference: $T = $REF;

                bencher.bench_local(|| {
                    <$T as BitPacking>::unpack::<W, B>(black_box(&packed), &mut output);
                    for v in output.iter_mut() {
                        *v = v.wrapping_add(reference);
                    }
                    black_box(&mut output);
                });
            }

            #[divan::bench]
            #[allow(non_snake_case)]
            fn [<fused_for__ $T __w $W>](bencher: Bencher) {
                const W: usize = $W;
                const B: usize = 1024 * W / <$T>::T;
                let mut input = [0 as $T; 1024];
                for (i, v) in input.iter_mut().enumerate() {
                    *v = i as $T;
                }
                let reference: $T = $REF;
                let mut packed = [0 as $T; B];
                <$T as FoR>::for_pack::<W, B>(&input, reference, &mut packed);
                let mut output = [0 as $T; 1024];

                bencher.bench_local(|| {
                    <$T as FoR>::unfor_pack::<W, B>(black_box(&packed), reference, &mut output);
                    black_box(&mut output);
                });
            }
        }
    };
}

macro_rules! gen_for_type {
    ($T:ty, $REF:expr, $($W:literal),+ $(,)?) => {
        $( gen_bench!($T, $W, $REF); )+
    };
}

// All bit widths in 1..=T for every supported unsigned type.
// Signed types (i8/i16/i32/i64) are intentionally not benchmarked: at the bit
// level they are identical to the matching unsigned width, and the upstream
// vortex-fastlanes integration handles them by `reinterpret_cast`/transmute.
// See README.md for the justification.
gen_for_type!(u8, REF_U8, 1, 2, 3, 4, 5, 6, 7, 8);
gen_for_type!(
    u16, REF_U16, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
);
gen_for_type!(
    u32, REF_U32, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
    24, 25, 26, 27, 28, 29, 30, 31, 32
);
gen_for_type!(
    u64, REF_U64, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
    24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
    48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64
);
