// SPDX-FileCopyrightText: Copyright the Vortex contributors
// SPDX-License-Identifier: Apache-2.0

//! Multi-block throughput: time **N consecutive 1024-element blocks** in one
//! bench closure so that per-block function-call overhead is amortised.
//!
//! Compare to the single-block matrix: if `multi_block_time / N` is much
//! smaller than the single-block time, our single-block measurements have
//! been dominated by call overhead.
//!
//! N = 8 blocks per closure.

#![allow(clippy::all)]

use std::hint::black_box;

use divan::Bencher;
use fastlanes_kernel_bench::BitPacking;
use fastlanes_kernel_bench::FastLanes;
use fastlanes_kernel_bench::FoR;

fn main() {
    divan::main();
}

const N: usize = 8;

const REF_U8: u8 = 7;
const REF_U16: u16 = 1_009;
const REF_U32: u32 = 1_000_003;
const REF_U64: u64 = 1_000_000_007;

macro_rules! gen_bench {
    ($T:ty, $W:literal, $REF:expr) => {
        paste::paste! {
            #[divan::bench]
            #[allow(non_snake_case)]
            fn [<bare_unpack_n8__ $T __w $W>](bencher: Bencher) {
                const W: usize = $W;
                const B: usize = 1024 * W / <$T>::T;
                let mut input = [0 as $T; 1024];
                for (i, v) in input.iter_mut().enumerate() {
                    *v = i as $T;
                }
                let mut packed = [0 as $T; B];
                <$T as BitPacking>::pack::<W, B>(&input, &mut packed);
                // N independent output buffers so we exercise N * 1024 outputs.
                let mut outputs: Vec<[$T; 1024]> = (0..N).map(|_| [0 as $T; 1024]).collect();

                bencher.bench_local(|| {
                    for i in 0..N {
                        <$T as BitPacking>::unpack::<W, B>(black_box(&packed), &mut outputs[i]);
                    }
                    black_box(&mut outputs);
                });
            }

            #[divan::bench]
            #[allow(non_snake_case)]
            fn [<fused_for_n8__ $T __w $W>](bencher: Bencher) {
                const W: usize = $W;
                const B: usize = 1024 * W / <$T>::T;
                let mut input = [0 as $T; 1024];
                for (i, v) in input.iter_mut().enumerate() {
                    *v = i as $T;
                }
                let reference: $T = $REF;
                let mut packed = [0 as $T; B];
                <$T as FoR>::for_pack::<W, B>(&input, reference, &mut packed);
                let mut outputs: Vec<[$T; 1024]> = (0..N).map(|_| [0 as $T; 1024]).collect();

                bencher.bench_local(|| {
                    for i in 0..N {
                        <$T as FoR>::unfor_pack::<W, B>(
                            black_box(&packed),
                            reference,
                            &mut outputs[i],
                        );
                    }
                    black_box(&mut outputs);
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

// A modest subset of W per type -- enough to cover narrow/mid/wide and identity.
gen_for_type!(u8, REF_U8, 1, 3, 5, 8);
gen_for_type!(u16, REF_U16, 1, 4, 7, 11, 15, 16);
gen_for_type!(u32, REF_U32, 1, 5, 8, 10, 17, 24, 25, 32);
gen_for_type!(u64, REF_U64, 1, 8, 11, 33, 55, 64);
