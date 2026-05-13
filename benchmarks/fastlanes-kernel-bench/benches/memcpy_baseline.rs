// SPDX-FileCopyrightText: Copyright the Vortex contributors
// SPDX-License-Identifier: Apache-2.0

//! Pure memcpy baseline: a `std::ptr::copy_nonoverlapping` of the same byte
//! volume that a single 1024-element `(T, W)` unpack reads + writes. This is
//! the absolute memory lower bound -- no kernel can be faster than copying
//! the same byte count between two L1-resident buffers.
//!
//! For each `(T, W)` we copy `(1024 * W / 8) + (1024 * T / 8)` bytes, which
//! is exactly the packed-input size plus the unpacked-output size.

#![allow(clippy::all)]

use std::hint::black_box;

use divan::Bencher;

fn main() {
    divan::main();
}

/// Generate a `memcpy` bench whose copy length equals the per-call byte
/// volume of an unpack: `(1024 * W / 8) + (1024 * T / 8)`.
macro_rules! gen_bench {
    ($T:ty, $W:literal) => {
        paste::paste! {
            #[divan::bench]
            #[allow(non_snake_case)]
            fn [<memcpy__ $T __w $W>](bencher: Bencher) {
                const W: usize = $W;
                const T_BITS: usize = std::mem::size_of::<$T>() * 8;
                const BYTES: usize = (1024 * W / 8) + (1024 * T_BITS / 8);
                // Two heap buffers so they're not subject to stack alignment
                // games, but kept around the same size as the matched unpack.
                let src = vec![0u8; BYTES];
                let mut dst = vec![0u8; BYTES];
                bencher.bench_local(|| {
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            black_box(src.as_ptr()),
                            dst.as_mut_ptr(),
                            BYTES,
                        );
                    }
                    black_box(&mut dst);
                });
            }
        }
    };
}

macro_rules! gen_for_type {
    ($T:ty, $($W:literal),+ $(,)?) => {
        $( gen_bench!($T, $W); )+
    };
}

gen_for_type!(u8, 1, 2, 3, 4, 5, 6, 7, 8);
gen_for_type!(u16, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
gen_for_type!(
    u32, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31, 32
);
gen_for_type!(
    u64, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
    50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64
);
