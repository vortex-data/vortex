// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::use_debug)]

//! Helper to inspect assembly output for transpose functions.
//!
//! Compile with: RUSTFLAGS="-C target-cpu=native" cargo build --release --example check_asm -p vortex-fastlanes
//! Then disassemble: objdump -d target/release/examples/check_asm | grep -A 200 "transpose_1024"

use std::hint::black_box;

use vortex_fastlanes::transpose;

fn main() {
    let input = [42u8; 128];
    let mut output = [0u8; 128];

    // Call each function to ensure it's not optimized away
    transpose::transpose_1024_baseline(black_box(&input), black_box(&mut output));
    println!("baseline: {:?}", &output[..8]);

    transpose::transpose_1024_scalar(black_box(&input), black_box(&mut output));
    println!("scalar: {:?}", &output[..8]);

    #[cfg(target_arch = "x86_64")]
    {
        use vortex_fastlanes::transpose::x86;

        if x86::has_avx2() {
            unsafe { x86::transpose_1024_avx2(black_box(&input), black_box(&mut output)) };
            println!("avx2: {:?}", &output[..8]);
        }

        if x86::has_avx2() && x86::has_gfni() {
            unsafe { x86::transpose_1024_avx2_gfni(black_box(&input), black_box(&mut output)) };
            println!("avx2_gfni: {:?}", &output[..8]);
        }

        if x86::has_avx512() && x86::has_gfni() {
            unsafe { x86::transpose_1024_avx512_gfni(black_box(&input), black_box(&mut output)) };
            println!("avx512_gfni: {:?}", &output[..8]);
        }
    }
}
