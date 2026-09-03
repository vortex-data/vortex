// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Minimal, dependency-free reproduction of the miscompile behind the `c7gd.metal` (Graviton3)
//! benchmark failures: a widening sum of `u8` values into `usize`/`u64` returns the wrong
//! total when the binary is built with `-C target-cpu=neoverse-v1` (what `-C target-cpu=native`
//! resolves to on Graviton3) and runs with an SVE vector length of 256 bits or more. The same
//! binary is correct at a 128-bit vector length, and `u16`/`u32`/`i32` sums are correct at
//! every length.
//!
//! Vortex hits this in `OnPairDecodePlan::new` and `FsstDecodePlan::new`, which size their
//! output buffer as the sum of the `u8` `uncompressed_lengths` child of arrays read back from a
//! file; the undersized buffer surfaces as `OnPair codes decode to more bytes than
//! uncompressed_lengths records`, `FSST decoded N bytes, expected M`, and the `fsst-rs`
//! `output buffer sized too small` panic.
//!
//! ```text
//! RUSTFLAGS="-C target-cpu=neoverse-v1" cargo build --release \
//!     -p string-bench --example sve_widening_sum --features unstable_encodings
//! target/release/examples/sve_widening_sum          # on Graviton3, or under
//! qemu-aarch64 -cpu max,sve-default-vector-length=32 target/.../sve_widening_sum
//! ```
//!
//! Exits non-zero and prints every mismatching input length. Building with
//! `-C target-feature=-sve,-sve2` makes it pass.

use std::hint::black_box;
use std::process::ExitCode;

#[inline(never)]
fn sum_u8_as_usize(xs: &[u8]) -> usize {
    xs.iter().map(|&x| x as usize).sum()
}

#[inline(never)]
fn sum_u8_as_u64(xs: &[u8]) -> u64 {
    xs.iter().map(|&x| x as u64).sum()
}

#[inline(never)]
fn sum_u16_as_usize(xs: &[u16]) -> usize {
    xs.iter().map(|&x| x as usize).sum()
}

#[inline(never)]
fn sum_u32_as_usize(xs: &[u32]) -> usize {
    xs.iter().map(|&x| x as usize).sum()
}

/// Reference total: the accumulator goes through `black_box` every step so the loop cannot be
/// vectorized.
#[inline(never)]
fn reference(xs: &[u8]) -> usize {
    let mut acc = 0usize;
    for &x in xs {
        acc = black_box(acc + x as usize);
    }
    acc
}

fn main() -> ExitCode {
    // Deterministic pseudo-random lengths in 19..=250, the range of ClickBench URL lengths.
    let mut seed = 0x9E37_79B9_7F4A_7C15_u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        #[allow(clippy::cast_possible_truncation)]
        let byte = (seed % 232) as u8;
        19 + byte
    };

    let mut mismatches = 0;
    for n in [
        1usize, 7, 8, 15, 16, 31, 32, 33, 63, 64, 65, 100, 127, 128, 129, 255, 256, 257, 1000,
        1023, 1024, 4096, 8191, 8192, 8193, 100_000,
    ] {
        let xs: Vec<u8> = (0..n).map(|_| next()).collect();
        let xs = black_box(xs);
        let want = reference(&xs);
        let u8_usize = sum_u8_as_usize(&xs);
        let u8_u64 = usize::try_from(sum_u8_as_u64(&xs)).unwrap_or(usize::MAX);
        let u16_usize = sum_u16_as_usize(&xs.iter().map(|&x| u16::from(x)).collect::<Vec<_>>());
        let u32_usize = sum_u32_as_usize(&xs.iter().map(|&x| u32::from(x)).collect::<Vec<_>>());
        let ok = u8_usize == want && u8_u64 == want && u16_usize == want && u32_usize == want;
        if !ok {
            mismatches += 1;
            println!(
                "n={n}: reference={want} u8->usize={u8_usize} u8->u64={u8_u64} \
                 u16->usize={u16_usize} u32->usize={u32_usize}"
            );
        }
    }
    println!("mismatches: {mismatches}");
    if mismatches == 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
