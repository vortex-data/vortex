//! Microbenchmark: delta-undo on a 1024-byte block in the step-major
//! 32-lane × 32-step layout described in `src/delta.rs`.
//!
//! Compares:
//!   1. Scalar reference (per-lane carry, autovectorization is up to LLVM).
//!   2. Hand-written AVX2 intrinsics (one `vpaddb` per step).
//!   3. A "stencil-JIT" entry that currently forwards to the AVX2 version.
//!      A follow-up session will replace this with a copy-and-patch JIT.
//!
//! Reports ns/call and GB/s on the 1024-byte block.

#![cfg(all(target_arch = "x86_64", target_os = "linux"))]

use std::hint::black_box;
use std::time::Instant;

use stencil_jit::delta::{BLOCK, LANES, undelta_avx2, undelta_jit_placeholder, undelta_scalar};

const ITERS: usize = 2_000_000;

fn fill_pseudo_random(buf: &mut [u8], mut seed: u32) {
    for byte in buf.iter_mut() {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        *byte = seed as u8;
    }
}

fn time_loop(label: &str, mut body: impl FnMut() -> u8) {
    for _ in 0..1_000 {
        black_box(body());
    }
    let t0 = Instant::now();
    let mut sink: u8 = 0;
    for _ in 0..ITERS {
        sink = sink.wrapping_add(black_box(body()));
    }
    let elapsed = t0.elapsed();
    black_box(sink);
    let ns_per_call = elapsed.as_nanos() as f64 / ITERS as f64;
    let bytes_per_sec = (ITERS as f64 * BLOCK as f64) / elapsed.as_secs_f64();
    println!(
        "  {:32}  {:7.2} ns/call  {:7.2} GB/s",
        label,
        ns_per_call,
        bytes_per_sec / 1e9,
    );
}

fn main() {
    if !is_x86_feature_detected!("avx2") {
        eprintln!("AVX2 is required for this bench");
        return;
    }

    let mut input = vec![0u8; BLOCK].into_boxed_slice();
    let mut base = [0u8; LANES];
    fill_pseudo_random(&mut input, 0xC0FFEE);
    fill_pseudo_random(&mut base, 0xBADF00D);

    let input: Box<[u8; BLOCK]> = input.try_into().expect("size matches");
    let mut output = vec![0u8; BLOCK].into_boxed_slice();
    let output_ref: &mut [u8; BLOCK] = (&mut output[..]).try_into().expect("size matches");

    println!("Delta-undo: 32-lane x 32-step step-major u8 block ({BLOCK} bytes)");
    println!("Iters: {ITERS}");
    println!();

    // Cross-check before timing.
    let mut a = vec![0u8; BLOCK].into_boxed_slice();
    let mut b = vec![0u8; BLOCK].into_boxed_slice();
    let a_ref: &mut [u8; BLOCK] = (&mut a[..]).try_into().unwrap();
    let b_ref: &mut [u8; BLOCK] = (&mut b[..]).try_into().unwrap();
    undelta_scalar(&input, &base, a_ref);
    // SAFETY: AVX2 verified above.
    unsafe { undelta_avx2(&input, &base, b_ref) };
    assert_eq!(a, b, "scalar and AVX2 disagree");

    println!("Timings:");

    time_loop("scalar baseline", || {
        undelta_scalar(&input, &base, output_ref);
        output_ref[0]
    });

    time_loop("aot intrinsics avx2", || {
        // SAFETY: AVX2 verified above.
        unsafe { undelta_avx2(&input, &base, output_ref) };
        output_ref[0]
    });

    time_loop("stencil-jit (placeholder)", || {
        // SAFETY: AVX2 verified above. Real JIT goes here later.
        unsafe { undelta_jit_placeholder(&input, &base, output_ref) };
        output_ref[0]
    });
}

#[cfg(not(all(target_arch = "x86_64", target_os = "linux")))]
fn main() {
    eprintln!("bench_delta requires x86_64 Linux + AVX2");
}
