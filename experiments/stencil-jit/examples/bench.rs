//! Microbenchmark: chain "FFoR-add + compare(==)" on 32 packed u8 lanes,
//! comparing:
//!   1. The stencil-JIT'd kernel (copy-and-patch).
//!   2. An AOT Rust function with a closure for the op (rustc/LLVM
//!      autovectorizes; this is the "AOT fused" reference).
//!   3. A hand-written AVX2-intrinsics version (apples-to-apples to the JIT).
//!   4. A scalar baseline.
//!
//! Reports ns/call and GB/s on the 32-byte block. Also prints the one-time
//! cost to compile a JIT kernel (mmap + memcpy + mprotect).

#![cfg(all(target_arch = "x86_64", target_os = "linux"))]

use std::arch::x86_64::*;
use std::hint::black_box;
use std::time::Instant;

use stencil_jit::{ChainConfig, CmpOp, Kernel};

const ITERS: usize = 50_000_000;

#[inline(never)]
fn aot_closure_eq(packed: &[u8; 32], ffor_ref: u8, constant: u8) -> u32 {
    aot_closure(packed, ffor_ref, constant, |a, b| a == b)
}

#[inline(never)]
fn aot_closure<F: Fn(u8, u8) -> bool>(
    packed: &[u8; 32],
    ffor_ref: u8,
    constant: u8,
    f: F,
) -> u32 {
    let mut mask = 0u32;
    for (i, &b) in packed.iter().enumerate() {
        if f(b.wrapping_add(ffor_ref), constant) {
            mask |= 1u32 << i;
        }
    }
    mask
}

#[inline(never)]
#[target_feature(enable = "avx2")]
unsafe fn aot_intrinsics_ffor_eq(packed: *const u8, ffor_ref: u8, constant: u8) -> u32 {
    // SAFETY: caller guarantees 32 readable bytes at `packed`.
    let data = unsafe { _mm256_loadu_si256(packed as *const __m256i) };
    let r = _mm256_set1_epi8(ffor_ref as i8);
    let c = _mm256_set1_epi8(constant as i8);
    let shifted = _mm256_add_epi8(data, r);
    let eq = _mm256_cmpeq_epi8(shifted, c);
    _mm256_movemask_epi8(eq) as u32
}

#[inline(never)]
fn scalar_eq(packed: &[u8; 32], ffor_ref: u8, constant: u8) -> u32 {
    let mut mask = 0u32;
    for (i, &b) in packed.iter().enumerate() {
        if b.wrapping_add(ffor_ref) == constant {
            mask |= 1u32 << i;
        }
    }
    mask
}

fn time_loop(label: &str, mut body: impl FnMut() -> u32) {
    // Warm-up
    for _ in 0..1_000 {
        black_box(body());
    }
    let t0 = Instant::now();
    let mut sink = 0u32;
    for _ in 0..ITERS {
        sink = sink.wrapping_add(black_box(body()));
    }
    let elapsed = t0.elapsed();
    black_box(sink);
    let ns_per_call = elapsed.as_nanos() as f64 / ITERS as f64;
    let bytes_per_sec = (ITERS as f64 * 32.0) / elapsed.as_secs_f64();
    println!(
        "  {:32}  {:6.2} ns/call  {:7.2} GB/s",
        label,
        ns_per_call,
        bytes_per_sec / 1e9,
    );
}

fn main() {
    let packed: [u8; 32] = core::array::from_fn(|i| (i as u8).wrapping_mul(17).wrapping_add(3));
    let ffor_ref: u8 = 7;
    let constant: u8 = 42;

    println!("Chain: FFoR-add + compare(==) on 32 packed u8 lanes");
    println!("Iters: {ITERS}");
    println!();

    // One-time JIT compile cost.
    let t0 = Instant::now();
    let kernel = Kernel::compile(ChainConfig::ffor_then_compare(CmpOp::Eq)).expect("compile");
    let compile_ns = t0.elapsed().as_nanos();
    println!("JIT compile cost (one-time): {} ns", compile_ns);

    // Compile a second, to amortize mmap warm-up out of the headline number.
    let t0 = Instant::now();
    let _kernel2 = Kernel::compile(ChainConfig::ffor_then_compare(CmpOp::Eq)).expect("compile");
    println!("JIT compile cost (warm):     {} ns", t0.elapsed().as_nanos());
    println!();

    // Quick correctness cross-check before timing.
    let mut out: u32 = 0;
    // SAFETY: 32 readable + 4 writable.
    unsafe { kernel.call(packed.as_ptr(), constant, &mut out as *mut u32, ffor_ref) };
    let want = scalar_eq(&packed, ffor_ref, constant);
    assert_eq!(out, want, "kernel disagrees with scalar reference");

    println!("Timings:");

    let pkt = &packed;
    time_loop("stencil-jit (FFoR + eq)", || {
        let mut out: u32 = 0;
        // SAFETY: same as above.
        unsafe { kernel.call(pkt.as_ptr(), constant, &mut out as *mut u32, ffor_ref) };
        out
    });

    time_loop("aot closure-based eq", || {
        aot_closure_eq(pkt, ffor_ref, constant)
    });

    time_loop("aot intrinsics avx2 eq", || {
        // SAFETY: 32 readable bytes; AVX2 verified available at runtime.
        unsafe { aot_intrinsics_ffor_eq(pkt.as_ptr(), ffor_ref, constant) }
    });

    time_loop("scalar baseline eq", || scalar_eq(pkt, ffor_ref, constant));
}

#[cfg(not(all(target_arch = "x86_64", target_os = "linux")))]
fn main() {
    eprintln!("bench requires x86_64 Linux + AVX2");
}
