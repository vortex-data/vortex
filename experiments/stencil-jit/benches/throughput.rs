//! Throughput benchmark: bulk-mode FFoR-add + compare(==) on 32 packed u8
//! lanes per block, comparing the stencil-JIT kernel against AOT
//! alternatives.
//!
//! Kernels are compiled once (cost ignored). Each measured iteration
//! processes `N_BLOCKS` 32-byte blocks per call. Throughput in GB/s is the
//! relevant number; ns/call is also reported.

#![cfg(all(target_arch = "x86_64", target_os = "linux"))]

use std::arch::x86_64::*;
use std::hint::black_box;

use divan::{Bencher, counter::BytesCount};
use stencil_jit::{BulkKernel, ChainConfig, CmpOp, Kernel};

const N_BLOCKS: usize = 1024;
const BYTES: usize = N_BLOCKS * 32;

fn make_input() -> Vec<u8> {
    (0..BYTES)
        .map(|i| (i as u8).wrapping_mul(17).wrapping_add(3))
        .collect()
}

fn bytes_counter() -> BytesCount {
    BytesCount::new(BYTES)
}

// ---------- stencil-JIT, bulk -------------------------------------------------

#[divan::bench]
fn stencil_jit_bulk(bencher: Bencher) {
    let kernel = BulkKernel::compile(ChainConfig::ffor_then_compare(CmpOp::Eq)).unwrap();
    let input = make_input();
    let mut out = vec![0u32; N_BLOCKS];
    bencher.counter(bytes_counter()).bench_local(|| {
        // SAFETY: input is BYTES readable, out is N_BLOCKS * 4 writable.
        unsafe {
            kernel.call(
                black_box(input.as_ptr()),
                black_box(42u8),
                black_box(out.as_mut_ptr()),
                black_box(7u8),
                N_BLOCKS,
            )
        };
        black_box(out[0])
    });
}

// ---------- stencil-JIT, single-block (called in a loop) ----------------------

#[divan::bench]
fn stencil_jit_per_block_loop(bencher: Bencher) {
    let kernel = Kernel::compile(ChainConfig::ffor_then_compare(CmpOp::Eq)).unwrap();
    let input = make_input();
    let mut out = vec![0u32; N_BLOCKS];
    bencher.counter(bytes_counter()).bench_local(|| {
        for i in 0..N_BLOCKS {
            // SAFETY: 32-byte block + 4-byte u32 out.
            unsafe {
                kernel.call(
                    input.as_ptr().add(i * 32),
                    black_box(42u8),
                    out.as_mut_ptr().add(i),
                    black_box(7u8),
                );
            }
        }
        black_box(out[0])
    });
}

// ---------- AOT: AVX2 intrinsics, bulk loop in Rust ---------------------------

#[inline(never)]
#[target_feature(enable = "avx2")]
unsafe fn aot_intrinsics_bulk(
    packed: *const u8,
    ffor_ref: u8,
    constant: u8,
    out: *mut u32,
    n_blocks: usize,
) {
    let r = _mm256_set1_epi8(ffor_ref as i8);
    let c = _mm256_set1_epi8(constant as i8);
    for i in 0..n_blocks {
        // SAFETY: caller guarantees n_blocks * 32 readable, n_blocks * 4 writable.
        let data = unsafe { _mm256_loadu_si256(packed.add(i * 32) as *const __m256i) };
        let shifted = _mm256_add_epi8(data, r);
        let eq = _mm256_cmpeq_epi8(shifted, c);
        // SAFETY: as above for `out`.
        unsafe { *out.add(i) = _mm256_movemask_epi8(eq) as u32 };
    }
}

#[divan::bench]
fn aot_intrinsics(bencher: Bencher) {
    let input = make_input();
    let mut out = vec![0u32; N_BLOCKS];
    bencher.counter(bytes_counter()).bench_local(|| {
        // SAFETY: buffers sized correctly above.
        unsafe {
            aot_intrinsics_bulk(
                black_box(input.as_ptr()),
                black_box(7u8),
                black_box(42u8),
                black_box(out.as_mut_ptr()),
                N_BLOCKS,
            )
        };
        black_box(out[0])
    });
}

// ---------- AOT: closure-based scalar (rustc autovec) -------------------------

#[inline(never)]
fn aot_closure_bulk<F: Fn(u8, u8) -> bool>(
    packed: &[u8],
    ffor_ref: u8,
    constant: u8,
    out: &mut [u32],
    f: F,
) {
    debug_assert_eq!(out.len() * 32, packed.len());
    for (block_i, chunk) in packed.chunks_exact(32).enumerate() {
        let mut mask = 0u32;
        for (i, &b) in chunk.iter().enumerate() {
            if f(b.wrapping_add(ffor_ref), constant) {
                mask |= 1u32 << i;
            }
        }
        out[block_i] = mask;
    }
}

#[divan::bench]
fn aot_closure(bencher: Bencher) {
    let input = make_input();
    let mut out = vec![0u32; N_BLOCKS];
    bencher.counter(bytes_counter()).bench_local(|| {
        aot_closure_bulk(
            black_box(input.as_slice()),
            black_box(7u8),
            black_box(42u8),
            out.as_mut_slice(),
            |a, b| a == b,
        );
        black_box(out[0])
    });
}

// ---------- Scalar baseline ---------------------------------------------------

#[inline(never)]
fn scalar_bulk(packed: &[u8], ffor_ref: u8, constant: u8, out: &mut [u32]) {
    for (block_i, chunk) in packed.chunks_exact(32).enumerate() {
        let mut mask = 0u32;
        for (i, &b) in chunk.iter().enumerate() {
            if b.wrapping_add(ffor_ref) == constant {
                mask |= 1u32 << i;
            }
        }
        out[block_i] = mask;
    }
}

#[divan::bench]
fn scalar(bencher: Bencher) {
    let input = make_input();
    let mut out = vec![0u32; N_BLOCKS];
    bencher.counter(bytes_counter()).bench_local(|| {
        scalar_bulk(
            black_box(input.as_slice()),
            black_box(7u8),
            black_box(42u8),
            out.as_mut_slice(),
        );
        black_box(out[0])
    });
}

fn main() {
    divan::main();
}
