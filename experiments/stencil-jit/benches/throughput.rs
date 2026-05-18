//! Throughput benchmark for the FFoR-add + compare(==) chain on packed u8
//! lanes. Compares:
//!
//!   * Fused AOT (the ceiling — works only when the chain is known at
//!     compile time and someone wrote the intrinsics).
//!   * Unfused AOT pipeline (two single-op kernels chained via an
//!     intermediate scratch buffer — the realistic baseline when chains
//!     are runtime-defined).
//!   * Stencil-JIT bulk kernel (this prototype).
//!   * Scalar / closure-based reference.
//!
//! Each variant runs for several working-set sizes so we can see when the
//! unfused pipeline's intermediate buffer starts spilling out of cache.
//! At small sizes (scratch in L1) the unfused pipeline is competitive;
//! at larger sizes its second pass costs real memory bandwidth.

#![cfg(all(target_arch = "x86_64", target_os = "linux"))]

use std::arch::x86_64::*;
use std::hint::black_box;

use divan::{Bencher, counter::BytesCount};
use stencil_jit::{
    BulkKernel, ChainConfig, CmpOp, Kernel, SpecializedKernel, SpecializedKernel512,
    cranelift_kernel::CraneliftKernel,
};

/// Working-set sizes (in 32-byte blocks) chosen to span L1, L2, and beyond.
/// 128   blocks = 4 KB     (well inside L1)
/// 1024  blocks = 32 KB    (≈ L1 size on Skylake)
/// 8192  blocks = 256 KB   (inside L2)
/// 32768 blocks = 1 MB     (≈ L2 size; intermediate-buffer spill cost is real)
/// 131072 blocks = 4 MB    (inside L3)
const SIZES: &[usize] = &[128, 1024, 8192, 32768, 131072];

fn make_input(n_blocks: usize) -> Vec<u8> {
    (0..n_blocks * 32)
        .map(|i| (i as u8).wrapping_mul(17).wrapping_add(3))
        .collect()
}

fn counter(n_blocks: usize) -> BytesCount {
    BytesCount::new(n_blocks * 32)
}

// ---------- AOT: fused (the ceiling) -----------------------------------------

#[inline(never)]
#[target_feature(enable = "avx2")]
unsafe fn aot_intrinsics_fused(
    packed: *const u8,
    ffor_ref: u8,
    constant: u8,
    out: *mut u32,
    n_blocks: usize,
) {
    let r = _mm256_set1_epi8(ffor_ref as i8);
    let c = _mm256_set1_epi8(constant as i8);
    for i in 0..n_blocks {
        // SAFETY: caller guarantees buffer sizes.
        let data = unsafe { _mm256_loadu_si256(packed.add(i * 32) as *const __m256i) };
        let shifted = _mm256_add_epi8(data, r);
        let eq = _mm256_cmpeq_epi8(shifted, c);
        unsafe { *out.add(i) = _mm256_movemask_epi8(eq) as u32 };
    }
}

#[divan::bench(args = SIZES)]
fn aot_fused(bencher: Bencher, n_blocks: usize) {
    let input = make_input(n_blocks);
    let mut out = vec![0u32; n_blocks];
    bencher.counter(counter(n_blocks)).bench_local(|| {
        // SAFETY: buffers sized correctly above.
        unsafe {
            aot_intrinsics_fused(
                black_box(input.as_ptr()),
                black_box(7u8),
                black_box(42u8),
                black_box(out.as_mut_ptr()),
                n_blocks,
            )
        };
        black_box(out[0])
    });
}

// ---------- AOT: unfused pipeline (the realistic baseline) -------------------

#[inline(never)]
#[target_feature(enable = "avx2")]
unsafe fn aot_stage_ffor_add(
    packed: *const u8,
    ffor_ref: u8,
    scratch: *mut u8,
    n_blocks: usize,
) {
    let r = _mm256_set1_epi8(ffor_ref as i8);
    for i in 0..n_blocks {
        // SAFETY: caller-sized buffers.
        let data = unsafe { _mm256_loadu_si256(packed.add(i * 32) as *const __m256i) };
        let shifted = _mm256_add_epi8(data, r);
        unsafe { _mm256_storeu_si256(scratch.add(i * 32) as *mut __m256i, shifted) };
    }
}

#[inline(never)]
#[target_feature(enable = "avx2")]
unsafe fn aot_stage_compare_eq(
    data_in: *const u8,
    constant: u8,
    out: *mut u32,
    n_blocks: usize,
) {
    let c = _mm256_set1_epi8(constant as i8);
    for i in 0..n_blocks {
        // SAFETY: caller-sized buffers.
        let data = unsafe { _mm256_loadu_si256(data_in.add(i * 32) as *const __m256i) };
        let eq = _mm256_cmpeq_epi8(data, c);
        unsafe { *out.add(i) = _mm256_movemask_epi8(eq) as u32 };
    }
}

/// Vortex/FastLanes chunk size: 1024 elements per block. For `u8` data
/// that's 1024 bytes = 32 SIMD blocks of 32 bytes each. The scratch
/// buffer the unfused pipeline allocates is therefore *always* 1 KB and
/// always L1-resident, regardless of total dataset size.
const CHUNK_BLOCKS: usize = 32;
const CHUNK_BYTES: usize = CHUNK_BLOCKS * 32;

/// AOT, single-op kernels, **chunk-by-chunk** with a 1 KB L1-resident
/// scratch reused across chunks. This is the realistic baseline: chains
/// aren't known at compile time, so each op is its own AOT kernel; the
/// query engine drives the per-chunk loop.
#[divan::bench(args = SIZES)]
fn aot_chunked_unfused(bencher: Bencher, n_blocks: usize) {
    assert_eq!(n_blocks % CHUNK_BLOCKS, 0, "n_blocks must be a multiple of chunk size");
    let input = make_input(n_blocks);
    let mut scratch = vec![0u8; CHUNK_BYTES]; // 1 KB, hot in L1 across chunks
    let mut out = vec![0u32; n_blocks];
    let n_chunks = n_blocks / CHUNK_BLOCKS;
    bencher.counter(counter(n_blocks)).bench_local(|| {
        for chunk in 0..n_chunks {
            // SAFETY: input has n_blocks * 32 bytes, out has n_blocks * 4.
            unsafe {
                aot_stage_ffor_add(
                    input.as_ptr().add(chunk * CHUNK_BYTES),
                    black_box(7u8),
                    scratch.as_mut_ptr(),
                    CHUNK_BLOCKS,
                );
                aot_stage_compare_eq(
                    scratch.as_ptr(),
                    black_box(42u8),
                    out.as_mut_ptr().add(chunk * CHUNK_BLOCKS),
                    CHUNK_BLOCKS,
                );
            }
        }
        black_box(out[0])
    });
}

// ---------- Stencil-JIT bulk -------------------------------------------------

#[divan::bench(args = SIZES)]
fn stencil_jit_fused(bencher: Bencher, n_blocks: usize) {
    let kernel = BulkKernel::compile(ChainConfig::ffor_then_compare(CmpOp::Eq)).unwrap();
    let input = make_input(n_blocks);
    let mut out = vec![0u32; n_blocks];
    bencher.counter(counter(n_blocks)).bench_local(|| {
        // SAFETY: buffers sized for n_blocks * 32 / n_blocks * 4; n_blocks even.
        unsafe {
            kernel.call(
                black_box(input.as_ptr()),
                black_box(42u8),
                black_box(out.as_mut_ptr()),
                black_box(7u8),
                n_blocks,
            )
        };
        black_box(out[0])
    });
}

// ---------- Stencil-JIT specialized (constants baked, no FFoR slot, 4x unroll) ---
//
// The JIT-only win: at kernel-compile time we know both `ffor_ref` and
// `constant`, so `(x + r) == c` is algebraically folded into `x == (c - r)`.
// The loop body collapses to one memory-operand `vpcmpeqb` per block.

#[divan::bench(args = SIZES)]
fn stencil_jit_specialized(bencher: Bencher, n_blocks: usize) {
    let kernel = SpecializedKernel::compile_eq(black_box(42u8), black_box(7u8)).unwrap();
    let input = make_input(n_blocks);
    let mut out = vec![0u32; n_blocks];
    bencher.counter(counter(n_blocks)).bench_local(|| {
        // SAFETY: buffers sized for n_blocks * 32 / n_blocks * 4; n_blocks multiple of 4.
        unsafe {
            kernel.call(
                black_box(input.as_ptr()),
                black_box(out.as_mut_ptr()),
                n_blocks,
            )
        };
        black_box(out[0])
    });
}

// ---------- Stencil-JIT specialized AVX-512 (zmm, vpcmpeqb -> kmask, kmovq) ----
//
// AVX-512BW path: vpcmpeqb-on-zmm produces a 64-bit kmask directly, avoiding
// the AVX2 vpmovmskb port-0 bottleneck. kmovq writes 8 bytes of mask per
// 64-byte input block.

#[divan::bench(args = SIZES)]
fn stencil_jit_specialized_avx512(bencher: Bencher, n_blocks: usize) {
    if !std::is_x86_feature_detected!("avx512bw") {
        eprintln!("avx512bw not available; skipping");
        return;
    }
    let kernel = SpecializedKernel512::compile_eq(black_box(42u8), black_box(7u8)).unwrap();
    let input = make_input(n_blocks);
    let mut out = vec![0u32; n_blocks];
    bencher.counter(counter(n_blocks)).bench_local(|| {
        // SAFETY: n_blocks * 32 readable, n_blocks * 4 writable, multiple of 8.
        unsafe {
            kernel.call(
                black_box(input.as_ptr()),
                black_box(out.as_mut_ptr()),
                n_blocks,
            )
        };
        black_box(out[0])
    });
}

// ---------- F: Cranelift-at-build-time -----------------------------------------
//
// Bytes emitted by `build.rs` (cranelift-codegen 0.118 compiling IR for the
// eq kernel) and embedded via `include_bytes!`. The runtime path is the
// same `materialize()` D-spec uses; this entry only differs in *who wrote
// the bytes*. Note: Cranelift 0.118 doesn't yet support ymm/zmm in its
// x64 backend, so this kernel uses xmm (16-byte lanes). Newer Cranelift
// versions extend to AVX2/AVX-512, at which point F's throughput matches
// D-spec / D-spec-512 by construction.

#[divan::bench(args = SIZES)]
fn stencil_jit_cranelift_built(bencher: Bencher, n_blocks: usize) {
    let kernel = CraneliftKernel::new().unwrap();
    let input = make_input(n_blocks);
    let mut out = vec![0u32; n_blocks];
    bencher.counter(counter(n_blocks)).bench_local(|| {
        // SAFETY: n_blocks * 32 readable, n_blocks * 4 writable.
        unsafe {
            kernel.call(
                black_box(input.as_ptr()),
                black_box(out.as_mut_ptr()),
                n_blocks,
            )
        };
        black_box(out[0])
    });
}

// ---------- Stencil-JIT per-block (the original single-block kernel in a loop) ----

#[divan::bench(args = SIZES)]
fn stencil_jit_per_block(bencher: Bencher, n_blocks: usize) {
    let kernel = Kernel::compile(ChainConfig::ffor_then_compare(CmpOp::Eq)).unwrap();
    let input = make_input(n_blocks);
    let mut out = vec![0u32; n_blocks];
    bencher.counter(counter(n_blocks)).bench_local(|| {
        for i in 0..n_blocks {
            // SAFETY: 32-byte block + 4-byte out.
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

fn main() {
    divan::main();
}
