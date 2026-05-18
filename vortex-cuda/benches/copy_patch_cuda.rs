// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA benchmarks for the Copy-and-Patch prototype.
//!
//! The interesting numbers here are:
//!
//! * `cuda/copy_patch/link_cold` — wall-clock latency of `cuLinkCreate` +
//!   `cuLinkAddData` (one trampoline + three stencils) + `cuLinkComplete` +
//!   `cuModuleLoadData`. This is the cost Copy-and-Patch is supposed to keep
//!   small relative to running NVRTC/`nvcc` at runtime.
//!
//! * `cuda/copy_patch/launch_warm` — kernel-only time after the cache is
//!   primed, measured with CUDA events. This is the steady-state cost for
//!   queries that hit the link cache.
//!
//! * `cuda/copy_patch/end_to_end_cold` — fresh executor per iteration plus
//!   the launch, so the link and launch are both timed together. Useful for
//!   comparing against an alternative path that has zero per-query setup.
//!
//! These are gated to run only when CUDA is available, the same as every
//! other bench in this crate.

#![cfg(feature = "copy_patch_demo")]
#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation)]

// Shared helpers: `bench_config::cuda_bench_config` for the Criterion
// configuration, and `TimedLaunchStrategy` to measure kernel-only time.
// `BENCH_SIZES` in `bench_config` is intentionally unused here — we fix
// `BENCH_LEN` below to keep the link / launch comparisons apples-to-apples.
#[allow(dead_code)]
mod bench_config;
mod timed_launch_strategy;

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;
use std::time::Instant;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use cudarc::driver::CudaSlice;
use futures::executor::block_on;
use vortex::array::IntoArray;
use vortex::array::LEGACY_SESSION;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::validity::Validity::NonNullable;
use vortex::buffer::Buffer;
use vortex::encodings::fastlanes::BitPacked;
use vortex::encodings::fastlanes::BitPackedArrayExt;
use vortex::session::VortexSession;
use vortex_cuda::CudaBufferExt;
use vortex_cuda::CudaSession;
use vortex_cuda::copy_patch::ArithOp;
use vortex_cuda::copy_patch::CopyPatchExecutor;
use vortex_cuda::copy_patch::FilterOp;
use vortex_cuda::copy_patch::Plan;
use vortex_cuda::copy_patch::PostOp;
use vortex_cuda::executor::CudaExecutionCtx;
use vortex_cuda_macros::cuda_available;
use vortex_cuda_macros::cuda_not_available;

use crate::timed_launch_strategy::TimedLaunchStrategy;

/// Number of elements per benchmark input. Chosen to stay well above
/// per-launch overhead while keeping the link micro-benchmarks short.
const BENCH_LEN: usize = 1 << 20;
/// Bit width to pack the encoded values with. Picking a non-aligned width
/// exercises a non-trivial stencil rather than a degenerate copy.
const BENCH_BW: u8 = 10;

/// Build a fresh device-resident `u32` packed buffer of `len` elements.
fn build_packed_input(ctx: &mut CudaExecutionCtx, len: usize) -> CudaSlice<u32> {
    let encoded: Buffer<i32> = (0..len as i32).map(|i| i & ((1 << BENCH_BW) - 1)).collect();
    let prim = PrimitiveArray::new(encoded, NonNullable);
    let bp = BitPacked::encode(
        &prim.into_array(),
        BENCH_BW,
        &mut LEGACY_SESSION.create_execution_ctx(),
    )
    .unwrap();

    let packed_handle = block_on(ctx.ensure_on_device(bp.packed().clone())).unwrap();
    let packed_words = packed_handle.cuda_view::<u32>().unwrap();
    let n = packed_words.len();
    let mut owned = ctx.device_alloc::<u32>(n).unwrap();
    ctx.stream().memcpy_dtod(&packed_words, &mut owned).unwrap();
    owned
}

fn arith_plan() -> Plan {
    Plan {
        bit_width: BENCH_BW,
        f: 100.0,
        e: 1.0,
        post: PostOp::Arith {
            op: ArithOp::Mul,
            c: 2.0,
        },
    }
}

fn filter_plan() -> Plan {
    Plan {
        bit_width: BENCH_BW,
        f: 100.0,
        e: 1.0,
        post: PostOp::Filter {
            op: FilterOp::Gt,
            c: 50_000.0,
        },
    }
}

/// Time only the link step: every iteration creates a fresh executor (so
/// nothing is cached) and calls `warm_up` to drive `cuLink*` +
/// `cuModuleLoadData`. Measured with wall-clock time because the link
/// happens entirely on the host.
fn bench_link_cold(c: &mut Criterion) {
    let mut group = c.benchmark_group("cuda");
    group.bench_function(BenchmarkId::new("copy_patch/link_cold", "arith_mul"), |b| {
        b.iter_custom(|iters| {
            let cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty()).unwrap();
            let plan = arith_plan();
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                // Fresh executor per iteration so the cache misses.
                let executor = CopyPatchExecutor::new();
                let start = Instant::now();
                executor.warm_up(&cuda_ctx, &plan).unwrap();
                total += start.elapsed();
            }
            total
        });
    });
    group.bench_function(BenchmarkId::new("copy_patch/link_cold", "filter_gt"), |b| {
        b.iter_custom(|iters| {
            let cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty()).unwrap();
            let plan = filter_plan();
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let executor = CopyPatchExecutor::new();
                let start = Instant::now();
                executor.warm_up(&cuda_ctx, &plan).unwrap();
                total += start.elapsed();
            }
            total
        });
    });
    group.finish();
}

/// Kernel-only time after a warm cache. Uses the existing
/// `TimedLaunchStrategy` so the measurement is the CUDA-event duration
/// between the launch's surrounding events, not wall clock.
fn bench_launch_warm(c: &mut Criterion) {
    let mut group = c.benchmark_group("cuda");
    group.throughput(Throughput::Elements(BENCH_LEN as u64));

    for (name, plan) in [("arith_mul", arith_plan()), ("filter_gt", filter_plan())] {
        group.bench_function(BenchmarkId::new("copy_patch/launch_warm", name), |b| {
            b.iter_custom(|iters| {
                let timed = TimedLaunchStrategy::default();
                let timer = timed.timer();
                let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
                    .unwrap()
                    .with_launch_strategy(Arc::new(timed));
                let packed = build_packed_input(&mut cuda_ctx, BENCH_LEN);
                let executor = CopyPatchExecutor::new();
                executor.warm_up(&cuda_ctx, &plan).unwrap();

                for _ in 0..iters {
                    drop(
                        executor
                            .launch(&mut cuda_ctx, &plan, packed.as_view(), BENCH_LEN)
                            .unwrap(),
                    );
                }
                Duration::from_nanos(timer.load(Ordering::Relaxed))
            });
        });
    }
    group.finish();
}

/// Cold path: fresh executor + first launch, both timed (wall clock).
/// Represents the latency a first-time query pays.
fn bench_end_to_end_cold(c: &mut Criterion) {
    let mut group = c.benchmark_group("cuda");
    group.throughput(Throughput::Elements(BENCH_LEN as u64));

    for (name, plan) in [("arith_mul", arith_plan()), ("filter_gt", filter_plan())] {
        group.bench_function(BenchmarkId::new("copy_patch/end_to_end_cold", name), |b| {
            b.iter_custom(|iters| {
                let mut cuda_ctx =
                    CudaSession::create_execution_ctx(&VortexSession::empty()).unwrap();
                let packed = build_packed_input(&mut cuda_ctx, BENCH_LEN);

                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let executor = CopyPatchExecutor::new();
                    let start = Instant::now();
                    drop(
                        executor
                            .launch(&mut cuda_ctx, &plan, packed.as_view(), BENCH_LEN)
                            .unwrap(),
                    );
                    cuda_ctx.stream().synchronize().unwrap();
                    total += start.elapsed();
                }
                total
            });
        });
    }
    group.finish();
}

criterion::criterion_group! {
    name = benches;
    config = bench_config::cuda_bench_config();
    targets = bench_link_cold, bench_launch_warm, bench_end_to_end_cold
}

#[cuda_available]
criterion::criterion_main!(benches);

#[cuda_not_available]
fn main() {}
