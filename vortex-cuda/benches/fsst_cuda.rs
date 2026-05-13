// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA decompression benchmarks on the same ClickBench URL corpus.
//!
//! The FSST benchmark `cuda/fsst/decompress` keeps the same criterion id as
//! the CPU baseline committed earlier; this lets us swap the body between
//! implementations and compare with `--save-baseline`. This commit also
//! registers a `cuda/zstd/decompress` measurement inside the same bench
//! function so a single criterion run produces side-by-side FSST and ZSTD
//! numbers on the same corpus — comparing a baseline without zstd to one
//! with zstd just means re-running the bench at this commit.

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation)]

#[allow(dead_code)]
mod bench_config;
mod timed_launch_strategy;

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use cudarc::driver::DevicePtrMut;
use cudarc::driver::sys::CUevent_flags;
use futures::executor::block_on;
use vortex::array::IntoArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::VarBinViewArray;
use vortex::array::match_each_integer_ptype;
use vortex::array::vtable::child_to_validity;
use vortex::encodings::fsst::FSSTArrayExt;
use vortex::encodings::zstd::Zstd;
use vortex::encodings::zstd::ZstdArray;
use vortex::encodings::zstd::ZstdDataParts;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::session::VortexSession;
use vortex_cuda::CudaExecutionCtx;
use vortex_cuda::CudaSession;
use vortex_cuda::ZstdKernelPrep;
use vortex_cuda::executor::CudaArrayExt;
use vortex_cuda::nvcomp::zstd as nvcomp_zstd;
use vortex_cuda::zstd_kernel_prepare;
use vortex_cuda_macros::cuda_available;
use vortex_cuda_macros::cuda_not_available;
use vortex_fsst::test_utils::generate_clickbench_urls;
use vortex_fsst::test_utils::make_fsst_clickbench_urls;

use crate::timed_launch_strategy::TimedLaunchStrategy;

// Bench-local size instead of the workspace 100M default: each input is a
// clickbench URL, much heavier per-element than the fixed-width primitives
// other kernels benchmark.
const BENCH_SIZES: &[(usize, &str)] = &[(10_000_000, "10M")];

/// Compress the same ClickBench URL corpus the FSST bench uses with ZSTD,
/// so the two cases measure decompression of the same logical data.
fn make_zstd_clickbench_urls(
    n: usize,
    cuda_ctx: &mut CudaExecutionCtx,
) -> VortexResult<(ZstdArray, usize)> {
    let urls = generate_clickbench_urls(n);
    let var_bin_view = VarBinViewArray::from_iter_str(urls.iter().map(|s| s.as_str()));
    let uncompressed_size: usize = urls.iter().map(|s| s.len()).sum();
    // Match the existing ZSTD CUDA bench in this crate: level -10 (less
    // compression, faster), 2 KiB dictionary, nvCOMP-compatible (no dict).
    let zstd_compression_level = -10;
    let zstd_array = Zstd::from_var_bin_view_without_dict(
        &var_bin_view,
        zstd_compression_level,
        2048,
        cuda_ctx.execution_ctx(),
    )?;
    Ok((zstd_array, uncompressed_size))
}

/// Executes the ZSTD nvCOMP kernel and reports kernel-only time via CUDA events.
async fn execute_zstd_kernel(
    mut exec: ZstdKernelPrep,
    cuda_ctx: &mut CudaExecutionCtx,
) -> VortexResult<Duration> {
    let stream = cuda_ctx.stream();
    let ctx = stream.context();

    let start_event = ctx
        .new_event(Some(CUevent_flags::CU_EVENT_BLOCKING_SYNC))
        .map_err(|e| vortex_err!("failed to create start event: {:?}", e))?;
    start_event
        .record(stream)
        .map_err(|e| vortex_err!("failed to record start event: {:?}", e))?;

    let (device_actual_sizes_ptr, record_actual_sizes) =
        exec.device_actual_sizes.device_ptr_mut(stream);
    let (nvcomp_temp_buffer_ptr, record_temp) = exec.nvcomp_temp_buffer.device_ptr_mut(stream);
    let (device_statuses_ptr, record_statuses) = exec.device_statuses.device_ptr_mut(stream);

    unsafe {
        nvcomp_zstd::decompress_async(
            exec.frame_ptrs_ptr as _,
            exec.frame_sizes_ptr as _,
            exec.output_sizes_ptr as _,
            device_actual_sizes_ptr as _,
            exec.num_frames,
            nvcomp_temp_buffer_ptr as _,
            exec.nvcomp_temp_buffer_size,
            exec.output_ptrs_ptr as _,
            device_statuses_ptr as _,
            stream.cu_stream().cast(),
        )
        .map_err(|e| vortex_err!("nvcomp decompress_async failed: {}", e))?;
    }
    drop((record_actual_sizes, record_temp, record_statuses));

    let end_event = ctx
        .new_event(Some(CUevent_flags::CU_EVENT_BLOCKING_SYNC))
        .map_err(|e| vortex_err!("failed to create end event: {:?}", e))?;
    end_event
        .record(stream)
        .map_err(|e| vortex_err!("failed to record end event: {:?}", e))?;

    let elapsed_ms = start_event
        .elapsed_ms(&end_event)
        .map_err(|e| vortex_err!("failed to get elapsed time: {:?}", e))?;
    Ok(Duration::from_secs_f32(elapsed_ms / 1000.0))
}

fn benchmark_fsst_cuda_decompress(c: &mut Criterion) {
    let mut group = c.benchmark_group("cuda");

    for &(n, len_str) in BENCH_SIZES {
        let mut setup_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        // ===== FSST measurement on the corpus =====
        let fsst = make_fsst_clickbench_urls(n, setup_ctx.execution_ctx());

        let lens = fsst
            .uncompressed_lengths()
            .clone()
            .execute::<PrimitiveArray>(setup_ctx.execution_ctx())
            .vortex_expect("canonicalize uncompressed_lengths");
        let total_size: usize = match_each_integer_ptype!(lens.ptype(), |P| {
            lens.as_slice::<P>().iter().map(|x| *x as usize).sum()
        });
        let uncompressed_size = total_size as u64;

        let fsst_array = fsst.into_array();

        group.throughput(Throughput::Bytes(uncompressed_size));
        group.bench_with_input(
            BenchmarkId::new("cuda/fsst/decompress", len_str),
            &fsst_array,
            |b, fsst_array| {
                b.iter_custom(|iters| {
                    let timed = TimedLaunchStrategy::default();
                    let timer = timed.timer();

                    let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
                        .vortex_expect("failed to create execution context")
                        .with_launch_strategy(Arc::new(timed));

                    for _ in 0..iters {
                        block_on(fsst_array.clone().execute_cuda(&mut cuda_ctx)).unwrap();
                    }
                    Duration::from_nanos(timer.load(Ordering::Relaxed))
                });
            },
        );

        // ===== ZSTD comparison on the same corpus =====
        let (zstd_array, zstd_uncompressed_size) = make_zstd_clickbench_urls(n, &mut setup_ctx)
            .vortex_expect("failed to create ZSTD array");

        group.throughput(Throughput::Bytes(zstd_uncompressed_size as u64));
        group.bench_with_input(
            BenchmarkId::new("cuda/zstd/decompress", len_str),
            &zstd_array,
            |b, zstd_array| {
                b.iter_custom(|iters| {
                    let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
                        .vortex_expect("failed to create cuda execution context");
                    let mut total_time = Duration::ZERO;
                    for _ in 0..iters {
                        let ZstdDataParts {
                            frames, metadata, ..
                        } = {
                            let validity = child_to_validity(
                                zstd_array.as_ref().slots()[0].as_ref(),
                                zstd_array.dtype().nullability(),
                            );
                            zstd_array.clone().into_data().into_parts(validity)
                        };
                        let exec = block_on(zstd_kernel_prepare(frames, &metadata, &mut cuda_ctx))
                            .vortex_expect("kernel setup failed");
                        let kernel_time = block_on(execute_zstd_kernel(exec, &mut cuda_ctx))
                            .vortex_expect("kernel execution failed");
                        total_time += kernel_time;
                    }
                    total_time
                });
            },
        );
    }

    group.finish();
}

criterion::criterion_group! {
    name = benches;
    config = bench_config::cuda_bench_config();
    targets = benchmark_fsst_cuda_decompress
}

#[cuda_available]
criterion::criterion_main!(benches);

#[cuda_not_available]
fn main() {}
