// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used)]
#![allow(clippy::cast_possible_truncation)]

use std::time::Duration;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use cudarc::driver::DevicePtrMut;
use cudarc::driver::sys::CUevent_flags;
use futures::executor::block_on;
use vortex::array::arrays::VarBinViewArray;
use vortex::encodings::zstd::Zstd;
use vortex::encodings::zstd::ZstdArray;
use vortex::encodings::zstd::ZstdArrayParts;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::session::VortexSession;
use vortex_cuda::CudaSession;
use vortex_cuda::ZstdKernelPrep;
use vortex_cuda::nvcomp::zstd as nvcomp_zstd;
use vortex_cuda::zstd_kernel_prepare;
use vortex_cuda_macros::cuda_available;
use vortex_cuda_macros::cuda_not_available;

const BENCH_ARGS: &[(usize, &str)] = &[
    (1_000_000, "1M"),
    (10_000_000, "10M"),
    (100_000_000, "100M"),
];

/// Generate compressible string data by repeating patterns.
fn generate_string_data(count: usize) -> Vec<&'static str> {
    let patterns = &[
        "the quick brown fox jumps over the lazy dog",
        "hello world from vortex compression benchmark",
        "lorem ipsum dolor sit amet consectetur adipiscing",
        "testing cuda zstd decompression kernel performance",
        "this is a longer test string with more characters here",
        "short",
        "another string for benchmarking purposes and timing",
        "data compression is important for system performance",
        "cuda acceleration enables faster data processing",
        "vortex provides efficient data handling capabilities",
    ];

    (0..count).map(|i| patterns[i % patterns.len()]).collect()
}

/// Create a ZSTD-compressed array
fn make_zstd_array(num_strings: usize) -> VortexResult<(ZstdArray, usize)> {
    let strings = generate_string_data(num_strings);
    let var_bin_view = VarBinViewArray::from_iter_str(strings.iter().copied());
    let uncompressed_size: usize = strings.iter().map(|s| s.len()).sum();
    let zstd_compression_level = -10; // Less compression but faster.
    let zstd_array =
        // Disable dictionary as nvCOMP doesn't support ZSTD dictionaries.
        Zstd::from_var_bin_view_without_dict(&var_bin_view, zstd_compression_level, 2048)?;

    Ok((zstd_array, uncompressed_size))
}

/// Executes the ZSTD kernel and measures execution time using CUDA events.
async fn execute_zstd_kernel(
    mut exec: ZstdKernelPrep,
    cuda_ctx: &mut vortex_cuda::CudaExecutionCtx,
) -> VortexResult<Duration> {
    let stream = cuda_ctx.stream();
    let ctx = stream.context();

    let start_event = ctx
        .new_event(Some(CUevent_flags::CU_EVENT_BLOCKING_SYNC))
        .map_err(|e| vortex_err!("Failed to create start event: {:?}", e))?;
    start_event
        .record(stream)
        .map_err(|e| vortex_err!("Failed to record start event: {:?}", e))?;

    let (device_actual_sizes_ptr, record_actual_sizes) =
        exec.device_actual_sizes.device_ptr_mut(stream);
    let (nvcomp_temp_buffer_ptr, record_temp) = exec.nvcomp_temp_buffer.device_ptr_mut(stream);
    let (device_statuses_ptr, record_statuses) = exec.device_statuses.device_ptr_mut(stream);

    // Launch the kernel
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
        .map_err(|e| vortex_err!("Failed to create end event: {:?}", e))?;

    end_event
        .record(stream)
        .map_err(|e| vortex_err!("Failed to record end event: {:?}", e))?;

    let elapsed_ms = start_event
        .elapsed_ms(&end_event)
        .map_err(|e| vortex_err!("Failed to get elapsed time: {:?}", e))?;

    Ok(Duration::from_secs_f32(elapsed_ms / 1000.0))
}

/// Benchmark ZSTD CUDA decompression kernel performance
fn benchmark_zstd_cuda_decompress(c: &mut Criterion) {
    let mut group = c.benchmark_group("ZSTD_cuda");
    group.sample_size(10);

    for (num_strings, label) in BENCH_ARGS {
        let (zstd_array, uncompressed_size) =
            make_zstd_array(*num_strings).vortex_expect("failed to create ZSTD array");

        group.throughput(Throughput::Bytes(uncompressed_size as u64));
        group.bench_with_input(
            BenchmarkId::new("decompress_kernel", label),
            &zstd_array,
            |b, zstd_array| {
                b.iter_custom(|iters| {
                    let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
                        .vortex_expect("failed to create execution context");

                    let mut total_time = Duration::ZERO;

                    for _ in 0..iters {
                        let ZstdArrayParts {
                            frames, metadata, ..
                        } = zstd_array.clone().into_data().into_parts();
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

criterion::criterion_group!(benches, benchmark_zstd_cuda_decompress);

#[cuda_available]
criterion::criterion_main!(benches);

#[cuda_not_available]
fn main() {}
