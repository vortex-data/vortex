// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::time::Duration;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use cudarc::driver::DevicePtrMut;
use cudarc::driver::sys::CUevent_flags;
use futures::executor::block_on;
use mimalloc::MiMalloc;
use vortex::array::arrays::VarBinViewArray;
use vortex::array::vtable::child_to_validity;
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
use vortex_cuda::nvcomp::zstd as nvcomp_zstd;
use vortex_cuda::zstd_kernel_prepare;
use vortex_cuda_macros::cuda_available;
use vortex_cuda_macros::cuda_not_available;
use vortex_fsst::test_utils::generate_clickbench_urls;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

const BENCH_ARGS: &[(usize, &str)] = &[(1_000_000, "1M"), (5_000_000, "5M"), (10_000_000, "10M")];

fn make_zstd_clickbench_urls(
    n: usize,
    cuda_ctx: &mut CudaExecutionCtx,
) -> VortexResult<(ZstdArray, usize)> {
    let urls = generate_clickbench_urls(n);
    let var_bin_view = VarBinViewArray::from_iter_str(urls.iter().map(|s| s.as_str()));
    let uncompressed_size: usize = urls.iter().map(|s| s.len()).sum();
    let zstd_compression_level = -10;
    let zstd_array = Zstd::from_var_bin_view_without_dict(
        &var_bin_view,
        zstd_compression_level,
        2048,
        cuda_ctx.execution_ctx(),
    )?;
    Ok((zstd_array, uncompressed_size))
}

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

fn benchmark_zstd_cuda_decompress(c: &mut Criterion) {
    let mut group = c.benchmark_group("ZSTD_cuda");

    for (n, label) in BENCH_ARGS {
        let mut setup_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create cuda execution context");
        let (zstd_array, uncompressed_size) = make_zstd_clickbench_urls(*n, &mut setup_ctx)
            .vortex_expect("failed to create ZSTD array");

        group.throughput(Throughput::Bytes(uncompressed_size as u64));
        group.bench_with_input(
            BenchmarkId::new("decompress_kernel", label),
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
                                &zstd_array.as_ref().slots()[0],
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
    config = Criterion::default().without_plots()
        .sample_size(10)
        .warm_up_time(Duration::from_nanos(1))
        .measurement_time(Duration::from_nanos(1))
        .nresamples(10);
    targets = benchmark_zstd_cuda_decompress
}

#[cuda_available]
criterion::criterion_main!(benches);

#[cuda_not_available]
fn main() {}
