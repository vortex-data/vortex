// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]

use std::time::Duration;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use cudarc::driver::PushKernelArg;
use cudarc::driver::sys::CUevent_flags;
use futures::executor::block_on;
use mimalloc::MiMalloc;
use vortex::array::LEGACY_SESSION;
use vortex::array::VortexSessionExecute;
use vortex::array::aggregate_fn::fns::sum::sum;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::session::VortexSession;
use vortex_cuda::CudaBufferExt;
use vortex_cuda::CudaExecutionCtx;
use vortex_cuda::CudaSession;
use vortex_cuda::FsstKernelPrep;
use vortex_cuda::fsst_kernel_prepare;
use vortex_cuda_macros::cuda_available;
use vortex_cuda_macros::cuda_not_available;
use vortex_fsst::FSSTArrayExt;
use vortex_fsst::test_utils::make_fsst_clickbench_urls;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

const BENCH_ARGS: &[(usize, &str)] = &[(1_000_000, "1M"), (5_000_000, "5M"), (10_000_000, "10M")];

async fn execute_fsst_kernel(
    prep: &FsstKernelPrep,
    cuda_ctx: &mut CudaExecutionCtx,
) -> VortexResult<Duration> {
    let codes_bytes_view = prep.codes_bytes.cuda_view::<u8>()?;
    let codes_offsets_view = prep.codes_offsets.cuda_view::<i32>()?;
    let symbols_view = prep.symbols.cuda_view::<u64>()?;
    let symbol_lengths_view = prep.symbol_lengths.cuda_view::<u8>()?;
    let output_offsets_view = prep.output_offsets.cuda_view::<i32>()?;

    let cuda_function = cuda_ctx.load_function("fsst_decompress", &[])?;
    let num_strings_u64 = prep.num_strings as u64;

    let stream = cuda_ctx.stream();
    let ctx = stream.context();

    let start_event = ctx
        .new_event(Some(CUevent_flags::CU_EVENT_BLOCKING_SYNC))
        .map_err(|e| vortex_err!("failed to create start event: {:?}", e))?;
    start_event
        .record(stream)
        .map_err(|e| vortex_err!("failed to record start event: {:?}", e))?;

    cuda_ctx.launch_kernel(&cuda_function, prep.num_strings, |args| {
        args.arg(&codes_bytes_view)
            .arg(&codes_offsets_view)
            .arg(&symbols_view)
            .arg(&symbol_lengths_view)
            .arg(&output_offsets_view)
            .arg(&prep.device_output)
            .arg(&num_strings_u64);
    })?;

    let stream = cuda_ctx.stream();
    let ctx = stream.context();
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
    let mut group = c.benchmark_group("FSST_cuda");

    for (n, label) in BENCH_ARGS {
        let mut setup_ctx = LEGACY_SESSION.create_execution_ctx();
        let array = make_fsst_clickbench_urls(*n, &mut setup_ctx);
        let uncompressed_bytes = sum(array.uncompressed_lengths(), &mut setup_ctx)
            .unwrap()
            .as_primitive()
            .typed_value::<i64>()
            .unwrap();

        group.throughput(Throughput::Bytes(uncompressed_bytes as u64));
        group.bench_with_input(
            BenchmarkId::new("decompress_kernel", label),
            &array,
            |b, array| {
                b.iter_custom(|iters| {
                    let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
                        .vortex_expect("failed to create cuda execution context");
                    let mut total_time = Duration::ZERO;
                    for _ in 0..iters {
                        let prep = block_on(fsst_kernel_prepare(array.clone(), &mut cuda_ctx))
                            .vortex_expect("kernel setup failed");
                        let kernel_time = block_on(execute_fsst_kernel(&prep, &mut cuda_ctx))
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
    targets = benchmark_fsst_cuda_decompress
}

#[cuda_available]
criterion::criterion_main!(benches);

#[cuda_not_available]
fn main() {}
