// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks GPU device memory bandwidth at different ratios of reads ("input")
//! vs writes ("output"), running both concurrently on separate CUDA streams.
//!
//! Cases: 100% input only, 100% output only, and 50/50 mixed.

#![expect(clippy::unwrap_used)]
#![expect(clippy::expect_used)]

mod bench_config;
// Unused here but suppresses dead_code warning for the shared module.
const _: &[(usize, &str)] = bench_config::BENCH_SIZES;

use std::time::Duration;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use mimalloc::MiMalloc;
use vortex_cuda::CudaExecutionCtx;
use vortex_cuda::CudaSession;
use vortex_cuda_driver::DevicePtr;
use vortex_cuda_driver::DevicePtrMut;
use vortex_cuda_driver::SyncOnDrop;
use vortex_cuda_driver::sys;
use vortex_cuda_driver::sys::CUevent_flags;
use vortex_cuda_macros::cuda_available;
use vortex_cuda_macros::cuda_not_available;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

/// Total data budget per benchmark iteration (100 MiB).
const TOTAL_BYTES: usize = 100 * 1024 * 1024;

/// Benchmark configurations: (input_bytes, output_bytes, label).
const MIXES: &[(usize, usize, &str)] = &[
    (TOTAL_BYTES, 0, "100%_in_0%_out"),
    (TOTAL_BYTES / 2, TOTAL_BYTES / 2, "50%_in_50%_out"),
    (0, TOTAL_BYTES, "0%_in_100%_out"),
];

fn transfer_mix_timed(
    input_bytes: usize,
    output_bytes: usize,
    input_ctx: &mut CudaExecutionCtx,
    output_ctx: &mut CudaExecutionCtx,
) -> Duration {
    let in_stream = input_ctx.stream().clone();
    let out_stream = output_ctx.stream().clone();
    let cu_in = in_stream.cu_stream();
    let cu_out = out_stream.cu_stream();

    let dtod_src = input_ctx.device_alloc::<u8>(input_bytes.max(1)).unwrap();
    let mut dtod_dst = input_ctx.device_alloc::<u8>(input_bytes.max(1)).unwrap();
    let mut memset_dst = output_ctx
        .device_alloc::<u32>((output_bytes / size_of::<u32>()).max(1))
        .unwrap();

    let (src_ptr, record_src) = dtod_src.device_ptr(&in_stream).expect("access copy source");
    let (dst_ptr, record_dst) = dtod_dst
        .device_ptr_mut(&in_stream)
        .expect("access copy destination");
    let (memset_ptr, record_memset) = memset_dst
        .device_ptr_mut(&out_stream)
        .expect("access memset destination");

    in_stream.synchronize().unwrap();
    out_stream.synchronize().unwrap();

    let start = in_stream
        .record_event(Some(CUevent_flags::CU_EVENT_BLOCKING_SYNC))
        .unwrap();
    out_stream.wait(&start).unwrap();

    let copy = if input_bytes > 0 {
        unsafe { sys::cuMemcpyDtoDAsync_v2(dst_ptr, src_ptr, input_bytes, cu_in) }.result()
    } else {
        Ok(())
    };
    let memset = if output_bytes > 0 {
        let output_u32s = output_bytes / size_of::<u32>();
        unsafe { sys::cuMemsetD32Async(memset_ptr, 0xA5A5_A5A5_u32, output_u32s, cu_out) }.result()
    } else {
        Ok(())
    };
    let completion = SyncOnDrop::finish_all([record_src, record_dst, record_memset]);
    copy.inspect_err(|_| {
        if let Err(error) = &memset {
            tracing::error!(%error, "Failed to enqueue device memset after copy failure");
        }
        if let Err(error) = &completion {
            tracing::error!(%error, "Failed to track transfer buffer access after copy failure");
        }
    })
    .expect("enqueue device copy");
    memset
        .inspect_err(|_| {
            if let Err(error) = &completion {
                tracing::error!(%error, "Failed to track transfer buffer access after memset failure");
            }
        })
        .expect("enqueue device memset");
    completion.expect("track transfer buffer access");

    let end_in = in_stream
        .record_event(Some(CUevent_flags::CU_EVENT_BLOCKING_SYNC))
        .unwrap();
    let end_out = out_stream
        .record_event(Some(CUevent_flags::CU_EVENT_BLOCKING_SYNC))
        .unwrap();

    let elapsed_ms = f32::max(
        start.elapsed_ms(&end_in).unwrap(),
        start.elapsed_ms(&end_out).unwrap(),
    );

    Duration::from_secs_f32(elapsed_ms / 1000.0)
}

fn benchmark_transfer_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("cuda");
    for &(input_bytes, output_bytes, label) in MIXES {
        let total_mem_bytes = input_bytes * 2 + output_bytes;
        group.throughput(Throughput::Bytes(total_mem_bytes as u64));

        group.bench_with_input(
            BenchmarkId::new(format!("cuda/throughput/transfer/{label}"), "100MiB"),
            &(input_bytes, output_bytes),
            |b, &(in_bytes, out_bytes)| {
                b.iter_custom(|iters| {
                    let session = vortex_cuda::cuda_session();
                    let mut in_ctx = CudaSession::create_execution_ctx(&session).unwrap();
                    let mut out_ctx = CudaSession::create_execution_ctx(&session).unwrap();

                    let mut total = Duration::ZERO;
                    for _ in 0..iters {
                        total += transfer_mix_timed(in_bytes, out_bytes, &mut in_ctx, &mut out_ctx);
                    }
                    total
                });
            },
        );
    }

    group.finish();
}

criterion::criterion_group! {
    name = benches;
    config = bench_config::cuda_bench_config();
    targets = benchmark_transfer_throughput
}

#[cuda_available]
criterion::criterion_main!(benches);

#[cuda_not_available]
fn main() {}
