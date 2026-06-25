// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#![expect(clippy::cast_possible_truncation)]

//! CUDA benchmarks for Arrow validity bitmap repacking.

mod bench_config;
mod timed_launch_strategy;

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;
use std::time::Instant;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use futures::executor::block_on;
use vortex::array::IntoArray;
use vortex::array::arrays::BoolArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::buffer::BufferHandle;
use vortex::array::validity::Validity;
use vortex::buffer::BitBuffer;
use vortex::buffer::Buffer;
use vortex::dtype::PType;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex_cuda::CudaExecutionCtx;
use vortex_cuda::CudaSession;
use vortex_cuda::arrow::ArrowDeviceArray;
use vortex_cuda::arrow::DeviceArrayExt;
use vortex_cuda::arrow::test_harness;
use vortex_cuda_macros::cuda_available;
use vortex_cuda_macros::cuda_not_available;

use crate::timed_launch_strategy::TimedLaunchStrategy;

const INPUT_OFFSET: usize = 5;
const ARROW_OFFSET: usize = 3;
const EXPORT_BENCH_SIZES: &[(usize, &str)] = &[(100_000_000, "100M")];

fn validity_bitmap_byte_len(len: usize, bit_offset: usize) -> usize {
    (bit_offset + len).div_ceil(8)
}

fn create_cuda_execution_ctx() -> CudaExecutionCtx {
    CudaSession::create_execution_ctx(&vortex_cuda::cuda_session())
        .vortex_expect("failed to create execution context")
}

unsafe fn release_arrow_device_array(array: &mut ArrowDeviceArray) {
    unsafe {
        if let Some(release) = array.array.release {
            release(&raw mut array.array);
        }
    }
}

async fn device_validity_buffer(
    len: usize,
    validity_offset: usize,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<(usize, BufferHandle)> {
    let validity_bits = BitBuffer::collect_bool(len + validity_offset, |idx| idx % 3 != 0)
        .slice(validity_offset..validity_offset + len);
    let (validity_offset, _, validity_buffer) = validity_bits.into_inner();
    Ok((
        validity_offset,
        ctx.ensure_on_device(BufferHandle::new_host(validity_buffer))
            .await?,
    ))
}

async fn primitive_with_device_bool_validity(
    len: usize,
    validity_offset: usize,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<vortex::array::ArrayRef> {
    let values = Buffer::<i32>::from_iter((0..len).map(|idx| idx as i32));
    let values = ctx
        .ensure_on_device(BufferHandle::new_host(values.into_byte_buffer()))
        .await?;

    let (validity_offset, validity_buffer) =
        device_validity_buffer(len, validity_offset, ctx).await?;
    let validity =
        BoolArray::new_handle(validity_buffer, validity_offset, len, Validity::NonNullable)
            .into_array();

    Ok(
        PrimitiveArray::from_buffer_handle(values, PType::I32, Validity::Array(validity))
            .into_array(),
    )
}

fn benchmark_arrow_validity_export(c: &mut Criterion) {
    let mut group = c.benchmark_group("cuda");

    for &(len, len_label) in EXPORT_BENCH_SIZES {
        for (case, validity_offset) in
            [("device_bitmap", 0), ("device_bitmap_repack", INPUT_OFFSET)]
        {
            group.throughput(Throughput::Bytes(
                validity_bitmap_byte_len(len, validity_offset) as u64,
            ));
            group.bench_with_input(
                BenchmarkId::new(format!("cuda/arrow_validity/export/{case}"), len_label),
                &len,
                |b, &len| {
                    b.iter_custom(|iters| {
                        let mut cuda_ctx = create_cuda_execution_ctx();
                        let array = block_on(primitive_with_device_bool_validity(
                            len,
                            validity_offset,
                            &mut cuda_ctx,
                        ))
                        .vortex_expect("failed to create primitive fixture");

                        let mut exported_arrays = Vec::with_capacity(
                            usize::try_from(iters)
                                .vortex_expect("iteration count does not fit usize"),
                        );

                        let start = Instant::now();
                        for _ in 0..iters {
                            exported_arrays.push(
                                block_on(array.clone().export_device_array(&mut cuda_ctx))
                                    .vortex_expect("failed to export device array"),
                            );
                        }
                        let elapsed = start.elapsed();

                        for exported in &mut exported_arrays {
                            unsafe { release_arrow_device_array(exported) };
                        }

                        elapsed
                    });
                },
            );
        }
    }

    group.finish();
}

fn benchmark_arrow_validity_repack(c: &mut Criterion) {
    let mut group = c.benchmark_group("cuda");

    for &(len, len_label) in bench_config::BENCH_SIZES {
        group.throughput(Throughput::Bytes(
            validity_bitmap_byte_len(len, INPUT_OFFSET) as u64,
        ));
        group.bench_with_input(
            BenchmarkId::new("cuda/arrow_validity/repack", len_label),
            &len,
            |b, &len| {
                b.iter_custom(|iters| {
                    let timed = TimedLaunchStrategy::default();
                    let timer = timed.timer();
                    let mut cuda_ctx =
                        create_cuda_execution_ctx().with_launch_strategy(Arc::new(timed));
                    let (input_offset, input_buffer) =
                        block_on(device_validity_buffer(len, INPUT_OFFSET, &mut cuda_ctx))
                            .vortex_expect("failed to create validity fixture");

                    for _ in 0..iters {
                        let output = test_harness::repack_arrow_validity_buffer(
                            &input_buffer,
                            input_offset,
                            len,
                            ARROW_OFFSET,
                            &mut cuda_ctx,
                        )
                        .vortex_expect("failed to repack Arrow validity");
                        std::hint::black_box(output);
                    }

                    Duration::from_nanos(timer.load(Ordering::Relaxed))
                });
            },
        );
    }

    group.finish();
}

fn benchmark_arrow_validity_count_nulls(c: &mut Criterion) {
    let mut group = c.benchmark_group("cuda");

    for &(len, len_label) in bench_config::BENCH_SIZES {
        group.throughput(Throughput::Bytes(
            validity_bitmap_byte_len(len, ARROW_OFFSET) as u64,
        ));
        group.bench_with_input(
            BenchmarkId::new("cuda/arrow_validity/count_nulls", len_label),
            &len,
            |b, &len| {
                b.iter_custom(|iters| {
                    let timed = TimedLaunchStrategy::default();
                    let timer = timed.timer();
                    let mut cuda_ctx =
                        create_cuda_execution_ctx().with_launch_strategy(Arc::new(timed));
                    let (_, input_buffer) =
                        block_on(device_validity_buffer(len, ARROW_OFFSET, &mut cuda_ctx))
                            .vortex_expect("failed to create validity fixture");

                    for _ in 0..iters {
                        let null_count = test_harness::count_arrow_validity_nulls(
                            &input_buffer,
                            len,
                            ARROW_OFFSET,
                            &mut cuda_ctx,
                        )
                        .vortex_expect("failed to count Arrow validity nulls");
                        std::hint::black_box(null_count);
                    }

                    Duration::from_nanos(timer.load(Ordering::Relaxed))
                });
            },
        );
    }

    group.finish();
}

criterion::criterion_group! {
    name = benches;
    config = bench_config::cuda_bench_config();
    targets = benchmark_arrow_validity_repack, benchmark_arrow_validity_count_nulls, benchmark_arrow_validity_export
}

#[cuda_available]
criterion::criterion_main!(benches);

#[cuda_not_available]
fn main() {}
