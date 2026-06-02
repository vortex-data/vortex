// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA benchmarks for Arrow Device export of Vortex list-view arrays.

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
use futures::executor::block_on;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::arrays::ListViewArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::validity::Validity;
use vortex::dtype::PType;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::session::VortexSession;
use vortex_cuda::CudaExecutionCtx;
use vortex_cuda::CudaSession;
use vortex_cuda::arrow::ArrowDeviceArray;
use vortex_cuda::arrow::DeviceArrayExt;
use vortex_cuda_macros::cuda_available;
use vortex_cuda_macros::cuda_not_available;

use crate::timed_launch_strategy::TimedLaunchStrategy;

const LIST_VIEW_CONTIGUOUS_BENCH_SIZES: &[(usize, &str)] = &[(10_000_000, "10M")];
const LIST_VIEW_REBUILD_BENCH_SIZES: &[(usize, &str)] = &[(10_000_000, "10M")];

async fn primitive_i32_on_device(
    values: impl IntoIterator<Item = i32>,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<ArrayRef> {
    let primitive = PrimitiveArray::from_iter(values);
    let handle = ctx
        .ensure_on_device(primitive.buffer_handle().clone())
        .await?;
    Ok(PrimitiveArray::from_buffer_handle(handle, PType::I32, Validity::NonNullable).into_array())
}

async fn contiguous_list_view(len: usize, ctx: &mut CudaExecutionCtx) -> VortexResult<ArrayRef> {
    let elements = primitive_i32_on_device((0..len).map(|value| value as i32), ctx).await?;
    let offsets = primitive_i32_on_device((0..len).map(|value| value as i32), ctx).await?;
    let sizes = primitive_i32_on_device(std::iter::repeat_n(1i32, len), ctx).await?;

    Ok(ListViewArray::new(elements, offsets, sizes, Validity::NonNullable).into_array())
}

async fn non_contiguous_primitive_list_view(
    len: usize,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<ArrayRef> {
    let elements = primitive_i32_on_device((0..len).map(|value| value as i32), ctx).await?;
    let offsets = primitive_i32_on_device((0..len).rev().map(|value| value as i32), ctx).await?;
    let sizes = primitive_i32_on_device(std::iter::repeat_n(1i32, len), ctx).await?;

    Ok(ListViewArray::new(elements, offsets, sizes, Validity::NonNullable).into_array())
}

unsafe fn release_arrow_device_array(array: &mut ArrowDeviceArray) {
    unsafe {
        if let Some(release) = array.array.release {
            release(&raw mut array.array);
        }
    }
}

fn benchmark_list_view_export(c: &mut Criterion) {
    let mut group = c.benchmark_group("cuda");

    for &(len, len_label) in LIST_VIEW_CONTIGUOUS_BENCH_SIZES {
        // Contiguous path reads offsets/sizes and writes Arrow offsets.
        group.throughput(Throughput::Bytes((len * size_of::<i32>() * 3) as u64));
        group.bench_with_input(
            BenchmarkId::new("cuda/list_view/contiguous_offsets", len_label),
            &len,
            |b, &len| {
                b.iter_custom(|iters| {
                    let timed = TimedLaunchStrategy::default();
                    let timer = timed.timer();

                    let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
                        .vortex_expect("failed to create execution context")
                        .with_launch_strategy(Arc::new(timed));
                    let array = block_on(contiguous_list_view(len, &mut cuda_ctx))
                        .vortex_expect("failed to create list-view fixture");

                    for _ in 0..iters {
                        let mut exported =
                            block_on(array.clone().export_device_array(&mut cuda_ctx))
                                .vortex_expect("failed to export device array");
                        unsafe { release_arrow_device_array(&mut exported) };
                    }

                    Duration::from_nanos(timer.load(Ordering::Relaxed))
                });
            },
        );
    }

    for &(len, len_label) in LIST_VIEW_REBUILD_BENCH_SIZES {
        // Rebuild path scans sizes into Arrow offsets, then gathers primitive child values.
        group.throughput(Throughput::Bytes((len * size_of::<i32>() * 4) as u64));
        group.bench_with_input(
            BenchmarkId::new("cuda/list_view/rebuild_primitive", len_label),
            &len,
            |b, &len| {
                b.iter_custom(|iters| {
                    let timed = TimedLaunchStrategy::default();
                    let timer = timed.timer();

                    let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
                        .vortex_expect("failed to create execution context")
                        .with_launch_strategy(Arc::new(timed));
                    let array = block_on(non_contiguous_primitive_list_view(len, &mut cuda_ctx))
                        .vortex_expect("failed to create list-view fixture");

                    for _ in 0..iters {
                        let mut exported =
                            block_on(array.clone().export_device_array(&mut cuda_ctx))
                                .vortex_expect("failed to export device array");
                        unsafe { release_arrow_device_array(&mut exported) };
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
    targets = benchmark_list_view_export
}

#[cuda_available]
criterion::criterion_main!(benches);

#[cuda_not_available]
fn main() {}
