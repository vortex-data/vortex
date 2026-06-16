// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA kernel-time benchmark for Arrow Device export of binary view arrays as Arrow Binary.

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
use vortex::array::arrays::VarBinViewArray;
use vortex::array::arrays::varbinview::BinaryView;
use vortex::array::buffer::BufferHandle;
use vortex::array::validity::Validity;
use vortex::buffer::Buffer;
use vortex::buffer::ByteBuffer;
use vortex::dtype::DType;
use vortex::dtype::Nullability;
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

const BINARY_BENCH_SIZES: &[(usize, &str)] = &[(10_000_000, "10M")];
const VALUE_BYTES: usize = 16;

/// Build a non-nullable binary view array of `len` out-of-line values on the device.
async fn out_of_line_binary(len: usize, ctx: &mut CudaExecutionCtx) -> VortexResult<ArrayRef> {
    let values = ByteBuffer::copy_from(vec![b'x'; len * VALUE_BYTES]);
    let views = Buffer::from_iter((0..len).map(|idx| {
        let offset = idx * VALUE_BYTES;
        BinaryView::make_view(
            &values.slice(offset..offset + VALUE_BYTES),
            0,
            offset as u32,
        )
    }));

    let views = ctx
        .ensure_on_device(BufferHandle::new_host(views.into_byte_buffer()))
        .await?;
    let values = ctx.ensure_on_device(BufferHandle::new_host(values)).await?;

    Ok(VarBinViewArray::new_handle(
        views,
        Arc::from([values]),
        DType::Binary(Nullability::NonNullable),
        Validity::NonNullable,
    )
    .into_array())
}

unsafe fn release_arrow_device_array(array: &mut ArrowDeviceArray) {
    unsafe {
        if let Some(release) = array.array.release {
            release(&raw mut array.array);
        }
    }
}

fn benchmark_arrow_binary_export(c: &mut Criterion) {
    let mut group = c.benchmark_group("cuda");

    for &(len, len_label) in BINARY_BENCH_SIZES {
        // Kernels read views and value bytes and write offsets plus gathered values.
        group.throughput(Throughput::Bytes(
            (len * (size_of::<BinaryView>() + VALUE_BYTES + size_of::<i32>())) as u64,
        ));
        group.bench_with_input(
            BenchmarkId::new("cuda/arrow_binary_kernel_time/out_of_line", len_label),
            &len,
            |b, &len| {
                b.iter_custom(|iters| {
                    let timed = TimedLaunchStrategy::default();
                    let timer = timed.timer();

                    let mut cuda_ctx = CudaSession::create_execution_ctx(&vortex_cuda::cuda_session())
                        .vortex_expect("failed to create execution context")
                        .with_launch_strategy(Arc::new(timed));
                    let array = block_on(out_of_line_binary(len, &mut cuda_ctx))
                        .vortex_expect("failed to create binary fixture");

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
    targets = benchmark_arrow_binary_export
}

#[cuda_available]
criterion::criterion_main!(benches);

#[cuda_not_available]
fn main() {}
