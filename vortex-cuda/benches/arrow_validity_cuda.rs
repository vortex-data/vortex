// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA benchmarks for Arrow validity bitmap repacking.

mod bench_config;
mod timed_launch_strategy;

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use futures::executor::block_on;
use vortex::array::buffer::BufferHandle;
use vortex::buffer::BitBuffer;
use vortex::error::VortexExpect;
use vortex::session::VortexSession;
use vortex_cuda::CudaSession;
use vortex_cuda::arrow::test_harness;
use vortex_cuda_macros::cuda_available;
use vortex_cuda_macros::cuda_not_available;

use crate::timed_launch_strategy::TimedLaunchStrategy;

const INPUT_OFFSET: usize = 5;
const ARROW_OFFSET: usize = 3;

fn benchmark_arrow_validity_repack(c: &mut Criterion) {
    let mut group = c.benchmark_group("cuda");

    for &(len, len_label) in bench_config::BENCH_SIZES {
        group.throughput(Throughput::Elements(len as u64));
        group.bench_with_input(
            BenchmarkId::new("cuda/arrow_validity/repack", len_label),
            &len,
            |b, &len| {
                b.iter_custom(|iters| {
                    let timed = TimedLaunchStrategy::default();
                    let timer = timed.timer();

                    let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
                        .vortex_expect("failed to create execution context")
                        .with_launch_strategy(Arc::new(timed));
                    let source = BitBuffer::collect_bool(len + INPUT_OFFSET, |idx| idx % 3 != 0);
                    let sliced = source.slice(INPUT_OFFSET..INPUT_OFFSET + len);
                    let (input_offset, _, input_buffer) = sliced.into_inner();
                    let input_buffer =
                        block_on(cuda_ctx.ensure_on_device(BufferHandle::new_host(input_buffer)))
                            .vortex_expect("failed to copy validity input to device");

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

criterion::criterion_group! {
    name = benches;
    config = bench_config::cuda_bench_config();
    targets = benchmark_arrow_validity_repack
}

#[cuda_available]
criterion::criterion_main!(benches);

#[cuda_not_available]
fn main() {}
