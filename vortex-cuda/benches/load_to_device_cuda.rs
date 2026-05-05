// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::expect_used)]

mod bench_config;
// Unused here but suppresses dead_code warning for the shared module.
const _: &[(usize, &str)] = bench_config::BENCH_SIZES;

use criterion::BatchSize;
use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use vortex::array::buffer::BufferHandle;
use vortex::buffer::ByteBuffer;
use vortex::error::VortexExpect;
use vortex::session::VortexSession;
use vortex_cuda::CudaSession;
use vortex_cuda_macros::cuda_available;
use vortex_cuda_macros::cuda_not_available;

const LOAD_SIZES: &[(usize, &str)] = &[
    (16 * 1024 * 1024, "16MiB"),
    (64 * 1024 * 1024, "64MiB"),
    (256 * 1024 * 1024, "256MiB"),
    (1024 * 1024 * 1024, "1GiB"),
];

fn benchmark_load_to_device(c: &mut Criterion) {
    let mut group = c.benchmark_group("cuda");

    for &(size, size_name) in LOAD_SIZES {
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(
            BenchmarkId::new("cuda/load_to_device/ensure_on_device_sync", size_name),
            &size,
            |b, &size| {
                let session = VortexSession::empty();
                let cuda_ctx =
                    CudaSession::create_execution_ctx(&session).vortex_expect("cuda ctx");

                b.iter_batched(
                    || BufferHandle::new_host(ByteBuffer::from(vec![0xA5; size])),
                    |source| {
                        let handle = cuda_ctx
                            .ensure_on_device_sync(source)
                            .vortex_expect("ensure_on_device_sync");
                        assert!(handle.is_on_device());
                        // Keep the explicit sync here to ensure that we measure a sync copy. In
                        // case the default buffer allocation strategy in the future changes to use
                        // `cuMemHostAlloc`, the htod copy would change to being async, making the
                        // function return immediately.
                        cuda_ctx.stream().synchronize().expect("synchronize stream");
                    },
                    BatchSize::PerIteration,
                );

                drop(cuda_ctx);
            },
        );
    }

    group.finish();
}

criterion::criterion_group! {
    name = benches;
    config = bench_config::cuda_bench_config();
    targets = benchmark_load_to_device
}

#[cuda_available]
criterion::criterion_main!(benches);

#[cuda_not_available]
fn main() {}
