// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]

use std::thread;
use std::time::Duration;
use std::time::Instant;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use mimalloc::MiMalloc;
use vortex::array::Canonical;
use vortex::array::LEGACY_SESSION;
use vortex::array::VortexSessionExecute;
use vortex::array::aggregate_fn::fns::sum::sum;
use vortex_fsst::FSSTArrayExt;
use vortex_fsst::test_utils::make_fsst_clickbench_urls;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

const BENCH_ARGS: &[(usize, &str)] = &[(1_000_000, "1M"), (5_000_000, "5M"), (10_000_000, "10M")];

fn benchmark_fsst_cuda_decompress(c: &mut Criterion) {
    let num_threads = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let mut group = c.benchmark_group("FSST_cuda");

    for (n, label) in BENCH_ARGS {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let array = make_fsst_clickbench_urls(*n, &mut ctx);
        let uncompressed_bytes = sum(array.uncompressed_lengths(), &mut ctx)
            .unwrap()
            .as_primitive()
            .typed_value::<i64>()
            .unwrap();

        let shard_size = n.div_ceil(num_threads);

        group.throughput(Throughput::Bytes(uncompressed_bytes as u64));
        group.bench_with_input(
            BenchmarkId::new("decompress_kernel", label),
            &array,
            |b, array| {
                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iters {
                        let start = Instant::now();
                        thread::scope(|s| {
                            for i in 0..num_threads {
                                let lo = i * shard_size;
                                let hi = (lo + shard_size).min(*n);
                                if lo >= hi {
                                    continue;
                                }
                                let shard = array.slice(lo..hi).unwrap();
                                s.spawn(move || {
                                    let mut ctx = LEGACY_SESSION.create_execution_ctx();
                                    shard.execute::<Canonical>(&mut ctx).unwrap();
                                });
                            }
                        });
                        total += start.elapsed();
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
    config = Criterion::default().without_plots()
        .sample_size(10)
        .warm_up_time(Duration::from_nanos(1))
        .measurement_time(Duration::from_nanos(1))
        .nresamples(10);
    targets = benchmark_fsst_cuda_decompress
}

criterion::criterion_main!(benches);
