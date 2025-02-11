use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use vortex_array::array::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_btrblocks::integer::IntegerStats;
use vortex_btrblocks::{CompressorStats, GenerateStatsOptions};
use vortex_buffer::Buffer;

fn bench(c: &mut Criterion) {
    let values: Buffer<u32> = (0..1024).cycle().take(1_000_000).collect();
    let values = PrimitiveArray::new(values, Validity::NonNullable);

    let mut group = c.benchmark_group("stats_cal");
    group.throughput(Throughput::Elements(1_000_000));
    group.bench_function("integer_stats", |b| {
        b.iter(|| IntegerStats::generate(&values))
    });

    group.bench_function("integer_stats_nodict", |b| {
        b.iter(|| {
            IntegerStats::generate_opts(
                &values,
                GenerateStatsOptions {
                    count_distinct_values: false,
                },
            )
        })
    });

    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
