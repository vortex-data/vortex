//! Kernel microbench. Runs the dot-product kernel on a hot-cache L1/L2 buffer
//! so we measure compute throughput, not memory bandwidth.
//!
//! Reports throughput in GFLOPs/sec per iteration (dot product of D elements
//! is 2*D floating-point operations - one multiply + one add per lane).
#![allow(clippy::many_single_char_names)]

use std::hint::black_box;
use std::time::Duration;

use cosine_similarity_bench::kernel::DotKernel;
use cosine_similarity_bench::kernel::dot_scalar;
use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use criterion::criterion_group;
use criterion::criterion_main;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;

fn random_unit(dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut v: Vec<f32> = (0..dim).map(|_| rng.random_range(-1.0f32..1.0f32)).collect();
    let n = v
        .iter()
        .map(|x| x * x)
        .sum::<f32>()
        .sqrt()
        .max(f32::MIN_POSITIVE);
    for x in &mut v {
        *x /= n;
    }
    v
}

fn bench_dims(c: &mut Criterion) {
    let mut g = c.benchmark_group("dot_product");
    g.measurement_time(Duration::from_secs(3));

    let detected = DotKernel::detect();
    println!("kernel at bench time: {}", detected.name());

    for &dim in &[128usize, 512, 1024, 1536, 4096] {
        let a = random_unit(dim, 1);
        let b = random_unit(dim, 2);

        // Per iteration: 2*dim FLOPs (mul + add).
        g.throughput(Throughput::Elements((2 * dim) as u64));

        g.bench_with_input(BenchmarkId::new("scalar", dim), &dim, |bn, _| {
            bn.iter(|| {
                let s = dot_scalar(black_box(&a), black_box(&b));
                black_box(s)
            })
        });

        g.bench_with_input(BenchmarkId::new("dispatch", dim), &dim, |bn, _| {
            bn.iter(|| {
                let s = detected.dot(black_box(&a), black_box(&b));
                black_box(s)
            })
        });
    }
    g.finish();
}

criterion_group!(benches, bench_dims);
criterion_main!(benches);
