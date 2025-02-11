use criterion::{criterion_group, criterion_main, Criterion};
use rand::distributions::Uniform;
use rand::prelude::StdRng;
use rand::{Rng, SeedableRng};
use vortex_array::compute::{SearchSorted, SearchSortedSide};

fn search_sorted(c: &mut Criterion) {
    let mut group = c.benchmark_group("search_sorted");

    let mut rng = StdRng::seed_from_u64(3440);
    let range = Uniform::new(0, 100_000_000);
    let mut data: Vec<i32> = (0..10_000_000).map(|_| rng.sample(range)).collect();
    data.sort();
    let needle = rng.sample(range);

    group.bench_function("std", |b| b.iter(|| data.binary_search(&needle)));
    group.bench_function("vortex", |b| {
        b.iter(|| data.search_sorted(&needle, SearchSortedSide::Left))
    });
}

criterion_group!(benches, search_sorted);
criterion_main!(benches);
