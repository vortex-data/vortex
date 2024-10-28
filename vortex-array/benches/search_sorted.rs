use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::distributions::Uniform;
use rand::{thread_rng, Rng};
use vortex_array::compute::{SearchSorted, SearchSortedSide};

fn search_sorted(c: &mut Criterion) {
    let mut group = c.benchmark_group("search_sorted");

    let mut rng = thread_rng();
    let range = Uniform::new(0, 100_000_000);
    let mut data: Vec<i32> = (0..10_000_000).map(|_| rng.sample(range)).collect();
    data.sort();
    let needle = rng.sample(range);

    group.bench_function("std", |b| b.iter(|| black_box(data.binary_search(&needle))));

    group.bench_function("vortex", |b| {
        b.iter(|| black_box(data.search_sorted(&needle, SearchSortedSide::Left)))
    });
}

criterion_group!(benches, search_sorted);
criterion_main!(benches);
