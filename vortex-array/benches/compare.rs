#![allow(clippy::unwrap_used)]

use criterion::{criterion_group, criterion_main, Criterion};
use rand::distributions::Uniform;
use rand::{thread_rng, Rng};
use vortex_array::array::BoolArray;
use vortex_array::compute::Operator;
use vortex_array::IntoArray;
use vortex_buffer::Buffer;

fn compare_bool(c: &mut Criterion) {
    let mut group = c.benchmark_group("compare");

    let mut rng = thread_rng();
    let range = Uniform::new(0u8, 1);
    let arr = BoolArray::from_iter((0..10_000_000).map(|_| rng.sample(range) == 0)).into_array();
    let arr2 = BoolArray::from_iter((0..10_000_000).map(|_| rng.sample(range) == 0)).into_array();

    group.bench_function("compare_bool", |b| {
        b.iter(|| vortex_array::compute::compare(&arr, &arr2, Operator::Gte).unwrap());
    });
}

fn compare_primitive(c: &mut Criterion) {
    let mut group = c.benchmark_group("compare");

    let mut rng = thread_rng();
    let range = Uniform::new(0i64, 100_000_000);
    let arr = (0..10_000_000)
        .map(|_| rng.sample(range))
        .collect::<Buffer<_>>()
        .into_array();

    let arr2 = (0..10_000_000)
        .map(|_| rng.sample(range))
        .collect::<Buffer<_>>()
        .into_array();

    group.bench_function("compare_int", |b| {
        b.iter(|| vortex_array::compute::compare(&arr, &arr2, Operator::Gte).unwrap());
    });
}

criterion_group!(benches, compare_primitive, compare_bool);
criterion_main!(benches);
