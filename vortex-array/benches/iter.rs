#![allow(clippy::unwrap_used)]

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use itertools::Itertools;
use vortex_array::array::PrimitiveArray;
use vortex_array::iter::VectorizedArrayIter;
use vortex_array::validity::Validity;
use vortex_array::variants::ArrayVariants;

fn std_iter(c: &mut Criterion) {
    let data = (0_u32..1_000_000).map(Some).collect_vec();
    c.bench_function("std_iter", |b| {
        b.iter_batched(|| data.iter().copied(), do_work, BatchSize::SmallInput)
    });
}

fn std_iter_no_option(c: &mut Criterion) {
    let data = (0_u32..1_000_000).collect_vec();
    c.bench_function("std_iter_no_option", |b| {
        b.iter_batched(
            || data.iter().copied(),
            |mut iter| {
                let mut u = 0;
                for n in iter.by_ref() {
                    u += n;
                }
                u
            },
            BatchSize::SmallInput,
        )
    });
}

fn vortex_iter(c: &mut Criterion) {
    let data = PrimitiveArray::from_vec((0_u32..1_000_000).collect_vec(), Validity::AllValid);

    c.bench_function("vortex_iter", |b| {
        b.iter_batched(
            || data.as_primitive_array_unchecked().u32_iter().unwrap(),
            do_work_vortex,
            BatchSize::SmallInput,
        )
    });
}

fn vortex_iter_flat(c: &mut Criterion) {
    let data = PrimitiveArray::from_vec((0_u32..1_000_000).collect_vec(), Validity::AllValid);

    c.bench_function("vortex_iter_flat", |b| {
        b.iter_batched(
            || {
                data.as_primitive_array_unchecked()
                    .u32_iter()
                    .unwrap()
                    .flatten()
            },
            do_work,
            BatchSize::SmallInput,
        )
    });
}

fn arrow_iter(c: &mut Criterion) {
    let data = arrow_array::UInt32Array::from_iter(0_u32..1_000_000);
    c.bench_function("arrow_iter", |b| {
        b.iter_batched(|| data.iter(), do_work, BatchSize::SmallInput)
    });
}

fn do_work(
    mut iter: impl Iterator<Item = Option<u32>>,
) -> (u32, impl Iterator<Item = Option<u32>>) {
    let mut u = 0;
    for n in iter.by_ref() {
        u += n.unwrap();
    }
    (u, iter)
}

fn do_work_vortex(iter: VectorizedArrayIter<u32>) -> u32 {
    let mut sum = 0;
    for batch in iter {
        for idx in 0..batch.len() {
            if batch.is_valid(idx) {
                sum += unsafe { *batch.get_unchecked(idx) };
            }
        }
    }

    sum
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(100);
    targets = std_iter_no_option,
    std_iter,
    vortex_iter,
    vortex_iter_flat,
    arrow_iter,
);
criterion_main!(benches);
