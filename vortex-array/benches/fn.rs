#![allow(clippy::unwrap_used)]

use arrow_array::types::UInt32Type;
use arrow_array::UInt32Array;
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use vortex_array::array::PrimitiveArray;
use vortex_array::elementwise::{BinaryFn, UnaryFn};
use vortex_array::validity::Validity;
use vortex_array::IntoArrayData;

fn vortex_unary_add(c: &mut Criterion) {
    let data = PrimitiveArray::from_vec((0_u32..1_000_000).collect::<Vec<_>>(), Validity::AllValid);
    c.bench_function("vortex_unary_add", |b| {
        b.iter_batched(
            || (data.clone()),
            |data| data.unary(|v: u32| v + 1).unwrap(),
            BatchSize::SmallInput,
        )
    });
}

fn arrow_unary_add(c: &mut Criterion) {
    let data = UInt32Array::from_iter_values(0_u32..1_000_000);
    c.bench_function("arrow_unary_add", |b| {
        b.iter_batched(
            || data.clone(),
            |data: arrow_array::PrimitiveArray<UInt32Type>| data.unary::<_, UInt32Type>(|v| v + 1),
            BatchSize::SmallInput,
        )
    });
}

fn vortex_binary_add(c: &mut Criterion) {
    let lhs = PrimitiveArray::from_vec((0_u32..1_000_000).collect::<Vec<_>>(), Validity::AllValid);
    let rhs = PrimitiveArray::from_vec((0_u32..1_000_000).collect::<Vec<_>>(), Validity::AllValid)
        .into_array();
    c.bench_function("vortex_binary_add", |b| {
        b.iter_batched(
            || (lhs.clone(), rhs.clone()),
            |(lhs, rhs)| lhs.binary(rhs, |l: u32, r: u32| l + r),
            BatchSize::SmallInput,
        )
    });
}

fn arrow_binary_add(c: &mut Criterion) {
    let lhs = UInt32Array::from_iter_values(0_u32..1_000_000);
    let rhs = UInt32Array::from_iter_values(0_u32..1_000_000);
    c.bench_function("arrow_binary_add", |b| {
        b.iter_batched(
            || (lhs.clone(), rhs.clone()),
            |(lhs, rhs)| {
                arrow_arith::arity::binary::<_, _, _, UInt32Type>(&lhs, &rhs, |a, b| a + b).unwrap()
            },
            BatchSize::SmallInput,
        )
    });
}

criterion_group!(
    name = benches;
    config = Criterion::default();
    targets =
    arrow_unary_add,
    vortex_unary_add,
    arrow_binary_add,
    vortex_binary_add,
);
criterion_main!(benches);
