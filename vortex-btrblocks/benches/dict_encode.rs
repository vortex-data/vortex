#![allow(clippy::unwrap_used)]

use criterion::{criterion_group, criterion_main, Criterion};
use vortex_array::array::{BoolArray, PrimitiveArray};
use vortex_array::validity::Validity;
use vortex_array::IntoArray;
use vortex_btrblocks::integer::dictionary::dictionary_encode;
use vortex_btrblocks::integer::IntegerStats;
use vortex_btrblocks::CompressorStats;
use vortex_buffer::BufferMut;
use vortex_dict::builders::dict_encode;

fn encode(c: &mut Criterion) {
    let values: BufferMut<i32> = (0..50).cycle().take(64_000).collect();

    let nulls = BoolArray::from_iter(
        [true, true, true, true, true, true, false]
            .into_iter()
            .cycle()
            .take(64_000),
    )
    .into_array();

    let primitive = PrimitiveArray::new(values, Validity::Array(nulls));
    let array = primitive.clone().into_array();

    let stats = IntegerStats::generate(&primitive);

    // Generic variant
    c.bench_function("generic", |b| b.iter(|| dict_encode(&array).unwrap()));

    c.bench_function("specialized", |b| {
        b.iter(|| dictionary_encode(&stats).unwrap())
    });
}

criterion_group!(benches, encode);
criterion_main!(benches);
