#![allow(clippy::unwrap_used)]

use divan::Bencher;
use rand::distributions::Uniform;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use vortex_array::arrays::ChunkedArray;
use vortex_array::IntoArray;
use vortex_buffer::Buffer;

fn main() {
    divan::main();
}

#[divan::bench]
fn scalar_subtract(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(0);
    let range = Uniform::new(0i64, 100_000_000);

    let data1 = (0..100_000)
        .map(|_| rng.sample(range))
        .collect::<Buffer<i64>>()
        .into_array();

    let data2 = (0..100_000)
        .map(|_| rng.sample(range))
        .collect::<Buffer<i64>>()
        .into_array();

    let to_subtract = -1i64;

    let chunked = ChunkedArray::from_iter([data1, data2]).into_array();

    bencher.with_inputs(|| &chunked).bench_refs(|chunked| {
        let array = vortex_array::compute::sub_scalar(chunked, to_subtract.into()).unwrap();
        divan::black_box(array);
    });
}
