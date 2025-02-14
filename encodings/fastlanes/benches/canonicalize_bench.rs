use divan::Bencher;
use rand::prelude::StdRng;
use rand::SeedableRng;
use vortex_array::array::ChunkedArray;
use vortex_array::builders::{ArrayBuilder, PrimitiveBuilder};
use vortex_array::{IntoArray, IntoCanonical};
use vortex_error::{VortexExpect as _, VortexUnwrap};
use vortex_fastlanes::test_harness::make_array;

fn main() {
    divan::main();
}

const BENCH_ARGS: [(usize, usize, f64); 10] = [
    (100000, 1, 0.10),
    (100000, 1, 0.01),
    (100000, 1, 0.00),
    (100000, 10, 0.10),
    (100000, 10, 0.01),
    (100000, 10, 0.00),
    (100000, 100, 0.10),
    (100000, 100, 0.01),
    (100000, 100, 0.00),
    (100000, 1000, 0.00),
    // Too slow for 1000 samples. Try 10 samples.
    // (1000000, 100, 0.00),
    // (1000000, 1000, 0.00),
    // (10000000, 100, 0.00),
];

#[divan::bench(args = BENCH_ARGS)]
fn into_canonical_non_nullable(
    bencher: Bencher,
    (chunk_len, chunk_count, fraction_patched): (usize, usize, f64),
) {
    let mut rng = StdRng::seed_from_u64(0);

    let chunks = (0..chunk_count)
        .map(|_| {
            make_array(&mut rng, chunk_len, fraction_patched, 0.0).vortex_expect("make_array works")
        })
        .collect::<Vec<_>>();
    let chunked = ChunkedArray::from_iter(chunks).into_array();

    bencher
        .with_inputs(|| chunked.clone())
        .bench_values(|chunked| chunked.into_canonical().vortex_unwrap());
}

#[divan::bench(args = BENCH_ARGS)]
fn canonical_into_non_nullable(
    bencher: Bencher,
    (chunk_len, chunk_count, fraction_patched): (usize, usize, f64),
) {
    let mut rng = StdRng::seed_from_u64(0);

    let chunks = (0..chunk_count)
        .map(|_| {
            make_array(&mut rng, chunk_len, fraction_patched, 0.0).vortex_expect("make_array works")
        })
        .collect::<Vec<_>>();
    let chunked = ChunkedArray::from_iter(chunks).into_array();

    bencher
        .with_inputs(|| chunked.clone())
        .bench_values(|chunked| {
            let mut primitive_builder = PrimitiveBuilder::<i32>::with_capacity(
                chunked.dtype().nullability(),
                chunk_len * chunk_count,
            );
            chunked
                .canonicalize_into(&mut primitive_builder)
                .vortex_unwrap();
            primitive_builder.finish().vortex_unwrap()
        });
}

const NULLABLE_BENCH_ARGS: [(usize, usize, f64); 6] = [
    (100000, 1, 0.10),
    (100000, 1, 0.00),
    (100000, 10, 0.10),
    (100000, 10, 0.00),
    (100000, 100, 0.10),
    (100000, 100, 0.00),
];

#[divan::bench(args = NULLABLE_BENCH_ARGS)]
fn into_canonical_nullable(
    bencher: Bencher,
    (chunk_len, chunk_count, fraction_patched): (usize, usize, f64),
) {
    let mut rng = StdRng::seed_from_u64(0);

    let chunks = (0..chunk_count)
        .map(|_| {
            make_array(&mut rng, chunk_len, fraction_patched, 0.05)
                .vortex_expect("make_array works")
        })
        .collect::<Vec<_>>();
    let chunked = ChunkedArray::from_iter(chunks).into_array();

    bencher
        .with_inputs(|| chunked.clone())
        .bench_values(|chunked| chunked.into_canonical().vortex_unwrap());
}

#[divan::bench(args = NULLABLE_BENCH_ARGS)]
fn canonical_into_nullable(
    bencher: Bencher,
    (chunk_len, chunk_count, fraction_patched): (usize, usize, f64),
) {
    let mut rng = StdRng::seed_from_u64(0);

    let chunks = (0..chunk_count)
        .map(|_| {
            make_array(&mut rng, chunk_len, fraction_patched, 0.05)
                .vortex_expect("make_array works")
        })
        .collect::<Vec<_>>();
    let chunked = ChunkedArray::from_iter(chunks).into_array();

    bencher
        .with_inputs(|| chunked.clone())
        .bench_values(|chunked| {
            let mut primitive_builder = PrimitiveBuilder::<i32>::with_capacity(
                chunked.dtype().nullability(),
                chunk_len * chunk_count,
            );
            chunked
                .canonicalize_into(&mut primitive_builder)
                .vortex_unwrap();
            primitive_builder.finish().vortex_unwrap()
        });
}
