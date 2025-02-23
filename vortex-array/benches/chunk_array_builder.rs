use divan::Bencher;
use rand::prelude::StdRng;
use rand::{Rng, SeedableRng};
use vortex_array::arrays::{BoolArray, ChunkedArray};
use vortex_array::builders::{builder_with_capacity, ArrayBuilder, VarBinViewBuilder};
use vortex_array::{Array, ArrayRef};
use vortex_dtype::DType;
use vortex_error::VortexUnwrap;

fn main() {
    divan::main();
}

const BENCH_ARGS: &[(usize, usize)] = &[
    // length, chunk_count
    (10, 1000),
    (100, 100),
    (1000, 10),
];

#[divan::bench(args = BENCH_ARGS)]
fn chunked_bool_canonical_into(bencher: Bencher, (len, chunk_count): (usize, usize)) {
    let chunk = make_bool_chunks(len, chunk_count);

    bencher.with_inputs(|| chunk.clone()).bench_values(|chunk| {
        let mut builder = builder_with_capacity(chunk.dtype(), len * chunk_count);
        chunk.append_to_builder(builder.as_mut()).vortex_unwrap();
        builder.finish()
    })
}

#[divan::bench(args = BENCH_ARGS)]
fn chunked_opt_bool_canonical_into(bencher: Bencher, (len, chunk_count): (usize, usize)) {
    let chunk = make_opt_bool_chunks(len, chunk_count);

    bencher.with_inputs(|| chunk.clone()).bench_values(|chunk| {
        let mut builder = builder_with_capacity(chunk.dtype(), len * chunk_count);
        chunk
            .clone()
            .append_to_builder(builder.as_mut())
            .vortex_unwrap();
        builder.finish()
    })
}

#[divan::bench(args = BENCH_ARGS)]
fn chunked_bool_into_canonical(bencher: Bencher, (len, chunk_count): (usize, usize)) {
    let chunk = make_bool_chunks(len, chunk_count);

    bencher
        .with_inputs(|| chunk.clone())
        .bench_values(|chunk| chunk.to_canonical())
}

#[divan::bench(args = BENCH_ARGS)]
fn chunked_opt_bool_into_canonical(bencher: Bencher, (len, chunk_count): (usize, usize)) {
    let chunk = make_opt_bool_chunks(len, chunk_count);

    bencher
        .with_inputs(|| chunk.clone())
        .bench_values(|chunk| chunk.to_canonical())
}

#[divan::bench(args = BENCH_ARGS)]
fn chunked_varbinview_canonical_into(bencher: Bencher, (len, chunk_count): (usize, usize)) {
    let chunks = make_string_chunks(false, len, chunk_count);

    bencher
        .with_inputs(|| chunks.clone())
        .bench_values(|chunk| {
            let mut builder = VarBinViewBuilder::with_capacity(
                DType::Utf8(chunk.dtype().nullability()),
                len * chunk_count,
            );
            chunk.append_to_builder(&mut builder).vortex_unwrap();
            builder.finish()
        })
}

#[divan::bench(args = BENCH_ARGS)]
fn chunked_varbinview_into_canonical(bencher: Bencher, (len, chunk_count): (usize, usize)) {
    let chunks = make_string_chunks(false, len, chunk_count);

    bencher
        .with_inputs(|| chunks.clone())
        .bench_values(|chunk| chunk.to_canonical())
}

#[divan::bench(args = BENCH_ARGS)]
fn chunked_varbinview_opt_canonical_into(bencher: Bencher, (len, chunk_count): (usize, usize)) {
    let chunks = make_string_chunks(true, len, chunk_count);

    bencher
        .with_inputs(|| chunks.clone())
        .bench_values(|chunk| {
            let mut builder = VarBinViewBuilder::with_capacity(
                DType::Utf8(chunk.dtype().nullability()),
                len * chunk_count,
            );
            chunk.append_to_builder(&mut builder).vortex_unwrap();
            builder.finish()
        })
}

#[divan::bench(args = BENCH_ARGS)]
fn chunked_varbinview_opt_into_canonical(bencher: Bencher, (len, chunk_count): (usize, usize)) {
    let chunks = make_string_chunks(true, len, chunk_count);

    bencher
        .with_inputs(|| chunks.clone())
        .bench_values(|chunk| chunk.to_canonical())
}

fn make_opt_bool_chunks(len: usize, chunk_count: usize) -> ArrayRef {
    let mut rng = StdRng::seed_from_u64(0);

    const SPAN_LEN: usize = 10;
    assert!(len % SPAN_LEN == 0);

    (0..chunk_count)
        .map(|_| {
            BoolArray::from_iter(
                (0..len / SPAN_LEN)
                    .flat_map(|_| match rng.gen_range::<u8, _>(0..=2) {
                        0 => vec![Some(false); SPAN_LEN],
                        1 => vec![Some(true); SPAN_LEN],
                        2 => vec![None; SPAN_LEN],
                        _ => unreachable!(),
                    })
                    // To get a sized iterator
                    .collect::<Vec<Option<bool>>>(),
            )
            .into_array()
        })
        .collect::<ChunkedArray>()
        .into_array()
}

fn make_bool_chunks(len: usize, chunk_count: usize) -> ArrayRef {
    let mut rng = StdRng::seed_from_u64(0);

    (0..chunk_count)
        .map(|_| BoolArray::from_iter((0..len).map(|_| rng.gen_bool(0.5))).into_array())
        .collect::<ChunkedArray>()
        .into_array()
}

fn make_string_chunks(nullable: bool, len: usize, chunk_count: usize) -> ArrayRef {
    let mut rng = StdRng::seed_from_u64(123);

    (0..chunk_count)
        .map(|_| {
            let mut builder = VarBinViewBuilder::with_capacity(DType::Utf8(nullable.into()), len);
            (0..len).for_each(|_| {
                if nullable && rng.gen_bool(0.2) {
                    builder.append_null()
                } else {
                    builder.append_value(
                        (0..rng.gen_range(0..=20))
                            .map(|_| rng.gen_range(b'a'..=b'z'))
                            .collect::<Vec<u8>>(),
                    )
                }
            });
            builder.finish()
        })
        .collect::<ChunkedArray>()
        .into_array()
}
