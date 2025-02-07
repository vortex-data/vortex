use divan::Bencher;
use rand::prelude::StdRng;
use rand::{Rng, SeedableRng};
use vortex_array::array::{BoolArray, ChunkedArray};
use vortex_array::builders::builder_with_capacity;
use vortex_array::{Array, IntoArray, IntoCanonical};

fn main() {
    divan::main();
}

fn make_opt_bool_chunks(len: usize, chunk_count: usize) -> Array {
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

fn make_bool_chunks(len: usize, chunk_count: usize) -> Array {
    let mut rng = StdRng::seed_from_u64(0);

    (0..chunk_count)
        .map(|_| {
            BoolArray::from_iter((0..len).map(|_| match rng.gen_range::<u8, _>(0..=1) {
                0 => false,
                1 => true,
                _ => unreachable!(),
            }))
            .into_array()
        })
        .collect::<ChunkedArray>()
        .into_array()
}

fn params() -> impl Iterator<Item = &'static (usize, usize)> {
    [
        (1_000usize, 10usize),
        (1_000, 1_000),
        (10_000, 100),
        (100_000, 1000),
    ]
    .iter()
}

#[divan::bench(args=params())]
fn chunked_bool_canonical_into(bencher: Bencher, (len, chunk_count): (usize, usize)) {
    let chunk = make_bool_chunks(len, chunk_count);

    bencher.bench(|| {
        let mut builder = builder_with_capacity(chunk.dtype(), len * chunk_count);
        chunk.clone().canonicalize_into(builder.as_mut()).unwrap();
        builder.finish().unwrap()
    })
}

#[divan::bench(args=params())]
fn chunked_opt_bool_canonical_into(bencher: Bencher, (len, chunk_count): (usize, usize)) {
    let chunk = make_opt_bool_chunks(len, chunk_count);

    let mut builder = builder_with_capacity(chunk.dtype(), len * chunk_count);
    chunk.clone().canonicalize_into(builder.as_mut()).unwrap();
    let res = builder
        .finish()
        .unwrap()
        .into_canonical()
        .unwrap()
        .into_bool()
        .unwrap();

    let res2 = chunk.clone().into_canonical().unwrap().into_bool().unwrap();

    assert_eq!(res.validity(), res2.validity());
    assert_eq!(res.boolean_buffer(), res2.boolean_buffer());

    bencher.bench(|| {
        let mut builder = builder_with_capacity(chunk.dtype(), len * chunk_count);
        chunk.clone().canonicalize_into(builder.as_mut()).unwrap();
        builder.finish().unwrap()
    })
}

#[divan::bench(args=params())]
fn chunked_bool_into_canonical(bencher: Bencher, (len, chunk_count): (usize, usize)) {
    let chunk = make_bool_chunks(len, chunk_count);

    bencher.bench(|| chunk.clone().into_canonical())
}

#[divan::bench(args=params())]
fn chunked_opt_bool_into_canonical(bencher: Bencher, (len, chunk_count): (usize, usize)) {
    let chunk = make_opt_bool_chunks(len, chunk_count);

    bencher.bench(|| chunk.clone().into_canonical())
}
