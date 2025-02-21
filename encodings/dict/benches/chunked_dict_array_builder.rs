use divan::Bencher;
use rand::distributions::{Distribution, Standard};
use vortex_array::arrays::ChunkedArray;
use vortex_array::builders::builder_with_capacity;
use vortex_array::{Array, IntoArray, IntoCanonical};
use vortex_dict::test::{gen_dict_fsst_test_data, gen_dict_primitive_chunks};
use vortex_dtype::NativePType;
use vortex_error::VortexUnwrap;

fn main() {
    divan::main();
}

const BENCH_ARGS: &[(usize, usize, usize)] = &[
    (1000, 10, 10),
    (1000, 100, 10),
    (1000, 1000, 10),
    (1000, 10, 100),
    (1000, 100, 100),
    (1000, 1000, 100),
];

#[divan::bench(types = [u32, u64, f32, f64], args = BENCH_ARGS)]
fn chunked_dict_primitive_canonical_into<T: NativePType>(
    bencher: Bencher,
    (len, unique_values, chunk_count): (usize, usize, usize),
) where
    Standard: Distribution<T>,
{
    let chunk = gen_dict_primitive_chunks::<T, u16>(len, unique_values, chunk_count);

    bencher.with_inputs(|| chunk.clone()).bench_values(|chunk| {
        let mut builder = builder_with_capacity(chunk.dtype(), len * chunk_count);
        chunk.canonicalize_into(builder.as_mut()).vortex_unwrap();
        builder.finish()
    })
}

#[divan::bench(types = [u32, u64, f32, f64], args = BENCH_ARGS)]
fn chunked_dict_primitive_into_canonical<T: NativePType>(
    bencher: Bencher,
    (len, unique_values, chunk_count): (usize, usize, usize),
) where
    Standard: Distribution<T>,
{
    let chunk = gen_dict_primitive_chunks::<T, u16>(len, unique_values, chunk_count);

    bencher
        .with_inputs(|| chunk.clone())
        .bench_values(|chunk| chunk.into_canonical().vortex_unwrap())
}

fn make_dict_fsst_chunks<T: NativePType>(
    len: usize,
    unique_values: usize,
    chunk_count: usize,
) -> Array {
    (0..chunk_count)
        .map(|_| gen_dict_fsst_test_data::<T>(len, unique_values, 20, 30).into_array())
        .collect::<ChunkedArray>()
        .into_array()
}

#[divan::bench(args = BENCH_ARGS)]
fn chunked_dict_fsst_canonical_into(
    bencher: Bencher,
    (len, unique_values, chunk_count): (usize, usize, usize),
) {
    let chunk = make_dict_fsst_chunks::<u16>(len, unique_values, chunk_count);

    bencher.with_inputs(|| chunk.clone()).bench_values(|chunk| {
        let mut builder = builder_with_capacity(chunk.dtype(), len * chunk_count);
        chunk.canonicalize_into(builder.as_mut()).vortex_unwrap();
        builder.finish()
    })
}

#[divan::bench(args = BENCH_ARGS)]
fn chunked_dict_fsst_into_canonical(
    bencher: Bencher,
    (len, unique_values, chunk_count): (usize, usize, usize),
) {
    let chunk = make_dict_fsst_chunks::<u16>(len, unique_values, chunk_count);

    bencher
        .with_inputs(|| chunk.clone())
        .bench_values(|chunk| chunk.into_canonical().vortex_unwrap())
}
