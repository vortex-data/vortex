use divan::Bencher;
use rand::distributions::{Distribution, Standard};
use rand::prelude::StdRng;
use rand::{Rng, SeedableRng};
use vortex_array::array::{ChunkedArray, PrimitiveArray};
use vortex_array::builders::builder_with_capacity;
use vortex_array::{Array, IntoArray, IntoCanonical};
use vortex_dict::DictArray;
use vortex_dtype::NativePType;
use vortex_error::{VortexExpect, VortexUnwrap};

fn main() {
    divan::main();
}

fn make_dict_primitive_chunks<T: NativePType, U: NativePType>(
    len: usize,
    unique_values: usize,
    chunk_count: usize,
) -> Array
where
    Standard: Distribution<T>,
{
    let mut rng = StdRng::seed_from_u64(0);

    (0..chunk_count)
        .map(|_| {
            let values = (0..unique_values)
                .map(|_| rng.gen::<T>())
                .collect::<PrimitiveArray>();
            let codes = (0..len)
                .map(|_| U::from(rng.gen_range(0..unique_values)).vortex_expect("valid value"))
                .collect::<PrimitiveArray>();

            DictArray::try_new(codes.into_array(), values.into_array())
                .vortex_unwrap()
                .into_array()
        })
        .collect::<ChunkedArray>()
        .into_array()
}

const BENCH_ARGS: &[(usize, usize, usize)] = &[
    (8000, 10, 20),
    (8000, 100, 20),
    (8000, 1000, 20),
    (8000, 10, 200),
    (8000, 100, 200),
    (8000, 1000, 200),
    (8000, 10, 1000),
    (8000, 100, 1000),
    (8000, 1000, 1000),
    (32_000, 10, 100),
    (32_000, 100, 100),
    (32_000, 1000, 100),
    (32_000, 10, 1000),
    (32_000, 100, 1000),
    (32_000, 1000, 1000),
];

#[divan::bench(types = [u32], args=BENCH_ARGS)]
fn chunked_dict_primitive_canonical_into<T: NativePType>(
    bencher: Bencher,
    (len, unique_values, chunk_count): (usize, usize, usize),
) where
    Standard: Distribution<T>,
{
    let chunk = make_dict_primitive_chunks::<T, u16>(len, unique_values, chunk_count);

    bencher
        .with_inputs(|| chunk.clone())
        .bench_local_values(|chunk| {
            let mut builder = builder_with_capacity(chunk.dtype(), len * chunk_count);
            chunk.canonicalize_into(builder.as_mut()).vortex_unwrap();
            builder.finish().vortex_unwrap()
        })
}

#[divan::bench(types = [u32], args=BENCH_ARGS)]
fn chunked_dict_primitive_into_canonical<T: NativePType>(
    bencher: Bencher,
    (len, unique_values, chunk_count): (usize, usize, usize),
) where
    Standard: Distribution<T>,
{
    let chunk = make_dict_primitive_chunks::<T, u16>(len, unique_values, chunk_count);

    bencher
        .with_inputs(|| chunk.clone())
        .bench_local_values(|chunk| chunk.into_canonical().vortex_unwrap())
}
