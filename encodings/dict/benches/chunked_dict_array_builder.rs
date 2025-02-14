use divan::Bencher;
use rand::distributions::{Distribution, Standard};
use rand::prelude::StdRng;
use rand::SeedableRng;
use vortex_array::array::ChunkedArray;
use vortex_array::builders::builder_with_capacity;
use vortex_array::{Array, IntoArray, IntoCanonical};
use vortex_dict::test::gen_primitive_dict;
use vortex_dtype::NativePType;
use vortex_error::VortexUnwrap;

fn main() {
    divan::main();
}

fn make_dict_primitive_chunks<T: NativePType, O: NativePType>(
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
            gen_primitive_dict::<T, O>(&mut rng, len, unique_values)
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
    // Too slow for CI
    // (32_000, 10, 100),
    // (32_000, 100, 100),
    // (32_000, 1000, 100),
    // (32_000, 10, 1000),
    // (32_000, 100, 1000),
    // (32_000, 1000, 1000),
];

#[divan::bench(types = [u32, u64, f32, f64], args=BENCH_ARGS)]
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

#[divan::bench(types = [u32, u64, f32, f64], args=BENCH_ARGS)]
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

fn generate_test_data(
    rng: &mut StdRng,
    string_count: usize,
    avg_len: usize,
    unique_chars: u8,
) -> Array {
    let mut strings = Vec::with_capacity(string_count);

    for _ in 0..string_count {
        // Generate a random string with length around `avg_len`. The number of possible
        // characters within the random string is defined by `unique_chars`.
        let len = avg_len * rng.gen_range(50..=150) / 100;
        strings.push(Some(
            (0..len)
                .map(|_| rng.gen_range(b'a'..(b'a' + unique_chars)) as char)
                .collect::<String>()
                .into_bytes(),
        ));
    }

    let varbin = VarBinArray::from_iter(
        strings
            .into_iter()
            .map(|opt_s| opt_s.map(Vec::into_boxed_slice)),
        DType::Binary(Nullability::NonNullable),
    );
    let compressor = fsst_train_compressor(&varbin).vortex_unwrap();

    fsst_compress(&varbin, &compressor)
        .vortex_unwrap()
        .into_array()
}

fn make_dict_fsst_chunks<O: NativePType>(
    len: usize,
    unique_values: usize,
    chunk_count: usize,
) -> Array {
    let mut rng = StdRng::seed_from_u64(0);

    (0..chunk_count)
        .map(|_| {
            let values = generate_test_data(&mut rng, len, 20, 10);
            // let values = generate_test_data(&mut rng, 2, 4, 10);
            let codes = (0..len)
                .map(|_| O::from(rng.gen_range(0..unique_values)).vortex_expect("valid value"))
                .collect::<PrimitiveArray>();

            DictArray::try_new(codes.into_array(), values)
                .vortex_unwrap()
                .into_array()
        })
        .collect::<ChunkedArray>()
        .into_array()
}

#[divan::bench(args=BENCH_ARGS)]
fn chunked_dict_fsst_canonical_into(
    bencher: Bencher,
    (len, unique_values, chunk_count): (usize, usize, usize),
) {
    let chunk = make_dict_fsst_chunks::<u16>(len, unique_values, chunk_count);

    bencher
        .with_inputs(|| chunk.clone())
        .bench_local_values(|chunk| {
            let mut builder = builder_with_capacity(chunk.dtype(), len * chunk_count);
            chunk.canonicalize_into(builder.as_mut()).vortex_unwrap();
            builder.finish().vortex_unwrap()
        })
}

#[divan::bench(args=BENCH_ARGS)]
fn chunked_dict_fsst_into_canonical(
    bencher: Bencher,
    (len, unique_values, chunk_count): (usize, usize, usize),
) {
    let chunk = make_dict_fsst_chunks::<u16>(len, unique_values, chunk_count);

    bencher
        .with_inputs(|| chunk.clone())
        .bench_local_values(|chunk| chunk.into_canonical().vortex_unwrap())
}
