#![allow(clippy::unwrap_used)]

use divan::Bencher;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use vortex_array::arrays::{ChunkedArray, ConstantArray, VarBinArray};
use vortex_array::builders::{ArrayBuilder, VarBinViewBuilder};
use vortex_array::compute::{compare, Operator};
use vortex_array::{IntoArray, IntoCanonical};
use vortex_dtype::{DType, Nullability};
use vortex_fsst::{fsst_compress, fsst_train_compressor};
use vortex_scalar::Scalar;

fn main() {
    divan::main();
}

// [(string_count, avg_len, unique_chars)]
const BENCH_ARGS: &[(usize, usize, u8)] = &[
    (1_000, 4, 4),
    (1_000, 4, 8),
    (1_000, 16, 4),
    (1_000, 16, 8),
    (1_000, 64, 4),
    (1_000, 64, 8),
    (10_000, 4, 4),
    (10_000, 4, 8),
    (10_000, 16, 4),
    (10_000, 16, 8),
    (10_000, 64, 4),
    (10_000, 64, 8),
];

#[divan::bench(args = BENCH_ARGS)]
fn compress_fsst(bencher: Bencher, args: (usize, usize, u8)) {
    let (string_count, avg_len, unique_chars) = args;
    let array = generate_test_data(string_count, avg_len, unique_chars);
    let compressor = fsst_train_compressor(&array).unwrap();
    bencher.bench_local(|| fsst_compress(&array, &compressor).unwrap())
}

#[divan::bench(args = BENCH_ARGS)]
fn decompress_fsst(bencher: Bencher, args: (usize, usize, u8)) {
    let (string_count, avg_len, unique_chars) = args;
    let array = generate_test_data(string_count, avg_len, unique_chars);
    let compressor = fsst_train_compressor(&array).unwrap();
    let encoded = fsst_compress(&array, &compressor).unwrap();

    bencher
        .with_inputs(|| encoded.clone())
        .bench_local_values(|encoded| encoded.into_canonical().unwrap())
}

#[divan::bench(args = BENCH_ARGS)]
fn train_compressor(bencher: Bencher, args: (usize, usize, u8)) {
    let (string_count, avg_len, unique_chars) = args;
    let array = generate_test_data(string_count, avg_len, unique_chars);
    bencher.bench_local(|| fsst_train_compressor(&array).unwrap())
}

#[divan::bench(args = BENCH_ARGS)]
fn pushdown_compare(bencher: Bencher, args: (usize, usize, u8)) {
    let (string_count, avg_len, unique_chars) = args;
    let array = generate_test_data(string_count, avg_len, unique_chars);

    let constant = ConstantArray::new(Scalar::from(&b"const"[..]), array.len());
    bencher.bench_local(|| compare(&array, &constant, Operator::Eq).unwrap());
}

#[divan::bench(args = BENCH_ARGS)]
fn canonicalize_compare(bencher: Bencher, args: (usize, usize, u8)) {
    let (string_count, avg_len, unique_chars) = args;
    let array = generate_test_data(string_count, avg_len, unique_chars);

    let constant = ConstantArray::new(Scalar::from(&b"const"[..]), array.len());
    bencher
        .with_inputs(|| array.clone())
        .bench_local_values(|array| {
            compare(array.into_canonical().unwrap(), &constant, Operator::Eq).unwrap()
        });
}

// [(chunk_size, string_count, avg_len, unique_chars)]
const CHUNKED_BENCH_ARGS: &[(usize, usize, usize, u8)] = &[
    (1000, 100, 16, 4),
    (1000, 100, 16, 16),
    (1000, 100, 16, 64),
    (1000, 50, 8, 4),
    (1000, 50, 8, 16),
    (1000, 50, 8, 64),
    (10, 10_000, 4, 4),
    (10, 10_000, 16, 4),
    (10, 10_000, 64, 4),
];

#[divan::bench(args = CHUNKED_BENCH_ARGS)]
fn chunked_canonicalize_into(
    bencher: Bencher,
    (chunk_size, string_count, avg_len, unique_chars): (usize, usize, usize, u8),
) {
    let array = generate_chunked_test_data(chunk_size, string_count, avg_len, unique_chars);

    bencher
        .with_inputs(|| array.clone())
        .bench_local_values(|array| {
            let mut builder = VarBinViewBuilder::with_capacity(
                DType::Binary(Nullability::NonNullable),
                array.len(),
            );
            array.canonicalize_into(&mut builder).unwrap();
            builder.finish()
        });
}

#[divan::bench(args = CHUNKED_BENCH_ARGS)]
fn chunked_into_canonical(
    bencher: Bencher,
    (chunk_size, string_count, avg_len, unique_chars): (usize, usize, usize, u8),
) {
    let array = generate_chunked_test_data(chunk_size, string_count, avg_len, unique_chars);

    bencher
        .with_inputs(|| array.clone())
        .bench_local_values(|array| array.into_canonical().unwrap());
}

// Helper function to generate random string data.
fn generate_test_data(string_count: usize, avg_len: usize, unique_chars: u8) -> VarBinArray {
    let mut rng = StdRng::seed_from_u64(0);
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

    VarBinArray::from_iter(
        strings
            .into_iter()
            .map(|opt_s| opt_s.map(Vec::into_boxed_slice)),
        DType::Binary(Nullability::NonNullable),
    )
}

fn generate_chunked_test_data(
    chunk_size: usize,
    string_count: usize,
    avg_len: usize,
    unique_chars: u8,
) -> ChunkedArray {
    (0..chunk_size)
        .map(|_| {
            let array = generate_test_data(string_count, avg_len, unique_chars).into_array();
            let compressor = fsst_train_compressor(&array).unwrap();
            fsst_compress(&array, &compressor).unwrap().into_array()
        })
        .collect::<ChunkedArray>()
}
