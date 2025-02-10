#![allow(clippy::unwrap_used)]

use divan::Bencher;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use vortex_array::array::{ConstantArray, VarBinArray};
use vortex_array::compute::{compare, Operator};
use vortex_array::IntoCanonical;
use vortex_dtype::{DType, Nullability};
use vortex_fsst::{fsst_compress, fsst_train_compressor};
use vortex_scalar::Scalar;

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

fn main() {
    divan::main();
}

// [(string_count, avg_len, unique_chars)]
const BENCH_ARGS: &[(usize, usize, u8)] = &[
    (10_000, 4, 4),
    (10_000, 16, 4),
    (10_000, 64, 4),
    (100_000, 4, 4),
    (100_000, 16, 4),
    (100_000, 64, 4),
    (10_000, 4, 8),
    (10_000, 16, 8),
    (10_000, 64, 8),
    (100_000, 4, 8),
    (100_000, 16, 8),
    (100_000, 64, 8),
];
