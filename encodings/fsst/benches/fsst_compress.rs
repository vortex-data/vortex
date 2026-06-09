// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::RecursiveCanonical;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::builders::ArrayBuilder;
use vortex_array::builders::VarBinViewBuilder;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::session::ArraySession;
use vortex_fsst::fsst_compress;
use vortex_fsst::fsst_train_compressor;
use vortex_session::VortexSession;

fn main() {
    divan::main();
}

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

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
fn compress_fsst(bencher: Bencher, (string_count, avg_len, unique_chars): (usize, usize, u8)) {
    let array = generate_test_data(string_count, avg_len, unique_chars);
    let compressor = fsst_train_compressor(&array);
    bencher
        .with_inputs(|| (&array, &compressor, SESSION.create_execution_ctx()))
        .bench_refs(|(array, compressor, ctx)| {
            fsst_compress(*array, array.len(), array.dtype(), compressor, ctx)
        })
}

#[divan::bench(args = BENCH_ARGS)]
fn decompress_fsst(bencher: Bencher, (string_count, avg_len, unique_chars): (usize, usize, u8)) {
    let array = generate_test_data(string_count, avg_len, unique_chars);
    let compressor = fsst_train_compressor(&array);
    let len = array.len();
    let dtype = array.dtype().clone();
    let encoded = fsst_compress(
        array,
        len,
        &dtype,
        &compressor,
        &mut SESSION.create_execution_ctx(),
    );

    bencher
        .with_inputs(|| (&encoded, SESSION.create_execution_ctx()))
        .bench_refs(|(encoded, ctx)| (**encoded).clone().into_array().execute::<Canonical>(ctx))
}

#[divan::bench(args = BENCH_ARGS)]
fn train_compressor(bencher: Bencher, (string_count, avg_len, unique_chars): (usize, usize, u8)) {
    let array = generate_test_data(string_count, avg_len, unique_chars);
    bencher
        .with_inputs(|| &array)
        .bench_refs(|array| fsst_train_compressor(array))
}

#[divan::bench(args = BENCH_ARGS)]
fn pushdown_compare(bencher: Bencher, (string_count, avg_len, unique_chars): (usize, usize, u8)) {
    let array = generate_test_data(string_count, avg_len, unique_chars);
    let compressor = fsst_train_compressor(&array);
    let fsst_array = fsst_compress(
        &array,
        array.len(),
        array.dtype(),
        &compressor,
        &mut SESSION.create_execution_ctx(),
    );
    let constant = ConstantArray::new(Scalar::from(&b"const"[..]), array.len());

    bencher
        .with_inputs(|| (&fsst_array, &constant, SESSION.create_execution_ctx()))
        .bench_refs(|(fsst_array, constant, ctx)| {
            fsst_array
                .clone()
                .into_array()
                .binary(constant.clone().into_array(), Operator::Eq)
                .unwrap()
                .execute::<RecursiveCanonical>(ctx)
                .unwrap();
        })
}

#[divan::bench(args = BENCH_ARGS)]
fn canonicalize_compare(
    bencher: Bencher,
    (string_count, avg_len, unique_chars): (usize, usize, u8),
) {
    let array = generate_test_data(string_count, avg_len, unique_chars);
    let compressor = fsst_train_compressor(&array);
    let fsst_array = fsst_compress(
        &array,
        array.len(),
        array.dtype(),
        &compressor,
        &mut SESSION.create_execution_ctx(),
    );
    let constant = ConstantArray::new(Scalar::from(&b"const"[..]), array.len());

    bencher
        .with_inputs(|| (&fsst_array, &constant, SESSION.create_execution_ctx()))
        .bench_refs(|(fsst_array, constant, ctx)| {
            (*fsst_array)
                .clone()
                .into_array()
                .execute::<Canonical>(ctx)
                .unwrap()
                .into_array()
                .binary(constant.clone().into_array(), Operator::Eq)
                .unwrap()
                .execute::<RecursiveCanonical>(ctx)
                .unwrap();
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
        .with_inputs(|| (&array, SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| {
            let mut builder = VarBinViewBuilder::with_capacity(
                DType::Binary(Nullability::NonNullable),
                array.len(),
            );
            array.append_to_builder(&mut builder, ctx).unwrap();
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
        .with_inputs(|| (&array, SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| (**array).clone().into_array().execute::<Canonical>(ctx));
}

/// Helper function to generate random string data.
fn generate_test_data(string_count: usize, avg_len: usize, unique_chars: u8) -> VarBinArray {
    let mut rng = StdRng::seed_from_u64(0);
    let mut strings = Vec::with_capacity(string_count);

    for _ in 0..string_count {
        // Generate a random string with length around `avg_len`. The number of possible
        // characters within the random string is defined by `unique_chars`.
        let len = avg_len * rng.random_range(50..=150) / 100;
        strings.push(Some(
            (0..len)
                .map(|_| rng.random_range(b'a'..(b'a' + unique_chars)) as char)
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
    let mut ctx = SESSION.create_execution_ctx();
    (0..chunk_size)
        .map(|_| {
            let array = generate_test_data(string_count, avg_len, unique_chars);
            let compressor = fsst_train_compressor(&array);
            let len = array.len();
            let dtype = array.dtype().clone();
            fsst_compress(array, len, &dtype, &compressor, &mut ctx).into_array()
        })
        .collect::<ChunkedArray>()
}
