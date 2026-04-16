// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use divan::Bencher;
use rand::RngExt;
use rand::SeedableRng;
use rand::prelude::StdRng;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::builders::ArrayBuilder;
use vortex_array::builders::VarBinViewBuilder;
use vortex_array::builders::builder_with_capacity;
use vortex_array::dtype::DType;
use vortex_array::session::ArraySession;
use vortex_error::VortexExpect;
use vortex_session::VortexSession;

fn main() {
    divan::main();
}

const BENCH_ARGS: &[(usize, usize)] = &[
    // length, chunk_count
    (10, 1000),
    (100, 100),
    (1000, 10),
];

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

#[divan::bench(args = BENCH_ARGS)]
fn chunked_bool_canonical_into(bencher: Bencher, (len, chunk_count): (usize, usize)) {
    let chunk = make_bool_chunks(len, chunk_count);

    bencher
        .with_inputs(|| (&chunk, SESSION.create_execution_ctx()))
        .bench_refs(|(chunk, ctx)| {
            let mut builder = builder_with_capacity(chunk.dtype(), len * chunk_count);
            chunk
                .append_to_builder(builder.as_mut(), ctx)
                .vortex_expect("append failed");
            builder.finish()
        })
}

#[divan::bench(args = BENCH_ARGS)]
fn chunked_opt_bool_canonical_into(bencher: Bencher, (len, chunk_count): (usize, usize)) {
    let chunk = make_opt_bool_chunks(len, chunk_count);

    bencher
        .with_inputs(|| (&chunk, SESSION.create_execution_ctx()))
        .bench_refs(|(chunk, ctx)| {
            let mut builder = builder_with_capacity(chunk.dtype(), len * chunk_count);
            chunk
                .append_to_builder(builder.as_mut(), ctx)
                .vortex_expect("append failed");
            builder.finish()
        })
}

#[divan::bench(args = BENCH_ARGS)]
fn chunked_opt_bool_into_canonical(bencher: Bencher, (len, chunk_count): (usize, usize)) {
    let chunk = make_opt_bool_chunks(len, chunk_count);

    bencher
        .with_inputs(|| &chunk)
        .bench_refs(|chunk| chunk.to_canonical())
}

#[divan::bench(args = BENCH_ARGS)]
fn chunked_varbinview_canonical_into(bencher: Bencher, (len, chunk_count): (usize, usize)) {
    let chunks = make_string_chunks(false, len, chunk_count);

    bencher
        .with_inputs(|| (&chunks, SESSION.create_execution_ctx()))
        .bench_refs(|(chunk, ctx)| {
            let mut builder = VarBinViewBuilder::with_capacity(
                DType::Utf8(chunk.dtype().nullability()),
                len * chunk_count,
            );
            chunk
                .append_to_builder(&mut builder, ctx)
                .vortex_expect("append failed");
            builder.finish()
        })
}

#[divan::bench(args = BENCH_ARGS)]
fn chunked_varbinview_into_canonical(bencher: Bencher, (len, chunk_count): (usize, usize)) {
    let chunks = make_string_chunks(false, len, chunk_count);

    bencher
        .with_inputs(|| &chunks)
        .bench_refs(|chunk| chunk.to_canonical())
}

#[divan::bench(args = BENCH_ARGS)]
fn chunked_varbinview_opt_canonical_into(bencher: Bencher, (len, chunk_count): (usize, usize)) {
    let chunks = make_string_chunks(true, len, chunk_count);

    bencher
        .with_inputs(|| (&chunks, SESSION.create_execution_ctx()))
        .bench_refs(|(chunk, ctx)| {
            let mut builder = VarBinViewBuilder::with_capacity(
                DType::Utf8(chunk.dtype().nullability()),
                len * chunk_count,
            );
            chunk
                .append_to_builder(&mut builder, ctx)
                .vortex_expect("append failed");
            builder.finish()
        })
}

#[divan::bench(args = BENCH_ARGS)]
fn chunked_varbinview_opt_into_canonical(bencher: Bencher, (len, chunk_count): (usize, usize)) {
    let chunks = make_string_chunks(true, len, chunk_count);

    bencher
        .with_inputs(|| &chunks)
        .bench_refs(|chunk| chunk.to_canonical())
}

#[divan::bench(args = BENCH_ARGS)]
fn chunked_constant_i32_append_to_builder(bencher: Bencher, (len, chunk_count): (usize, usize)) {
    let chunk = make_constant_i32_chunks(len, chunk_count);

    bencher
        .with_inputs(|| (&chunk, SESSION.create_execution_ctx()))
        .bench_refs(|(chunk, ctx)| {
            let mut builder = builder_with_capacity(chunk.dtype(), len * chunk_count);
            chunk
                .append_to_builder(builder.as_mut(), ctx)
                .vortex_expect("append failed");
            builder.finish()
        })
}

const CONSTANT_UTF8_BENCH_ARGS: &[(&str, usize, usize)] = &[
    // value, length, chunk_count
    ("hi", 1000, 10),            // inline (≤12 bytes)
    ("hello world!!", 1000, 10), // non-inline (>12 bytes)
];

#[divan::bench(args = CONSTANT_UTF8_BENCH_ARGS)]
fn chunked_constant_utf8_append_to_builder(
    bencher: Bencher,
    (value, len, chunk_count): (&str, usize, usize),
) {
    let chunk = make_constant_utf8_chunks(value, len, chunk_count);

    bencher
        .with_inputs(|| (&chunk, SESSION.create_execution_ctx()))
        .bench_refs(|(chunk, ctx)| {
            let mut builder = builder_with_capacity(chunk.dtype(), len * chunk_count);
            chunk
                .append_to_builder(builder.as_mut(), ctx)
                .vortex_expect("append failed");
            builder.finish()
        })
}

fn make_constant_utf8_chunks(value: &str, len: usize, chunk_count: usize) -> ArrayRef {
    use vortex_array::dtype::Nullability;
    use vortex_array::scalar::Scalar;

    (0..chunk_count)
        .map(|_| {
            ConstantArray::new(Scalar::utf8(value, Nullability::NonNullable), len).into_array()
        })
        .collect::<ChunkedArray>()
        .into_array()
}

fn make_constant_i32_chunks(len: usize, chunk_count: usize) -> ArrayRef {
    // Each chunk is a ConstantArray of i32; dtype is I32/NonNullable via From<i32> for Scalar.
    (0..chunk_count)
        .map(|_| ConstantArray::new(42i32, len).into_array())
        .collect::<ChunkedArray>()
        .into_array()
}

fn make_opt_bool_chunks(len: usize, chunk_count: usize) -> ArrayRef {
    let mut rng = StdRng::seed_from_u64(0);

    const SPAN_LEN: usize = 10;
    assert!(len.is_multiple_of(SPAN_LEN));

    (0..chunk_count)
        .map(|_| {
            BoolArray::from_iter(
                (0..len / SPAN_LEN)
                    .flat_map(|_| match rng.random_range::<u8, _>(0..=2) {
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
        .map(|_| BoolArray::from_iter((0..len).map(|_| rng.random_bool(0.5))).into_array())
        .collect::<ChunkedArray>()
        .into_array()
}

fn make_string_chunks(nullable: bool, len: usize, chunk_count: usize) -> ArrayRef {
    let mut rng = StdRng::seed_from_u64(123);

    (0..chunk_count)
        .map(|_| {
            let mut builder = VarBinViewBuilder::with_capacity(DType::Utf8(nullable.into()), len);
            (0..len).for_each(|_| {
                if nullable && rng.random_bool(0.2) {
                    builder.append_null()
                } else {
                    builder.append_value(
                        (0..rng.random_range(0..=20))
                            .map(|_| rng.random_range(b'a'..=b'z'))
                            .collect::<Vec<u8>>(),
                    )
                }
            });
            builder.finish()
        })
        .collect::<ChunkedArray>()
        .into_array()
}
