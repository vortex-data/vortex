// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use divan::Bencher;
use rand::SeedableRng;
use rand::prelude::StdRng;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ChunkedArray;
use vortex_array::builders::ArrayBuilder;
use vortex_array::builders::PrimitiveBuilder;
use vortex_array::session::ArraySession;
use vortex_error::VortexExpect;
use vortex_fastlanes::bitpack_compress::test_harness::make_array;
use vortex_session::VortexSession;

fn main() {
    divan::main();
}

const BENCH_ARGS: &[(usize, usize, f64)] = &[
    // chunk_len, chunk_count, fraction_patched
    (10000, 1, 0.10),
    (10000, 1, 0.01),
    (10000, 1, 0.00),
    (10000, 10, 0.10),
    (10000, 10, 0.01),
    (10000, 10, 0.00),
    (10000, 100, 0.10),
    (10000, 100, 0.01),
    (10000, 100, 0.00),
    (10000, 1000, 0.00),
];

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

#[cfg(not(codspeed))]
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

    bencher
        .with_inputs(|| {
            (
                ChunkedArray::from_iter(chunks.clone()).into_array(),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_refs(|(chunked, ctx)| chunked.clone().execute::<Canonical>(ctx));
}

#[cfg(not(codspeed))]
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

    bencher
        .with_inputs(|| {
            let chunked = ChunkedArray::from_iter(chunks.clone()).into_array();
            let primitive_builder = PrimitiveBuilder::<i32>::with_capacity(
                chunked.dtype().nullability(),
                chunk_len * chunk_count,
            );
            (chunked, primitive_builder, SESSION.create_execution_ctx())
        })
        .bench_refs(|(chunked, primitive_builder, ctx)| {
            chunked
                .append_to_builder(primitive_builder, ctx)
                .vortex_expect("append failed");
            primitive_builder.finish()
        });
}

const NULLABLE_BENCH_ARGS: &[(usize, usize, f64)] = &[
    // chunk_len, chunk_count, fraction_patched
    (10000, 1, 0.10),
    (10000, 1, 0.00),
    (10000, 10, 0.10),
    (10000, 10, 0.00),
    (10000, 100, 0.10),
    (10000, 100, 0.00),
];

#[cfg(not(codspeed))]
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

    bencher
        .with_inputs(|| {
            (
                ChunkedArray::from_iter(chunks.clone()).into_array(),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_values(|(chunked, mut ctx)| chunked.execute::<Canonical>(&mut ctx));
}

#[cfg(not(codspeed))]
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

    bencher
        .with_inputs(|| {
            let chunked = ChunkedArray::from_iter(chunks.clone()).into_array();
            let primitive_builder = PrimitiveBuilder::<i32>::with_capacity(
                chunked.dtype().nullability(),
                chunk_len * chunk_count,
            );
            (chunked, primitive_builder, SESSION.create_execution_ctx())
        })
        .bench_refs(|(chunked, primitive_builder, ctx)| {
            chunked
                .append_to_builder(primitive_builder, ctx)
                .vortex_expect("append failed");
            primitive_builder.finish()
        });
}
