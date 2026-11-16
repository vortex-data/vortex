// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use rand::RngExt;
use rand::SeedableRng;
use rand::prelude::StdRng;
use rand::seq::SliceRandom;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::assert_arrays_eq;
use vortex_array::builders::ArrayBuilder;
use vortex_array::builders::VarBinBuilder;
use vortex_array::builders::VarBinViewBuilder;
use vortex_array::builders::builder_with_capacity_in;
use vortex_array::dtype::DType;
use vortex_array::memory::MemorySession;
use vortex_array::optimizer::kernels::KernelSession;
use vortex_array::test_harness::mark_optimized;
use vortex_buffer::BitBuffer;
use vortex_error::VortexExpect;
use vortex_mask::Mask;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

const FILTERED_UTF8_ARGS: &[(usize, bool)] = &[
    (64, false),
    (6_553, false),
    (32_768, false),
    (62_259, false),
    (64, true),
    (6_553, true),
    (32_768, true),
    (62_259, true),
];

fn bench_filtered_utf8_append(
    bencher: Bencher,
    (selected, nullable): (usize, bool),
    compact: bool,
    contiguous: bool,
) {
    let len = 65_536;
    let values = if nullable {
        VarBinViewArray::from_iter_nullable_str(
            (0..len).map(|row| (row % 11 != 0).then(|| format!("non-inline string value {row}"))),
        )
    } else {
        VarBinViewArray::from_iter_str((0..len).map(|row| format!("non-inline string value {row}")))
    }
    .into_array();
    mark_optimized(&values, &SESSION).unwrap();
    let mut indices = (0..len).collect::<Vec<_>>();
    if contiguous {
        let start = (len - selected) / 2;
        indices = (start..start + selected).collect();
    } else {
        indices.shuffle(&mut StdRng::seed_from_u64(42));
        indices.truncate(selected);
    }
    let bits = BitBuffer::from_indices(len, indices);
    let allocator = SESSION.create_execution_ctx().allocator().clone();
    let make_builder = || {
        if compact {
            VarBinViewBuilder::with_compaction_in(
                values.dtype().clone(),
                selected,
                1.0,
                allocator.clone(),
            )
        } else {
            VarBinViewBuilder::with_capacity_in(values.dtype().clone(), selected, allocator.clone())
        }
    };
    let mut ctx = SESSION.create_execution_ctx();
    let filtered = values.filter(Mask::from_buffer(bits.clone())).unwrap();
    let expected = filtered
        .clone()
        .execute::<Canonical>(&mut ctx)
        .unwrap()
        .into_array();
    let mut checked = make_builder();
    filtered.append_to_builder(&mut checked, &mut ctx).unwrap();
    assert_arrays_eq!(checked.finish_into_varbinview(), expected, &mut ctx);

    bencher
        .with_inputs(|| {
            (
                values.filter(Mask::from_buffer(bits.clone())).unwrap(),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_values(|(filtered, mut ctx)| {
            let mut builder = make_builder();
            filtered.append_to_builder(&mut builder, &mut ctx).unwrap();
            builder.finish_into_varbinview()
        });
}

#[divan::bench(args = FILTERED_UTF8_ARGS)]
fn filtered_utf8_append_to_builder(bencher: Bencher, case: (usize, bool)) {
    bench_filtered_utf8_append(bencher, case, false, false);
}

#[divan::bench(args = FILTERED_UTF8_ARGS)]
fn filtered_utf8_compacting_append_to_builder(bencher: Bencher, case: (usize, bool)) {
    bench_filtered_utf8_append(bencher, case, true, false);
}

#[divan::bench(args = FILTERED_UTF8_ARGS)]
fn filtered_utf8_contiguous_append_to_builder(bencher: Bencher, case: (usize, bool)) {
    bench_filtered_utf8_append(bencher, case, false, true);
}

#[divan::bench(args = FILTERED_UTF8_ARGS)]
fn filtered_utf8_contiguous_compacting_append_to_builder(bencher: Bencher, case: (usize, bool)) {
    bench_filtered_utf8_append(bencher, case, true, true);
}

const BENCH_ARGS: &[(usize, usize)] = &[
    // length, chunk_count
    // Chunk counts sized to keep CodSpeed simulation under 1ms per benchmark.
    (10, 100),
    (100, 50),
    (1000, 10),
];

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    VortexSession::empty()
        .with::<KernelSession>()
        .with::<MemorySession>()
});

#[divan::bench(args = BENCH_ARGS)]
fn chunked_bool_canonical_into(bencher: Bencher, (len, chunk_count): (usize, usize)) {
    let chunk = make_bool_chunks(len, chunk_count);
    mark_optimized(&chunk, &SESSION).unwrap();

    bencher
        .with_inputs(|| (&chunk, SESSION.create_execution_ctx()))
        .bench_refs(|(chunk, ctx)| {
            let mut builder =
                builder_with_capacity_in(chunk.dtype(), len * chunk_count, ctx.allocator());
            chunk
                .append_to_builder(builder.as_mut(), ctx)
                .vortex_expect("append failed");
            builder.finish()
        })
}

// The largest nullable case varies on unchanged code in simulation. The cause is unknown.
#[divan::bench(args = if cfg!(codspeed) { &[(10, 100), (100, 50)] } else { BENCH_ARGS })]
fn chunked_opt_bool_canonical_into(bencher: Bencher, (len, chunk_count): (usize, usize)) {
    let chunk = make_opt_bool_chunks(len, chunk_count);
    mark_optimized(&chunk, &SESSION).unwrap();

    bencher
        .with_inputs(|| (&chunk, SESSION.create_execution_ctx()))
        .bench_refs(|(chunk, ctx)| {
            let mut builder =
                builder_with_capacity_in(chunk.dtype(), len * chunk_count, ctx.allocator());
            chunk
                .append_to_builder(builder.as_mut(), ctx)
                .vortex_expect("append failed");
            builder.finish()
        })
}

#[divan::bench(args = BENCH_ARGS)]
fn chunked_opt_bool_into_canonical(bencher: Bencher, (len, chunk_count): (usize, usize)) {
    let chunk = make_opt_bool_chunks(len, chunk_count);
    mark_optimized(&chunk, &SESSION).unwrap();

    bencher
        .with_inputs(|| (&chunk, SESSION.create_execution_ctx()))
        .bench_refs(|(chunk, ctx)| (**chunk).clone().execute::<Canonical>(ctx))
}

#[divan::bench(args = BENCH_ARGS)]
fn chunked_varbinview_canonical_into(bencher: Bencher, (len, chunk_count): (usize, usize)) {
    let chunks = make_string_chunks(false, len, chunk_count);
    mark_optimized(&chunks, &SESSION).unwrap();

    bencher
        .with_inputs(|| (&chunks, SESSION.create_execution_ctx()))
        .bench_refs(|(chunk, ctx)| {
            let mut builder = VarBinViewBuilder::with_capacity_in(
                DType::Utf8(chunk.dtype().nullability()),
                len * chunk_count,
                ctx.allocator().clone(),
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
    mark_optimized(&chunks, &SESSION).unwrap();

    bencher
        .with_inputs(|| (&chunks, SESSION.create_execution_ctx()))
        .bench_refs(|(chunk, ctx)| (**chunk).clone().execute::<Canonical>(ctx))
}

#[divan::bench(args = BENCH_ARGS)]
fn chunked_varbinview_opt_canonical_into(bencher: Bencher, (len, chunk_count): (usize, usize)) {
    let chunks = make_string_chunks(true, len, chunk_count);
    mark_optimized(&chunks, &SESSION).unwrap();

    bencher
        .with_inputs(|| (&chunks, SESSION.create_execution_ctx()))
        .bench_refs(|(chunk, ctx)| {
            let mut builder = VarBinViewBuilder::with_capacity_in(
                DType::Utf8(chunk.dtype().nullability()),
                len * chunk_count,
                ctx.allocator().clone(),
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
    mark_optimized(&chunks, &SESSION).unwrap();

    bencher
        .with_inputs(|| (&chunks, SESSION.create_execution_ctx()))
        .bench_refs(|(chunk, ctx)| (**chunk).clone().execute::<Canonical>(ctx))
}

// Fewer rows than BENCH_ARGS: decoding VarBin values is the most expensive work in this file.
const VARBIN_BENCH_ARGS: &[(usize, usize)] = &[
    // length, chunk_count
    (10, 100),
    (500, 2),
];

#[divan::bench(args = VARBIN_BENCH_ARGS)]
fn chunked_varbin_to_varbinview_builder(bencher: Bencher, (len, chunk_count): (usize, usize)) {
    let chunks = make_varbin_chunks(false, len, chunk_count);
    mark_optimized(&chunks, &SESSION).unwrap();

    bencher
        .with_inputs(|| (&chunks, SESSION.create_execution_ctx()))
        .bench_refs(|(chunk, ctx)| {
            let mut builder = VarBinViewBuilder::with_capacity_in(
                chunk.dtype().clone(),
                len * chunk_count,
                ctx.allocator().clone(),
            );
            chunk
                .append_to_builder(&mut builder, ctx)
                .vortex_expect("append failed");
            builder.finish()
        })
}

#[divan::bench(args = VARBIN_BENCH_ARGS)]
fn chunked_varbin_opt_to_varbinview_builder(bencher: Bencher, (len, chunk_count): (usize, usize)) {
    let chunks = make_varbin_chunks(true, len, chunk_count);
    mark_optimized(&chunks, &SESSION).unwrap();

    bencher
        .with_inputs(|| (&chunks, SESSION.create_execution_ctx()))
        .bench_refs(|(chunk, ctx)| {
            let mut builder = VarBinViewBuilder::with_capacity_in(
                chunk.dtype().clone(),
                len * chunk_count,
                ctx.allocator().clone(),
            );
            chunk
                .append_to_builder(&mut builder, ctx)
                .vortex_expect("append failed");
            builder.finish()
        })
}

#[divan::bench(args = VARBIN_BENCH_ARGS)]
fn chunked_varbin_into_canonical(bencher: Bencher, (len, chunk_count): (usize, usize)) {
    let chunks = make_varbin_chunks(false, len, chunk_count);
    mark_optimized(&chunks, &SESSION).unwrap();

    bencher
        .with_inputs(|| (&chunks, SESSION.create_execution_ctx()))
        .bench_refs(|(chunk, ctx)| (**chunk).clone().execute::<Canonical>(ctx))
}

#[divan::bench(args = BENCH_ARGS)]
fn chunked_constant_i32_append_to_builder(bencher: Bencher, (len, chunk_count): (usize, usize)) {
    let chunk = make_constant_i32_chunks(len, chunk_count);
    mark_optimized(&chunk, &SESSION).unwrap();

    bencher
        .with_inputs(|| (&chunk, SESSION.create_execution_ctx()))
        .bench_refs(|(chunk, ctx)| {
            let mut builder =
                builder_with_capacity_in(chunk.dtype(), len * chunk_count, ctx.allocator());
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
    mark_optimized(&chunk, &SESSION).unwrap();

    bencher
        .with_inputs(|| (&chunk, SESSION.create_execution_ctx()))
        .bench_refs(|(chunk, ctx)| {
            let mut builder =
                builder_with_capacity_in(chunk.dtype(), len * chunk_count, ctx.allocator());
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

fn make_varbin_chunks(nullable: bool, len: usize, chunk_count: usize) -> ArrayRef {
    let mut rng = StdRng::seed_from_u64(123);
    let dtype = DType::Utf8(nullable.into());

    (0..chunk_count)
        .map(|_| {
            let mut builder = VarBinBuilder::<i32>::with_capacity_in(
                dtype.clone(),
                len,
                vortex_buffer::BufferAllocatorRef::static_ref(),
            );
            (0..len).for_each(|_| {
                if nullable && rng.random_bool(0.2) {
                    builder.push_null()
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

fn make_string_chunks(nullable: bool, len: usize, chunk_count: usize) -> ArrayRef {
    let mut rng = StdRng::seed_from_u64(123);

    (0..chunk_count)
        .map(|_| {
            let mut builder = VarBinViewBuilder::with_capacity_in(
                DType::Utf8(nullable.into()),
                len,
                vortex_buffer::BufferAllocatorRef::statically_allocated(),
            );
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
