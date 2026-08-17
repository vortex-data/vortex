// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks scan split collection (`SplitBy::Layout`) over written files.
//!
//! The default write strategy produces `struct -> zoned -> chunked -> flat` per column, so
//! split collection walks every chunk of every column. `cold` builds a fresh reader tree per
//! iteration (as a first scan over a file would); `warm` reuses the reader tree so lazily
//! cached child readers persist across iterations. `cold_misaligned` scans a file whose
//! columns share no interior chunk boundaries, so the split set cannot be collapsed by run
//! deduplication.

#![expect(clippy::unwrap_used)]

use std::sync::Arc;
use std::sync::LazyLock;

use divan::Bencher;
use vortex_array::IntoArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::Field;
use vortex_array::dtype::FieldMask;
use vortex_array::dtype::session::DTypeSessionExt;
use vortex_array::session::ArraySessionExt;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBufferMut;
use vortex_edition::ComponentKind;
use vortex_edition::Edition;
use vortex_edition::EditionId;
use vortex_edition::EditionInclusion;
use vortex_edition::EditionSessionExt;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::VortexFile;
use vortex_file::WriteOptionsSessionExt;
use vortex_io::session::RuntimeSession;
use vortex_io::session::RuntimeSessionExt;
use vortex_layout::layouts::chunked::writer::ChunkedLayoutStrategy;
use vortex_layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex_layout::layouts::repartition::RepartitionStrategy;
use vortex_layout::layouts::repartition::RepartitionWriterOptions;
use vortex_layout::scan::split_by::SplitBy;
use vortex_layout::session::LayoutSession;
use vortex_layout::session::LayoutSessionExt;
use vortex_session::VortexSession;
use vortex_utils::aliases::hash_map::HashMap;

fn main() {
    divan::main();
}

const ROWS_PER_CHUNK: usize = 1024;

/// (columns, chunks) configurations.
const CONFIGS: &[(usize, usize)] = &[(64, 256)];

static RUNTIME: LazyLock<tokio::runtime::Runtime> = LazyLock::new(|| {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
});

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let _guard = RUNTIME.enter();
    let session = vortex_array::array_session()
        .with::<LayoutSession>()
        .with::<RuntimeSession>()
        .with_tokio();
    vortex_file::register_default_encodings(&session);
    enable_all_registered_array_encodings(&session);
    session
});

const BENCH_EDITION: EditionId = EditionId::new("bench", 2026, 8, 0);

fn enable_all_registered_array_encodings(session: &VortexSession) {
    let editions = session.editions();
    editions
        .declare_edition(Edition {
            id: BENCH_EDITION,
            min_vortex_version: None,
        })
        .unwrap();
    let component_ids = [
        (
            ComponentKind::Array,
            session
                .arrays()
                .registry()
                .read(|map| map.keys().copied().collect::<Vec<_>>()),
        ),
        (
            ComponentKind::Layout,
            session
                .layouts()
                .registry()
                .read(|map| map.keys().copied().collect::<Vec<_>>()),
        ),
        (
            ComponentKind::DType,
            session
                .dtypes()
                .registry()
                .read(|map| map.keys().copied().collect::<Vec<_>>()),
        ),
    ];
    for (kind, ids) in component_ids {
        for id in ids {
            editions
                .declare_inclusion(EditionInclusion::new(kind, &id, BENCH_EDITION))
                .unwrap();
        }
    }
    for id in [
        "vortex.bounded_max",
        "vortex.bounded_min",
        "vortex.max",
        "vortex.min",
        "vortex.nan_count",
        "vortex.null_count",
    ] {
        editions
            .declare_inclusion(EditionInclusion::new(
                ComponentKind::Aggregate,
                id,
                BENCH_EDITION,
            ))
            .unwrap();
    }
    session.enable_edition(BENCH_EDITION).unwrap();
}

fn make_file(columns: usize, chunks: usize) -> VortexFile {
    let field_names = (0..columns).map(|c| format!("col_{c}")).collect::<Vec<_>>();
    let struct_chunks = (0..chunks)
        .map(|chunk| {
            let fields = field_names
                .iter()
                .map(|name| {
                    let start = (chunk * ROWS_PER_CHUNK) as i64;
                    let values =
                        Buffer::from_iter(start..start + ROWS_PER_CHUNK as i64).into_array();
                    (name.as_str(), values)
                })
                .collect::<Vec<_>>();
            StructArray::from_fields(&fields).unwrap().into_array()
        })
        .collect::<Vec<_>>();
    let array = ChunkedArray::from_iter(struct_chunks).into_array();

    let strategy = vortex_file::WriteStrategyBuilder::default()
        .with_row_block_size(ROWS_PER_CHUNK)
        .with_data_block_target_bytes(None)
        .build();

    let mut buf = ByteBufferMut::empty();
    RUNTIME
        .block_on(
            SESSION
                .write_options()
                .with_strategy(strategy)
                .write(&mut buf, array.to_array_stream()),
        )
        .unwrap();

    SESSION.open_options().open_buffer(buf).unwrap()
}

static FILES: LazyLock<HashMap<(usize, usize), VortexFile>> = LazyLock::new(|| {
    CONFIGS
        .iter()
        .map(|&(columns, chunks)| ((columns, chunks), make_file(columns, chunks)))
        .collect()
});

/// (columns, average chunks per column) for the misaligned files. A single column cannot be
/// misaligned, so only multi-column configs are used.
const MISALIGNED_CONFIGS: &[(usize, usize)] = &[(64, 256)];

/// Per-column repartition block length: all distinct, so no two columns share interior chunk
/// boundaries. Mirrors real files where byte-size coalescing gives each column its own chunking.
fn misaligned_block_len(column: usize) -> usize {
    384 + column * 8
}

/// Like [`make_file`], but each column is chunked at a different row granularity so chunk
/// boundaries never align across columns: run deduplication in split collection cannot collapse
/// them, exercising the sort fallback.
fn make_misaligned_file(columns: usize, chunks: usize) -> VortexFile {
    let mean_block_len = (0..columns).map(misaligned_block_len).sum::<usize>() / columns.max(1);
    let rows = chunks * mean_block_len;

    let fields = (0..columns)
        .map(|c| {
            let values = Buffer::from_iter(0..rows as i64).into_array();
            (format!("col_{c}"), values)
        })
        .collect::<Vec<_>>();
    let array = StructArray::from_fields(
        &fields
            .iter()
            .map(|(name, values)| (name.as_str(), values.clone()))
            .collect::<Vec<_>>(),
    )
    .unwrap()
    .into_array();

    let mut strategy = vortex_file::WriteStrategyBuilder::default();
    for (c, (name, _)) in fields.iter().enumerate() {
        let field_strategy = RepartitionStrategy::new(
            ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
            RepartitionWriterOptions {
                block_size_minimum: 0,
                block_len_multiple: misaligned_block_len(c),
                block_size_target: None,
                canonicalize: false,
            },
        );
        strategy = strategy.with_field_writer(Field::from(name.as_str()), Arc::new(field_strategy));
    }

    let mut buf = ByteBufferMut::empty();
    RUNTIME
        .block_on(
            SESSION
                .write_options()
                .with_strategy(strategy.build())
                .write(&mut buf, array.to_array_stream()),
        )
        .unwrap();

    SESSION.open_options().open_buffer(buf).unwrap()
}

static MISALIGNED_FILES: LazyLock<HashMap<(usize, usize), VortexFile>> = LazyLock::new(|| {
    MISALIGNED_CONFIGS
        .iter()
        .map(|&(columns, chunks)| ((columns, chunks), make_misaligned_file(columns, chunks)))
        .collect()
});

fn collect_splits(file: &VortexFile) -> Vec<u64> {
    let reader = file.layout_reader().unwrap();
    SplitBy::Layout
        .splits(reader.as_ref(), &(0..file.row_count()), &[FieldMask::All])
        .unwrap()
}

/// Builds a fresh reader tree per iteration, so split collection pays child reader
/// construction for every chunk of every column, like the first scan over a file.
#[divan::bench(args = CONFIGS)]
fn cold(bencher: Bencher, config: &(usize, usize)) {
    let file = &FILES[config];
    // Sanity-check the written chunk structure once per config.
    let n_splits = collect_splits(file).len();
    assert!(
        n_splits > config.1 / 2,
        "expected roughly one split per chunk, got {n_splits} splits for {} chunks",
        config.1
    );

    bencher.bench(|| collect_splits(file));
}

/// Reuses the reader tree across iterations, so lazily constructed child readers are cached
/// and split collection measures only the layout walk.
#[divan::bench(args = CONFIGS)]
fn warm(bencher: Bencher, config: &(usize, usize)) {
    let file = &FILES[config];
    let reader = file.layout_reader().unwrap();
    let row_count = file.row_count();

    bencher.bench(|| {
        SplitBy::Layout
            .splits(reader.as_ref(), &(0..row_count), &[FieldMask::All])
            .unwrap()
    });
}

/// Like `cold`, but over a file whose columns share no interior chunk boundaries, so the split
/// set cannot be collapsed by run deduplication and must be sorted.
#[divan::bench(args = MISALIGNED_CONFIGS)]
fn cold_misaligned(bencher: Bencher, config: &(usize, usize)) {
    let file = &MISALIGNED_FILES[config];
    // Sanity-check the misalignment once per config: boundaries should be mostly distinct
    // across columns, i.e. roughly columns × chunks in total.
    let n_splits = collect_splits(file).len();
    assert!(
        n_splits > config.0 * config.1 / 2,
        "expected mostly-distinct boundaries, got {n_splits} splits for {} columns x {} chunks",
        config.0,
        config.1,
    );

    bencher.bench(|| collect_splits(file));
}
