// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Single-threaded file writes with the default write strategy.
//!
//! Each column is compressed in chunks of about one megabyte.
//!
//! The datasets:
//!
//! - `urls`: URL-like strings with 100,000 distinct values, about 47 bytes each.
//! - `drifting`: integers and strings whose best scheme changes halfway through the column.
//! - `table`: five columns of typical event data.

#![expect(clippy::unwrap_used)]

use std::sync::Arc;
use std::sync::LazyLock;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::Int64Array;
use arrow_array::RecordBatch;
use arrow_array::StringArray;
use divan::Bencher;
use mimalloc::MiMalloc;
use vortex::VortexSessionDefault;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::arrays::ChunkedArray;
use vortex::file::WriteOptionsSessionExt;
use vortex::file::WriteStrategyBuilder;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::runtime::current::CurrentThreadRuntime;
use vortex::io::session::RuntimeSessionExt;
use vortex::session::VortexSession;
use vortex_arrow::ArrowSessionExt;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    divan::main();
}

const ROWS: u64 = 131_072;
const BATCH: u64 = 8192;

const DATASETS: &[&str] = &["urls", "drifting", "table"];

static RUNTIME: LazyLock<CurrentThreadRuntime> = LazyLock::new(CurrentThreadRuntime::new);

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::default().with_handle(RUNTIME.handle()));

/// A deterministic hash, so that every run writes the same data.
fn splitmix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

fn url(row: u64) -> String {
    format!(
        "https://shop.example.com/product/{}?ref=home",
        splitmix(row) % 100_000
    )
}

/// Returns the columns of `dataset` for the rows `start..end`.
fn columns(dataset: &str, start: u64, end: u64) -> Vec<(&'static str, ArrowArrayRef)> {
    let rows = start..end;
    let half = ROWS / 2;

    match dataset {
        "urls" => vec![(
            "url",
            Arc::new(StringArray::from_iter_values(rows.map(url))),
        )],
        "drifting" => vec![
            (
                "runs_then_random",
                Arc::new(Int64Array::from_iter_values(rows.clone().map(|x| {
                    if x < half {
                        (x / 1000) as i64
                    } else {
                        (splitmix(x) % 1_000_000) as i64
                    }
                }))),
            ),
            (
                "random_then_runs",
                Arc::new(Int64Array::from_iter_values(rows.clone().map(|x| {
                    if x < half {
                        (splitmix(x) % 1_000_000) as i64
                    } else {
                        (x / 1000) as i64
                    }
                }))),
            ),
            (
                "categories_then_unique",
                Arc::new(StringArray::from_iter_values(rows.map(|x| {
                    if x < half {
                        format!("category-{}", splitmix(x) % 16)
                    } else {
                        format!("https://shop.example.com/item/{:x}", splitmix(x))
                    }
                }))),
            ),
        ],
        "table" => vec![
            (
                "id",
                Arc::new(Int64Array::from_iter_values(rows.clone().map(|x| x as i64))),
            ),
            (
                "timestamp",
                Arc::new(Int64Array::from_iter_values(
                    rows.clone()
                        .map(|x| 1_700_000_000_000 + (x * 37 + splitmix(x) % 30) as i64),
                )),
            ),
            (
                "price",
                Arc::new(Int64Array::from_iter_values(
                    rows.clone().map(|x| (splitmix(x) % 100_000) as i64),
                )),
            ),
            (
                "url",
                Arc::new(StringArray::from_iter_values(rows.clone().map(url))),
            ),
            (
                "status",
                Arc::new(StringArray::from_iter_values(
                    rows.map(|x| ["ok", "error", "pending"][(splitmix(x) % 3) as usize]),
                )),
            ),
        ],
        _ => unreachable!("unknown dataset {dataset}"),
    }
}

/// Returns `dataset` as a chunked array, converted from Arrow record batches.
fn chunks(dataset: &str) -> ArrayRef {
    let batches: Vec<RecordBatch> = (0..ROWS.div_ceil(BATCH))
        .map(|batch| {
            let end = ((batch + 1) * BATCH).min(ROWS);
            RecordBatch::try_from_iter(columns(dataset, batch * BATCH, end)).unwrap()
        })
        .collect();

    let schema = batches[0].schema();
    let dtype = SESSION.arrow().from_arrow_schema(&schema).unwrap();
    let chunks: Vec<ArrayRef> = batches
        .into_iter()
        .map(|batch| {
            SESSION
                .arrow()
                .from_arrow_record_batch(batch, &schema)
                .unwrap()
        })
        .collect();

    ChunkedArray::try_new(chunks, dtype).unwrap().into_array()
}

/// Writes `chunks` into memory and returns the file size.
fn write(chunks: &ArrayRef) -> usize {
    let strategy = WriteStrategyBuilder::from_session(&SESSION).build();

    let mut file = Vec::new();
    RUNTIME
        .block_on(
            SESSION
                .write_options()
                .with_strategy(strategy)
                .write(&mut file, chunks.to_array_stream()),
        )
        .unwrap();

    file.len()
}

#[divan::bench(args = DATASETS, sample_count = 5, sample_size = 1)]
fn write_default(bencher: Bencher, dataset: &str) {
    let chunks = chunks(dataset);
    bencher.bench_local(|| write(&chunks));
}
