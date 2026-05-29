// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
//! Decode-path benchmark for OnPair on *real* string data, comparing every
//! column against a fast baseline (FSST — the established Vortex string
//! encoding).
//!
//! Two corpora:
//!
//! * **TPC-H** string columns generated in-memory via `tpchgen`
//!   (`l_comment`, `o_comment`, `p_name`, `c_comment`). Deterministic, no
//!   network, no on-disk fixtures.
//! * **ClickBench** — a real parquet file pointed at by `ONPAIR_BENCH_PARQUET`
//!   (e.g. ClickBench `hits.parquet`); optionally `ONPAIR_BENCH_COLUMN` picks
//!   a specific UTF-8 column, otherwise the largest-by-bytes string column is
//!   used. If the env var is unset we fall back to a synthetic
//!   ClickBench-shaped URL corpus so the bench always runs.
//!
//! Each column is compressed once with OnPair and once with FSST; the bench
//! then times the canonicalisation (decompression) of each back to a
//! `VarBinViewArray`. The OnPair number is the optimised decode path; the
//! FSST number is the fast baseline to compare against.
//!
//! Env knobs:
//!   * `ONPAIR_BENCH_PARQUET`      — path to a ClickBench-style parquet file.
//!   * `ONPAIR_BENCH_COLUMN`       — specific UTF-8 column to pick from it.
//!   * `ONPAIR_BENCH_SCALE_FACTOR` — TPC-H scale factor (default 0.5).
//!   * `ONPAIR_BENCH_MAX_BYTES`    — per-column corpus cap (default 16 MiB).
//!
//! Run with: `cargo bench -p vortex-onpair --bench real_data`

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::clone_on_ref_ptr,
    clippy::disallowed_types,
    clippy::panic,
    clippy::tests_outside_test_module,
    clippy::unwrap_used,
    clippy::use_debug,
    clippy::expect_used
)]

use std::env;
use std::sync::LazyLock;

use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_schema::DataType;
use divan::Bencher;
use divan::counter::BytesCount;
use tpchgen::generators::CustomerGenerator;
use tpchgen::generators::LineItemGenerator;
use tpchgen::generators::OrderGenerator;
use tpchgen::generators::PartGenerator;
use tpchgen_arrow::CustomerArrow;
use tpchgen_arrow::LineItemArrow;
use tpchgen_arrow::OrderArrow;
use tpchgen_arrow::PartArrow;
use tpchgen_arrow::RecordBatchIterator;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::session::ArraySession;
use vortex_fsst::FSSTArray;
use vortex_fsst::fsst_compress;
use vortex_fsst::fsst_train_compressor;
use vortex_onpair::DEFAULT_DICT12_CONFIG;
use vortex_onpair::OnPairArray;
use vortex_onpair::onpair_compress;
use vortex_session::VortexSession;

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

const SCALE_FACTOR_DEFAULT: f64 = 0.5;
const MAX_BYTES_DEFAULT: usize = 16 << 20;
const BATCH_SIZE: usize = 8192 * 8;

fn scale_factor() -> f64 {
    env::var("ONPAIR_BENCH_SCALE_FACTOR")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(SCALE_FACTOR_DEFAULT)
}

fn max_bytes() -> usize {
    env::var("ONPAIR_BENCH_MAX_BYTES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(MAX_BYTES_DEFAULT)
}

/// One materialised string corpus, ready to compress.
struct Corpus {
    rows: Vec<Vec<u8>>,
    total_bytes: usize,
}

impl Corpus {
    fn varbin(&self) -> VarBinArray {
        VarBinArray::from_iter(
            self.rows.iter().map(|s| Some(s.as_slice())),
            DType::Utf8(Nullability::NonNullable),
        )
    }

    fn onpair(&self) -> OnPairArray {
        let varbin = self.varbin();
        onpair_compress(&varbin, varbin.len(), varbin.dtype(), DEFAULT_DICT12_CONFIG)
            .expect("onpair_compress")
    }

    fn fsst(&self) -> FSSTArray {
        let varbin = self.varbin();
        let compressor = fsst_train_compressor(&varbin);
        let mut ctx = SESSION.create_execution_ctx();
        fsst_compress(&varbin, varbin.len(), varbin.dtype(), &compressor, &mut ctx)
    }
}

/// Pull a single UTF-8 column out of a TPC-H table generator, capped at
/// `max_bytes` total decoded bytes.
fn collect_tpch<I: Iterator<Item = arrow_array::RecordBatch> + RecordBatchIterator>(
    it: I,
    col: &str,
    cap: usize,
) -> Corpus {
    let schema = it.schema().clone();
    let idx = schema
        .fields()
        .iter()
        .position(|f| f.name() == col)
        .unwrap_or_else(|| panic!("column `{col}` not found"));

    let mut rows = Vec::new();
    let mut total_bytes = 0usize;
    'outer: for batch in it {
        let arr = batch.column(idx).as_string_view();
        for v in arr.iter() {
            let s = v.unwrap_or("").as_bytes().to_vec();
            total_bytes += s.len();
            rows.push(s);
            if total_bytes >= cap {
                break 'outer;
            }
        }
    }
    Corpus { rows, total_bytes }
}

fn tpch_corpus(col: &'static str) -> Corpus {
    let sf = scale_factor();
    let cap = max_bytes();
    match col {
        "l_comment" => collect_tpch(
            LineItemArrow::new(LineItemGenerator::new(sf, 1, 1)).with_batch_size(BATCH_SIZE),
            col,
            cap,
        ),
        "o_comment" => collect_tpch(
            OrderArrow::new(OrderGenerator::new(sf, 1, 1)).with_batch_size(BATCH_SIZE),
            col,
            cap,
        ),
        "c_comment" => collect_tpch(
            CustomerArrow::new(CustomerGenerator::new(sf, 1, 1)).with_batch_size(BATCH_SIZE),
            col,
            cap,
        ),
        "p_name" => collect_tpch(
            PartArrow::new(PartGenerator::new(sf, 1, 1)).with_batch_size(BATCH_SIZE),
            col,
            cap,
        ),
        other => panic!("unknown TPC-H column `{other}`"),
    }
}

/// Load the ClickBench parquet column from `ONPAIR_BENCH_PARQUET`, or fall
/// back to a synthetic ClickBench-shaped URL corpus.
fn clickbench_corpus() -> Corpus {
    let cap = max_bytes();
    match env::var("ONPAIR_BENCH_PARQUET") {
        Ok(path) => load_parquet_column(&path, cap),
        Err(_) => synthetic_clickbench(cap),
    }
}

fn load_parquet_column(path: &str, cap: usize) -> Corpus {
    use std::fs::File;

    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    let file = File::open(path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).expect("parquet reader");
    let schema = builder.schema().clone();

    let wanted = env::var("ONPAIR_BENCH_COLUMN").ok();
    let idx = match &wanted {
        Some(name) => schema
            .fields()
            .iter()
            .position(|f| f.name() == name)
            .unwrap_or_else(|| panic!("column `{name}` not found in {path}")),
        None => schema
            .fields()
            .iter()
            .enumerate()
            .filter(|(_, f)| {
                matches!(
                    f.data_type(),
                    DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
                )
            })
            .map(|(i, _)| i)
            .next()
            .unwrap_or_else(|| panic!("no UTF-8 column in {path}")),
    };

    let reader = builder
        .with_batch_size(BATCH_SIZE)
        .build()
        .expect("parquet batch reader");

    let mut rows = Vec::new();
    let mut total_bytes = 0usize;
    'outer: for batch in reader {
        let batch = batch.expect("read batch");
        let col = batch.column(idx);
        let strs: Vec<Option<&str>> = match col.data_type() {
            DataType::Utf8 => col.as_string::<i32>().iter().collect(),
            DataType::LargeUtf8 => col.as_string::<i64>().iter().collect(),
            DataType::Utf8View => col.as_string_view().iter().collect(),
            other => panic!("unexpected column type {other:?}"),
        };
        for v in strs {
            let s = v.unwrap_or("").as_bytes().to_vec();
            total_bytes += s.len();
            rows.push(s);
            if total_bytes >= cap {
                break 'outer;
            }
        }
    }
    eprintln!(
        "[onpair real_data] clickbench column #{idx} ({:?}): {} rows, {:.2} MiB",
        wanted,
        rows.len(),
        total_bytes as f64 / (1024.0 * 1024.0)
    );
    Corpus { rows, total_bytes }
}

/// Synthetic ClickBench-shaped URLs (heavy prefix sharing) used when no real
/// parquet file is supplied.
fn synthetic_clickbench(cap: usize) -> Corpus {
    let templates: &[&str] = &[
        "http://www.example.com/search?q={id}&page={p}",
        "http://shop.example.com/product/{id}/reviews",
        "https://news.example.org/2026/05/article-{id}.html",
        "http://maps.example.com/place/{id}?lat=55.7&lon=37.6",
        "https://api.example.io/v3/users/{id}/timeline",
    ];
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state
    };
    let mut rows = Vec::new();
    let mut total_bytes = 0usize;
    while total_bytes < cap {
        let s = next();
        let t = templates[(s as usize) % templates.len()];
        let row = t
            .replace("{id}", &format!("{:08x}", s as u32))
            .replace("{p}", &format!("{}", (s >> 32) % 100))
            .into_bytes();
        total_bytes += row.len();
        rows.push(row);
    }
    Corpus { rows, total_bytes }
}

/// Generate (or fetch from cache) the corpus for a column. Cached behind a
/// `Mutex<HashMap>` and `Box::leak`'d so each bench closure gets a `&'static`
/// reference and the expensive TPC-H generation only runs once per column.
fn corpus_for(name: &str) -> &'static Corpus {
    use std::collections::HashMap;
    use std::sync::Mutex;
    use std::sync::OnceLock;

    static CACHE: OnceLock<Mutex<HashMap<String, &'static Corpus>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut map = cache.lock().expect("cache poisoned");
    if let Some(&c) = map.get(name) {
        return c;
    }
    let corpus: &'static Corpus = Box::leak(Box::new(match name {
        "clickbench" => clickbench_corpus(),
        col => tpch_corpus(col_static(col)),
    }));
    map.insert(name.to_string(), corpus);
    corpus
}

/// TPC-H dispatch needs a `&'static str`; the column set is fixed.
fn col_static(col: &str) -> &'static str {
    COLUMNS
        .iter()
        .copied()
        .find(|&c| c == col)
        .unwrap_or_else(|| panic!("unknown column `{col}`"))
}

const COLUMNS: &[&str] = &[
    "clickbench",
    "l_comment",
    "o_comment",
    "c_comment",
    "p_name",
];

/// OnPair decode (optimised canonicalisation path).
#[divan::bench(args = COLUMNS)]
fn onpair_decode(bencher: Bencher, col: &str) {
    let corpus = corpus_for(col);
    let arr = corpus.onpair();
    bencher
        .counter(BytesCount::new(corpus.total_bytes))
        .with_inputs(|| arr.clone().into_array())
        .bench_local_values(|arr| {
            let mut ctx = SESSION.create_execution_ctx();
            divan::black_box(
                arr.execute::<VarBinViewArray>(&mut ctx)
                    .expect("onpair decode"),
            )
        });
}

/// FSST decode — the fast baseline to compare each column against.
#[divan::bench(args = COLUMNS)]
fn fsst_decode(bencher: Bencher, col: &str) {
    let corpus = corpus_for(col);
    let arr = corpus.fsst();
    bencher
        .counter(BytesCount::new(corpus.total_bytes))
        .with_inputs(|| arr.clone().into_array())
        .bench_local_values(|arr| {
            let mut ctx = SESSION.create_execution_ctx();
            divan::black_box(
                arr.execute::<VarBinViewArray>(&mut ctx)
                    .expect("fsst decode"),
            )
        });
}

fn main() {
    divan::main();
}
