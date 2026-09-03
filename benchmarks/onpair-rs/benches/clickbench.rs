// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//! End-to-end OnPair benchmark over a ClickBench string column.
#![allow(
    clippy::cast_possible_truncation,
    clippy::expect_used,
    clippy::missing_panics_doc,
    clippy::unwrap_used
)]
//
// End-to-end benchmark suite over a real parquet file (ClickBench-style
// hits or any UTF-8 string column).
//
// Data source resolution, in order:
//   1. env var `ONPAIR_BENCH_PARQUET` — path to a parquet file
//      (e.g. ClickBench `hits.parquet`). Optionally set
//      `ONPAIR_BENCH_COLUMN` to pick a specific UTF-8 column; otherwise
//      we pick the first Utf8 / LargeUtf8 / Utf8View column in the schema.
//   2. `/tmp/userdata1.parquet` if present (small real-world parquet,
//      good for smoke runs).
//   3. A synthetic ClickBench-shaped URL corpus (100 000 rows of
//      repetitive URLs with realistic prefix sharing).
//
// Run with: cargo bench --bench clickbench

use std::env;
use std::fs::File;
use std::path::PathBuf;
use std::sync::OnceLock;

use std::mem::MaybeUninit;

use arrow_array::Array;
use arrow_array::cast::AsArray;
use divan::Bencher;
use onpair::Column;
use onpair::Config;
use onpair::DECODE_PADDING;
use onpair::MaxDictBits;
use onpair::Threshold;
use onpair::compress;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

const BITS_CONFIGS: &[u8] = &[12, 16];

/// Pack `Vec<Vec<u8>>` (the corpus) into `(bytes, offsets)`.
fn pack(strings: &[Vec<u8>]) -> (Vec<u8>, Vec<u64>) {
    let mut bytes = Vec::with_capacity(strings.iter().map(|s| s.len()).sum());
    let mut offsets = Vec::with_capacity(strings.len() + 1);
    offsets.push(0u64);
    for s in strings {
        bytes.extend_from_slice(s);
        offsets.push(bytes.len() as u64);
    }
    (bytes, offsets)
}

// ─────────────────────────────────────────────────────────────────────────────
// Corpus loading.
// ─────────────────────────────────────────────────────────────────────────────

struct Corpus {
    /// Where the rows came from (printed at startup).
    source: String,
    rows: Vec<Vec<u8>>,
    /// Bytes packed once, reused across benches.
    bytes: Vec<u8>,
    offsets: Vec<u64>,
    total_bytes: usize,
}

fn corpus() -> &'static Corpus {
    static CORPUS: OnceLock<Corpus> = OnceLock::new();
    CORPUS.get_or_init(|| {
        let (source, rows) = load_corpus();
        let (bytes, offsets) = pack(&rows);
        let total_bytes = bytes.len();
        let c = Corpus {
            source,
            rows,
            bytes,
            offsets,
            total_bytes,
        };
        eprintln!(
            "[onpair bench] corpus: {} ({} rows, {:.2} MiB)",
            c.source,
            c.rows.len(),
            c.total_bytes as f64 / (1024.0 * 1024.0)
        );
        c
    })
}

fn load_corpus() -> (String, Vec<Vec<u8>>) {
    if let Ok(path) = env::var("ONPAIR_BENCH_PARQUET")
        && let Some(rows) = read_parquet_strings(&PathBuf::from(&path))
    {
        return (format!("{path} (env)"), rows);
    }
    let fallback = PathBuf::from("/tmp/userdata1.parquet");
    if fallback.exists()
        && let Some(rows) = read_parquet_strings(&fallback)
    {
        return (format!("{} (auto-detected)", fallback.display()), rows);
    }
    let rows = synthetic_clickbench_urls(100_000);
    ("synthetic ClickBench-shaped URL corpus".to_string(), rows)
}

/// Load the largest UTF-8-typed column from a parquet file and return it as
/// `Vec<Vec<u8>>`. Honours `ONPAIR_BENCH_COLUMN` if set.
fn read_parquet_strings(path: &PathBuf) -> Option<Vec<Vec<u8>>> {
    let file = File::open(path).ok()?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).ok()?;
    let schema = builder.schema().clone();

    let col_name = env::var("ONPAIR_BENCH_COLUMN").ok();
    let picked = match col_name.as_deref() {
        Some(name) => schema.fields().iter().position(|f| f.name() == name)?,
        None => {
            // Pick first Utf8 / LargeUtf8 / Utf8View column.
            schema.fields().iter().position(|f| {
                use arrow_schema::DataType::*;
                matches!(f.data_type(), Utf8 | LargeUtf8 | Utf8View)
            })?
        }
    };
    let col_field = schema.fields().get(picked)?.clone();
    eprintln!(
        "[onpair bench] reading column #{picked} `{}` ({})",
        col_field.name(),
        col_field.data_type()
    );

    let mut rows: Vec<Vec<u8>> = Vec::new();
    let reader = builder.build().ok()?;
    for batch in reader.flatten() {
        let arr = batch.column(picked);
        use arrow_schema::DataType::*;
        match arr.data_type() {
            Utf8 => {
                for s in arr.as_string::<i32>().iter() {
                    rows.push(s.unwrap_or("").as_bytes().to_vec());
                }
            }
            LargeUtf8 => {
                for s in arr.as_string::<i64>().iter() {
                    rows.push(s.unwrap_or("").as_bytes().to_vec());
                }
            }
            Utf8View => {
                for s in arr.as_string_view().iter() {
                    rows.push(s.unwrap_or("").as_bytes().to_vec());
                }
            }
            _ => return None,
        }
    }
    Some(rows)
}

/// 100 000 URLs whose distribution roughly matches ClickBench's URL column:
///   * heavy prefix sharing on `https://`, `http://`, `ftp://`
///   * a handful of repeating host roots
///   * variable path / query parts
fn synthetic_clickbench_urls(n: usize) -> Vec<Vec<u8>> {
    const HOSTS: &[&str] = &[
        "https://www.yandex.ru",
        "https://www.google.com",
        "https://news.ycombinator.com",
        "https://www.example.com",
        "https://docs.example.org",
        "https://api.example.net",
        "http://m.yandex.ru",
        "https://maps.example.com",
        "https://shop.example.com",
        "ftp://files.example.com",
    ];
    const PATHS: &[&str] = &[
        "/",
        "/page",
        "/news",
        "/search?q=",
        "/profile",
        "/login",
        "/api/v1/data",
        "/static/asset.png",
        "/blog/post-",
        "/feed.xml",
        "/sitemap.xml",
        "/users/",
        "/admin/dashboard",
        "/categories/electronics",
        "/cart/checkout",
    ];
    const TAILS: &[&str] = &["", "alpha", "beta", "gamma", "delta", "001", "002", "003"];
    let mut out = Vec::with_capacity(n);
    let mut x = 0x9E3779B97F4A7C15u64;
    for _ in 0..n {
        // SplitMix64-style state advance — deterministic, no rand dep.
        x = x.wrapping_add(0x9E3779B97F4A7C15);
        let h = HOSTS[(x as usize) % HOSTS.len()];
        let p = PATHS[((x >> 16) as usize) % PATHS.len()];
        let t = TAILS[((x >> 32) as usize) % TAILS.len()];
        let n = (x >> 48) as u16;
        out.push(format!("{h}{p}{t}{n}").into_bytes());
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers shared by the bench groups.
// ─────────────────────────────────────────────────────────────────────────────

fn compress_column(bits: u8) -> Column<u64> {
    let c = corpus();
    let cfg = Config {
        max_dict_bits: MaxDictBits::new(bits).unwrap(),
        threshold: Threshold::new(0.5).unwrap(),
        seed: Some(42),
    };
    compress(&c.bytes, &c.offsets, cfg).unwrap()
}

// ─────────────────────────────────────────────────────────────────────────────
// Benches.
// ─────────────────────────────────────────────────────────────────────────────

#[divan::bench(args = BITS_CONFIGS)]
fn train_and_compress(bencher: Bencher, bits: u8) {
    let c = corpus();
    bencher
        .counter(divan::counter::BytesCount::new(c.total_bytes))
        .bench(|| {
            let cfg = Config {
                max_dict_bits: MaxDictBits::new(bits).unwrap(),
                threshold: Threshold::new(0.5).unwrap(),
                seed: Some(42),
            };
            compress(
                divan::black_box(&c.bytes),
                divan::black_box(&c.offsets),
                cfg,
            )
            .unwrap()
        });
}

#[divan::bench(args = BITS_CONFIGS)]
fn decompress_all(bencher: Bencher, bits: u8) {
    let c = corpus();
    let col = compress_column(bits);
    let cap = col.view().decoded_len() + DECODE_PADDING;
    bencher
        .counter(divan::counter::BytesCount::new(c.total_bytes))
        .bench(|| {
            let mut buf = vec![MaybeUninit::uninit(); cap];
            // SAFETY: trusted column; `buf` is sized to decoded_len() + DECODE_PADDING.
            let n = unsafe { col.view().decompress_into(&mut buf) };
            divan::black_box(&buf[..n]);
        });
}

fn main() {
    // Touch the corpus so the source line prints before divan begins.
    let _ = corpus();
    divan::main();
}
