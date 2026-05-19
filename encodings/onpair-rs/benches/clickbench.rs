// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#![allow(
    clippy::cast_possible_truncation,
    clippy::clone_on_ref_ptr,
    clippy::expect_used,
    clippy::many_single_char_names,
    clippy::missing_panics_doc,
    clippy::unwrap_in_result,
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
//      we pick the first BYTE_ARRAY / Utf8 / Utf8View column with the
//      largest total byte volume.
//   2. `/tmp/userdata1.parquet` if present (small real-world parquet,
//      good for smoke runs).
//   3. A synthetic ClickBench-shaped URL corpus (100 000 rows of
//      repetitive URLs with realistic prefix sharing).
//
// Each benchmark group runs three configurations:
//   * the full pipeline   (`train_and_compress`)
//   * a single op against an already-built `Column`
//
// Run with: cargo bench -p onpair-lib --bench clickbench

use std::env;
use std::fs::File;
use std::path::PathBuf;
use std::sync::OnceLock;

use arrow_array::Array;
use arrow_array::cast::AsArray;
use divan::Bencher;
use onpair_lib::{
    AhoCorasickAutomaton, Column, EqAutomaton, KmpAutomaton, OnPairTrainingConfig,
    PrefixAutomaton, and, not,
};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

const BITS_CONFIGS: &[u32] = &[12, 16];

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
        let c = Corpus { source, rows, bytes, offsets, total_bytes };
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
        return (
            format!("{} (auto-detected)", fallback.display()),
            rows,
        );
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
        Some(name) => schema
            .fields()
            .iter()
            .position(|f| f.name() == name)?,
        None => {
            // Pick first byte-string-typed column (Utf8 / Binary in any form).
            schema.fields().iter().position(|f| {
                use arrow_schema::DataType::*;
                matches!(
                    f.data_type(),
                    Utf8 | LargeUtf8 | Utf8View | Binary | LargeBinary | BinaryView
                )
            })?
        }
    };
    let col_field = schema.fields().get(picked).unwrap().clone();
    eprintln!(
        "[onpair bench] reading column #{picked} `{}` ({})",
        col_field.name(),
        col_field.data_type()
    );

    // Optional row cap for the cases where we want to fit a full corpus in
    // L3 / cap a single-bench run.
    let cap: Option<usize> = env::var("ONPAIR_BENCH_ROWS")
        .ok()
        .and_then(|s| s.parse().ok());

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
            Binary => {
                for s in arr.as_binary::<i32>().iter() {
                    rows.push(s.unwrap_or(&[]).to_vec());
                }
            }
            LargeBinary => {
                for s in arr.as_binary::<i64>().iter() {
                    rows.push(s.unwrap_or(&[]).to_vec());
                }
            }
            BinaryView => {
                for s in arr.as_binary_view().iter() {
                    rows.push(s.unwrap_or(&[]).to_vec());
                }
            }
            _ => return None,
        }
        if let Some(c) = cap
            && rows.len() >= c
        {
            rows.truncate(c);
            break;
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
        "/", "/page", "/news", "/search?q=", "/profile",
        "/login", "/api/v1/data", "/static/asset.png", "/blog/post-",
        "/feed.xml", "/sitemap.xml", "/users/", "/admin/dashboard",
        "/categories/electronics", "/cart/checkout",
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

fn compress_column(bits: u32) -> Column {
    let c = corpus();
    let cfg = OnPairTrainingConfig { bits, threshold: 0.5, seed: 42 };
    Column::compress(&c.bytes, &c.offsets, cfg).unwrap()
}

/// Pick a needle that almost certainly appears in the corpus (for substring
/// queries) and one that definitely doesn't (for negative queries).
fn substring_needle() -> &'static [u8] {
    b"example"
}

fn equality_needle() -> Vec<u8> {
    corpus().rows.get(corpus().rows.len() / 2).cloned().unwrap_or_default()
}

fn prefix_needle() -> &'static [u8] {
    b"https://"
}

// ─────────────────────────────────────────────────────────────────────────────
// Benches.
// ─────────────────────────────────────────────────────────────────────────────

#[divan::bench(args = BITS_CONFIGS)]
fn train_and_compress(bencher: Bencher, bits: u32) {
    let c = corpus();
    bencher
        .counter(divan::counter::BytesCount::new(c.total_bytes))
        .bench(|| {
            let cfg = OnPairTrainingConfig { bits, threshold: 0.5, seed: 42 };
            Column::compress(divan::black_box(&c.bytes), divan::black_box(&c.offsets), cfg).unwrap()
        });
}

#[divan::bench]
fn train_and_compress_auto(bencher: Bencher) {
    let c = corpus();
    bencher
        .counter(divan::counter::BytesCount::new(c.total_bytes))
        .bench(|| {
            Column::compress_auto(divan::black_box(&c.bytes), divan::black_box(&c.offsets)).unwrap()
        });
}

#[divan::bench(args = BITS_CONFIGS)]
fn decompress_row_random(bencher: Bencher, bits: u32) {
    let col = compress_column(bits);
    let n = col.len();
    let mut buf = Vec::with_capacity(256);
    let mut x = 0xC2B2AE3D27D4EB4Fu64;
    bencher.bench_local(|| {
        x = x.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
        let row = (x as usize) % n;
        let _ = col.decompress_row(divan::black_box(row), &mut buf);
        divan::black_box(&buf);
    });
}

#[divan::bench(args = BITS_CONFIGS)]
fn decode_all(bencher: Bencher, bits: u32) {
    let c = corpus();
    let col = compress_column(bits);
    bencher
        .counter(divan::counter::BytesCount::new(c.total_bytes))
        .bench(|| {
            divan::black_box(col.decode_all());
        });
}

// ── Bitmap (decompress-then-match) predicates ─────────────────────────────────

#[divan::bench(args = BITS_CONFIGS)]
fn equals_bitmap(bencher: Bencher, bits: u32) {
    let col = compress_column(bits);
    let needle = equality_needle();
    bencher
        .counter(divan::counter::ItemsCount::new(col.len()))
        .bench(|| divan::black_box(col.equals_bitmap(&needle)));
}

#[divan::bench(args = BITS_CONFIGS)]
fn starts_with_bitmap(bencher: Bencher, bits: u32) {
    let col = compress_column(bits);
    bencher
        .counter(divan::counter::ItemsCount::new(col.len()))
        .bench(|| divan::black_box(col.starts_with_bitmap(prefix_needle())));
}

#[divan::bench(args = BITS_CONFIGS)]
fn contains_bitmap(bencher: Bencher, bits: u32) {
    let col = compress_column(bits);
    bencher
        .counter(divan::counter::ItemsCount::new(col.len()))
        .bench(|| divan::black_box(col.contains_bitmap(substring_needle())));
}

#[divan::bench(args = BITS_CONFIGS)]
fn multi_pattern_bitmap(bencher: Bencher, bits: u32) {
    let col = compress_column(bits);
    let needles: &[&[u8]] = &[b"example", b"yandex", b"google", b"news"];
    bencher
        .counter(divan::counter::ItemsCount::new(col.len()))
        .bench(|| divan::black_box(col.multi_pattern_bitmap(needles)));
}

// ── Token-automaton (compressed-domain) predicates ────────────────────────────

#[divan::bench(args = BITS_CONFIGS)]
fn eq_automaton(bencher: Bencher, bits: u32) {
    let col = compress_column(bits);
    let dict = col.dictionary().clone();
    let needle = equality_needle();
    bencher
        .counter(divan::counter::ItemsCount::new(col.len()))
        .bench(|| {
            let eq = EqAutomaton::new(&needle, &dict);
            divan::black_box(col.scan_bitmap(eq));
        });
}

#[divan::bench(args = BITS_CONFIGS)]
fn prefix_automaton(bencher: Bencher, bits: u32) {
    let col = compress_column(bits);
    let dict = col.dictionary().clone();
    bencher
        .counter(divan::counter::ItemsCount::new(col.len()))
        .bench(|| {
            let pa = PrefixAutomaton::new(prefix_needle(), &dict);
            divan::black_box(col.scan_bitmap(pa));
        });
}

#[divan::bench(args = BITS_CONFIGS)]
fn kmp_automaton(bencher: Bencher, bits: u32) {
    let col = compress_column(bits);
    let dict = col.dictionary().clone();
    bencher
        .counter(divan::counter::ItemsCount::new(col.len()))
        .bench(|| {
            let kmp = KmpAutomaton::new(substring_needle(), &dict);
            divan::black_box(col.scan_bitmap(kmp));
        });
}

#[divan::bench(args = BITS_CONFIGS)]
fn ac_automaton(bencher: Bencher, bits: u32) {
    let col = compress_column(bits);
    let dict = col.dictionary().clone();
    let needles: &[&[u8]] = &[b"example", b"yandex", b"google", b"news"];
    bencher
        .counter(divan::counter::ItemsCount::new(col.len()))
        .bench(|| {
            let ac = AhoCorasickAutomaton::new(needles, &dict);
            divan::black_box(col.scan_bitmap(ac));
        });
}

#[divan::bench(args = BITS_CONFIGS)]
fn and_not_compressed_domain(bencher: Bencher, bits: u32) {
    let col = compress_column(bits);
    let dict = col.dictionary().clone();
    bencher
        .counter(divan::counter::ItemsCount::new(col.len()))
        .bench(|| {
            let mut a = KmpAutomaton::new(b"example", &dict);
            let mut b = KmpAutomaton::new(b"yandex", &dict);
            divan::black_box(col.scan_bitmap(and(&mut a, not(&mut b))));
        });
}

// ── C++ comparison via vortex-onpair-sys ──────────────────────────────────────

#[divan::bench(args = BITS_CONFIGS)]
fn cpp_train_and_compress(bencher: Bencher, bits: u32) {
    use vortex_onpair_sys::Column as CppColumn;
    use vortex_onpair_sys::OnPairTrainingConfig as CppCfg;
    let c = corpus();
    bencher
        .counter(divan::counter::BytesCount::new(c.total_bytes))
        .bench(|| {
            let cfg = CppCfg { bits, threshold: 0.5, seed: 42 };
            CppColumn::compress(&c.bytes, &c.offsets, cfg).unwrap()
        });
}

#[divan::bench(args = BITS_CONFIGS)]
fn cpp_contains_bitmap(bencher: Bencher, bits: u32) {
    use vortex_onpair_sys::Column as CppColumn;
    use vortex_onpair_sys::OnPairTrainingConfig as CppCfg;
    let c = corpus();
    let cfg = CppCfg { bits, threshold: 0.5, seed: 42 };
    let col = CppColumn::compress(&c.bytes, &c.offsets, cfg).unwrap();
    bencher
        .counter(divan::counter::ItemsCount::new(col.len()))
        .bench(|| divan::black_box(col.contains_bitmap(substring_needle())));
}

fn main() {
    // Touch the corpus so the source line prints before divan begins.
    let _ = corpus();
    divan::main();
}
