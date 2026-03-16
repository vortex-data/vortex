// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmark comparing FSST symbol table compression against zstd and snappy baselines.
//!
//! Uses static, deterministic datasets (both random and structured) to measure:
//! - Compressed size (bytes)
//! - Compression throughput

#![allow(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::accessor::ArrayAccessor;
use vortex_array::arrays::VarBinArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_fsst::fsst_compress;
use vortex_fsst::fsst_train_compressor;
use vortex_fsst::test_utils::generate_clickbench_urls;
use vortex_fsst::test_utils::generate_emails;
use vortex_fsst::test_utils::generate_json_strings;
use vortex_fsst::test_utils::generate_log_lines;

fn main() {
    // Print size report first, then run throughput benchmarks.
    print_size_report();
    divan::main();
}

// ---------------------------------------------------------------------------
// Static benchmark datasets
// ---------------------------------------------------------------------------

const N: usize = 10_000;

/// Mostly-random binary data: random bytes with a few repeated 4-byte patterns.
fn make_random_binary_dataset() -> Vec<Vec<u8>> {
    let mut rng = StdRng::seed_from_u64(0xDEAD);
    let patterns: Vec<[u8; 4]> = (0..8).map(|_| rng.random()).collect();
    (0..N)
        .map(|_| {
            let len = rng.random_range(20..200);
            let mut buf = Vec::with_capacity(len);
            while buf.len() < len {
                if rng.random_bool(0.15) {
                    let pat = &patterns[rng.random_range(0..patterns.len())];
                    buf.extend_from_slice(pat);
                } else {
                    buf.push(rng.random());
                }
            }
            buf.truncate(len);
            buf
        })
        .collect()
}

/// Structured string data: URLs from the ClickBench-style generator.
fn make_structured_string_dataset() -> Vec<String> {
    generate_clickbench_urls(N)
}

/// Log lines dataset (structured but with numeric variance).
fn make_log_dataset() -> Vec<String> {
    generate_log_lines(N)
}

/// JSON strings dataset.
fn make_json_dataset() -> Vec<String> {
    generate_json_strings(N)
}

/// Email addresses dataset.
fn make_email_dataset() -> Vec<String> {
    generate_emails(N)
}

// ---------------------------------------------------------------------------
// Lazy static datasets
// ---------------------------------------------------------------------------

static RANDOM_BINARY: LazyLock<Vec<Vec<u8>>> = LazyLock::new(make_random_binary_dataset);
static STRUCTURED_URLS: LazyLock<Vec<String>> = LazyLock::new(make_structured_string_dataset);
static LOG_LINES: LazyLock<Vec<String>> = LazyLock::new(make_log_dataset);
static JSON_STRINGS: LazyLock<Vec<String>> = LazyLock::new(make_json_dataset);
static EMAIL_STRINGS: LazyLock<Vec<String>> = LazyLock::new(make_email_dataset);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn concat_bytes(data: &[Vec<u8>]) -> Vec<u8> {
    data.iter().flat_map(|v| v.iter().copied()).collect()
}

fn concat_strings(data: &[String]) -> Vec<u8> {
    data.iter()
        .flat_map(|s| s.as_bytes().iter().copied())
        .collect()
}

fn strings_to_varbin(data: &[String]) -> VarBinArray {
    VarBinArray::from_iter(
        data.iter().map(|s| Some(s.as_str())),
        DType::Utf8(Nullability::NonNullable),
    )
}

fn bytes_to_varbin(data: &[Vec<u8>]) -> VarBinArray {
    VarBinArray::from_iter(
        data.iter().map(|b| Some(Box::<[u8]>::from(b.as_slice()))),
        DType::Binary(Nullability::NonNullable),
    )
}

fn fsst_compressed_size(varbin: &VarBinArray) -> usize {
    let compressor = fsst_train_compressor(varbin);
    let fsst_array = fsst_compress(varbin, &compressor);

    // Symbol table overhead
    let symbol_table_size = fsst_array.symbols().len() * 8 + fsst_array.symbol_lengths().len();

    // Compressed codes size
    let codes_size: usize = fsst_array
        .codes()
        .with_iterator(|it| it.map(|opt| opt.map_or(0, |b| b.len())).sum());

    symbol_table_size + codes_size
}

fn zstd_compressed_size(data: &[u8]) -> usize {
    zstd::encode_all(data, 3).unwrap().len()
}

fn snappy_compressed_size(data: &[u8]) -> usize {
    let mut encoder = snap::raw::Encoder::new();
    encoder.compress_vec(data).unwrap().len()
}

// ---------------------------------------------------------------------------
// Size report (printed before benchmarks)
// ---------------------------------------------------------------------------

fn print_size_report() {
    println!("\n{}", "=".repeat(90));
    println!("FSST vs Baselines - Compressed Size Report");
    println!("{}", "=".repeat(90));

    let random_bytes = concat_bytes(&RANDOM_BINARY);
    let url_bytes = concat_strings(&STRUCTURED_URLS);
    let log_bytes = concat_strings(&LOG_LINES);
    let json_bytes = concat_strings(&JSON_STRINGS);
    let email_bytes = concat_strings(&EMAIL_STRINGS);

    let random_varbin = bytes_to_varbin(&RANDOM_BINARY);
    let url_varbin = strings_to_varbin(&STRUCTURED_URLS);
    let log_varbin = strings_to_varbin(&LOG_LINES);
    let json_varbin = strings_to_varbin(&JSON_STRINGS);
    let email_varbin = strings_to_varbin(&EMAIL_STRINGS);

    let datasets: Vec<(&str, &[u8], &VarBinArray)> = vec![
        ("random_binary", &random_bytes, &random_varbin),
        ("urls", &url_bytes, &url_varbin),
        ("log_lines", &log_bytes, &log_varbin),
        ("json", &json_bytes, &json_varbin),
        ("emails", &email_bytes, &email_varbin),
    ];

    println!(
        "{:<16} {:>10} {:>10} {:>10} {:>10} {:>8} {:>8} {:>8}",
        "dataset", "raw_bytes", "fsst", "zstd", "snappy", "f_ratio", "z_ratio", "s_ratio"
    );
    println!("{}", "-".repeat(90));

    for (name, raw, varbin) in &datasets {
        let raw_size = raw.len();
        let fsst_size = fsst_compressed_size(varbin);
        let zstd_size = zstd_compressed_size(raw);
        let snappy_size = snappy_compressed_size(raw);

        println!(
            "{:<16} {:>10} {:>10} {:>10} {:>10} {:>8.2} {:>8.2} {:>8.2}",
            name,
            raw_size,
            fsst_size,
            zstd_size,
            snappy_size,
            raw_size as f64 / fsst_size as f64,
            raw_size as f64 / zstd_size as f64,
            raw_size as f64 / snappy_size as f64,
        );
    }

    println!("{}", "=".repeat(90));
    drop(datasets);
    println!();
}

// ---------------------------------------------------------------------------
// Throughput benchmarks: FSST compress
// ---------------------------------------------------------------------------

#[divan::bench]
fn compress_fsst_random_binary(bencher: Bencher) {
    let varbin = bytes_to_varbin(&RANDOM_BINARY);
    let compressor = fsst_train_compressor(&varbin);
    bencher
        .with_inputs(|| (&varbin, &compressor))
        .bench_refs(|(v, c)| fsst_compress(*v, c));
}

#[divan::bench]
fn compress_fsst_urls(bencher: Bencher) {
    let varbin = strings_to_varbin(&STRUCTURED_URLS);
    let compressor = fsst_train_compressor(&varbin);
    bencher
        .with_inputs(|| (&varbin, &compressor))
        .bench_refs(|(v, c)| fsst_compress(*v, c));
}

#[divan::bench]
fn compress_fsst_log_lines(bencher: Bencher) {
    let varbin = strings_to_varbin(&LOG_LINES);
    let compressor = fsst_train_compressor(&varbin);
    bencher
        .with_inputs(|| (&varbin, &compressor))
        .bench_refs(|(v, c)| fsst_compress(*v, c));
}

#[divan::bench]
fn compress_fsst_json(bencher: Bencher) {
    let varbin = strings_to_varbin(&JSON_STRINGS);
    let compressor = fsst_train_compressor(&varbin);
    bencher
        .with_inputs(|| (&varbin, &compressor))
        .bench_refs(|(v, c)| fsst_compress(*v, c));
}

#[divan::bench]
fn compress_fsst_emails(bencher: Bencher) {
    let varbin = strings_to_varbin(&EMAIL_STRINGS);
    let compressor = fsst_train_compressor(&varbin);
    bencher
        .with_inputs(|| (&varbin, &compressor))
        .bench_refs(|(v, c)| fsst_compress(*v, c));
}

// ---------------------------------------------------------------------------
// Throughput benchmarks: FSST train
// ---------------------------------------------------------------------------

#[divan::bench]
fn train_fsst_random_binary(bencher: Bencher) {
    let varbin = bytes_to_varbin(&RANDOM_BINARY);
    bencher
        .with_inputs(|| &varbin)
        .bench_refs(|v| fsst_train_compressor(v));
}

#[divan::bench]
fn train_fsst_urls(bencher: Bencher) {
    let varbin = strings_to_varbin(&STRUCTURED_URLS);
    bencher
        .with_inputs(|| &varbin)
        .bench_refs(|v| fsst_train_compressor(v));
}

#[divan::bench]
fn train_fsst_log_lines(bencher: Bencher) {
    let varbin = strings_to_varbin(&LOG_LINES);
    bencher
        .with_inputs(|| &varbin)
        .bench_refs(|v| fsst_train_compressor(v));
}

#[divan::bench]
fn train_fsst_json(bencher: Bencher) {
    let varbin = strings_to_varbin(&JSON_STRINGS);
    bencher
        .with_inputs(|| &varbin)
        .bench_refs(|v| fsst_train_compressor(v));
}

#[divan::bench]
fn train_fsst_emails(bencher: Bencher) {
    let varbin = strings_to_varbin(&EMAIL_STRINGS);
    bencher
        .with_inputs(|| &varbin)
        .bench_refs(|v| fsst_train_compressor(v));
}

// ---------------------------------------------------------------------------
// Throughput benchmarks: zstd compress
// ---------------------------------------------------------------------------

#[divan::bench]
fn compress_zstd_random_binary(bencher: Bencher) {
    let data = concat_bytes(&RANDOM_BINARY);
    bencher
        .with_inputs(|| data.as_slice())
        .bench_refs(|d| zstd::encode_all(*d, 3).unwrap());
}

#[divan::bench]
fn compress_zstd_urls(bencher: Bencher) {
    let data = concat_strings(&STRUCTURED_URLS);
    bencher
        .with_inputs(|| data.as_slice())
        .bench_refs(|d| zstd::encode_all(*d, 3).unwrap());
}

#[divan::bench]
fn compress_zstd_log_lines(bencher: Bencher) {
    let data = concat_strings(&LOG_LINES);
    bencher
        .with_inputs(|| data.as_slice())
        .bench_refs(|d| zstd::encode_all(*d, 3).unwrap());
}

#[divan::bench]
fn compress_zstd_json(bencher: Bencher) {
    let data = concat_strings(&JSON_STRINGS);
    bencher
        .with_inputs(|| data.as_slice())
        .bench_refs(|d| zstd::encode_all(*d, 3).unwrap());
}

#[divan::bench]
fn compress_zstd_emails(bencher: Bencher) {
    let data = concat_strings(&EMAIL_STRINGS);
    bencher
        .with_inputs(|| data.as_slice())
        .bench_refs(|d| zstd::encode_all(*d, 3).unwrap());
}

// ---------------------------------------------------------------------------
// Throughput benchmarks: snappy compress
// ---------------------------------------------------------------------------

#[divan::bench]
fn compress_snappy_random_binary(bencher: Bencher) {
    let data = concat_bytes(&RANDOM_BINARY);
    bencher
        .with_inputs(|| (snap::raw::Encoder::new(), data.as_slice()))
        .bench_refs(|(enc, d)| enc.compress_vec(d).unwrap());
}

#[divan::bench]
fn compress_snappy_urls(bencher: Bencher) {
    let data = concat_strings(&STRUCTURED_URLS);
    bencher
        .with_inputs(|| (snap::raw::Encoder::new(), data.as_slice()))
        .bench_refs(|(enc, d)| enc.compress_vec(d).unwrap());
}

#[divan::bench]
fn compress_snappy_log_lines(bencher: Bencher) {
    let data = concat_strings(&LOG_LINES);
    bencher
        .with_inputs(|| (snap::raw::Encoder::new(), data.as_slice()))
        .bench_refs(|(enc, d)| enc.compress_vec(d).unwrap());
}

#[divan::bench]
fn compress_snappy_json(bencher: Bencher) {
    let data = concat_strings(&JSON_STRINGS);
    bencher
        .with_inputs(|| (snap::raw::Encoder::new(), data.as_slice()))
        .bench_refs(|(enc, d)| enc.compress_vec(d).unwrap());
}

#[divan::bench]
fn compress_snappy_emails(bencher: Bencher) {
    let data = concat_strings(&EMAIL_STRINGS);
    bencher
        .with_inputs(|| (snap::raw::Encoder::new(), data.as_slice()))
        .bench_refs(|(enc, d)| enc.compress_vec(d).unwrap());
}
