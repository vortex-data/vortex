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
use vortex_fsst::test_utils::generate_csv_rows;
use vortex_fsst::test_utils::generate_emails;
use vortex_fsst::test_utils::generate_file_paths;
use vortex_fsst::test_utils::generate_json_strings;
use vortex_fsst::test_utils::generate_key_value_config;
use vortex_fsst::test_utils::generate_log_lines;
use vortex_fsst::test_utils::generate_repeated_binary;
use vortex_fsst::test_utils::generate_sql_queries;
use vortex_fsst::test_utils::generate_timestamps_with_prefix;
use vortex_fsst::test_utils::generate_xml_fragments;

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

/// File paths dataset (Unix-style paths with common prefixes).
fn make_file_paths_dataset() -> Vec<String> {
    generate_file_paths(N)
}

/// Structured binary: key-value pairs with fixed-size headers and variable payloads.
/// Simulates protobuf/thrift-style wire format.
fn make_structured_binary_dataset() -> Vec<Vec<u8>> {
    let mut rng = StdRng::seed_from_u64(0xBEEF);
    let field_tags: Vec<[u8; 2]> = (0..16).map(|i| [0x08 + i, 0x12 + i]).collect();
    (0..N)
        .map(|_| {
            let n_fields = rng.random_range(3..12);
            let mut buf = Vec::with_capacity(n_fields * 20);
            for _ in 0..n_fields {
                let tag = &field_tags[rng.random_range(0..field_tags.len())];
                buf.extend_from_slice(tag);
                let val_len: u8 = rng.random_range(2..16);
                buf.push(val_len);
                for _ in 0..val_len {
                    buf.push(rng.random_range(0x20..0x7F));
                }
            }
            buf
        })
        .collect()
}

/// CSV rows with repeated values.
fn make_csv_dataset() -> Vec<String> {
    generate_csv_rows(N)
}

/// SQL queries with highly repeated keywords.
fn make_sql_dataset() -> Vec<String> {
    generate_sql_queries(N)
}

/// XML fragments with nested tags and namespaces.
fn make_xml_dataset() -> Vec<String> {
    generate_xml_fragments(N)
}

/// Binary data with high pattern repetition (zstd/snappy-friendly).
fn make_repeated_binary_dataset() -> Vec<Vec<u8>> {
    generate_repeated_binary(N)
}

/// Config key=value lines with shared prefixes.
fn make_config_dataset() -> Vec<String> {
    generate_key_value_config(N)
}

/// Timestamped log lines with highly repetitive prefixes.
fn make_timestamp_log_dataset() -> Vec<String> {
    generate_timestamps_with_prefix(N)
}

// ---------------------------------------------------------------------------
// Lazy static datasets
// ---------------------------------------------------------------------------

static RANDOM_BINARY: LazyLock<Vec<Vec<u8>>> = LazyLock::new(make_random_binary_dataset);
static STRUCTURED_BINARY: LazyLock<Vec<Vec<u8>>> = LazyLock::new(make_structured_binary_dataset);
static STRUCTURED_URLS: LazyLock<Vec<String>> = LazyLock::new(make_structured_string_dataset);
static LOG_LINES: LazyLock<Vec<String>> = LazyLock::new(make_log_dataset);
static JSON_STRINGS: LazyLock<Vec<String>> = LazyLock::new(make_json_dataset);
static EMAIL_STRINGS: LazyLock<Vec<String>> = LazyLock::new(make_email_dataset);
static FILE_PATHS: LazyLock<Vec<String>> = LazyLock::new(make_file_paths_dataset);
static CSV_ROWS: LazyLock<Vec<String>> = LazyLock::new(make_csv_dataset);
static SQL_QUERIES: LazyLock<Vec<String>> = LazyLock::new(make_sql_dataset);
static XML_FRAGMENTS: LazyLock<Vec<String>> = LazyLock::new(make_xml_dataset);
static REPEATED_BINARY: LazyLock<Vec<Vec<u8>>> = LazyLock::new(make_repeated_binary_dataset);
static CONFIG_LINES: LazyLock<Vec<String>> = LazyLock::new(make_config_dataset);
static TIMESTAMP_LOGS: LazyLock<Vec<String>> = LazyLock::new(make_timestamp_log_dataset);

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

struct FsstStats {
    compressed_size: usize,
    escape_count: usize,
    symbol_count: usize,
    num_symbols: u8,
    length_histogram: [u32; 8],
    #[allow(dead_code)]
    top_escape_bytes: Vec<u8>,
}

fn fsst_analyze(varbin: &VarBinArray) -> FsstStats {
    let compressor = fsst_train_compressor(varbin);
    let fsst_array = fsst_compress(varbin, &compressor);

    // Symbol table overhead
    let symbol_table_size = fsst_array.symbols().len() * 8 + fsst_array.symbol_lengths().len();

    // Compressed codes size and escape analysis
    let mut total_escapes = 0usize;
    let mut total_symbols = 0usize;
    #[allow(clippy::disallowed_types)]
    let mut all_escape_bytes = std::collections::HashMap::<u8, usize>::new();

    let codes_size: usize = fsst_array.codes().with_iterator(|it| {
        it.map(|opt| {
            opt.map_or(0, |b| {
                let (esc, sym, esc_bytes) = compressor.count_escapes(b);
                total_escapes += esc;
                total_symbols += sym;
                for &eb in &esc_bytes {
                    *all_escape_bytes.entry(eb).or_default() += 1;
                }
                b.len()
            })
        })
        .sum()
    });

    let mut top_escape_bytes: Vec<u8> = all_escape_bytes.keys().copied().collect();
    top_escape_bytes.sort_by(|a, b| all_escape_bytes[b].cmp(&all_escape_bytes[a]));
    top_escape_bytes.truncate(10);

    FsstStats {
        compressed_size: symbol_table_size + codes_size,
        escape_count: total_escapes,
        symbol_count: total_symbols,
        num_symbols: compressor.num_symbols(),
        length_histogram: compressor.length_histogram(),
        top_escape_bytes,
    }
}

/// Convenience wrapper for use in individual benchmarks.
#[allow(dead_code)]
fn fsst_compressed_size(varbin: &VarBinArray) -> usize {
    fsst_analyze(varbin).compressed_size
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
    let struct_bin_bytes = concat_bytes(&STRUCTURED_BINARY);
    let url_bytes = concat_strings(&STRUCTURED_URLS);
    let log_bytes = concat_strings(&LOG_LINES);
    let json_bytes = concat_strings(&JSON_STRINGS);
    let email_bytes = concat_strings(&EMAIL_STRINGS);
    let path_bytes = concat_strings(&FILE_PATHS);
    let csv_bytes = concat_strings(&CSV_ROWS);
    let sql_bytes = concat_strings(&SQL_QUERIES);
    let xml_bytes = concat_strings(&XML_FRAGMENTS);
    let repeated_bin_bytes = concat_bytes(&REPEATED_BINARY);
    let config_bytes = concat_strings(&CONFIG_LINES);
    let ts_log_bytes = concat_strings(&TIMESTAMP_LOGS);

    let random_varbin = bytes_to_varbin(&RANDOM_BINARY);
    let struct_bin_varbin = bytes_to_varbin(&STRUCTURED_BINARY);
    let url_varbin = strings_to_varbin(&STRUCTURED_URLS);
    let log_varbin = strings_to_varbin(&LOG_LINES);
    let json_varbin = strings_to_varbin(&JSON_STRINGS);
    let email_varbin = strings_to_varbin(&EMAIL_STRINGS);
    let path_varbin = strings_to_varbin(&FILE_PATHS);
    let csv_varbin = strings_to_varbin(&CSV_ROWS);
    let sql_varbin = strings_to_varbin(&SQL_QUERIES);
    let xml_varbin = strings_to_varbin(&XML_FRAGMENTS);
    let repeated_bin_varbin = bytes_to_varbin(&REPEATED_BINARY);
    let config_varbin = strings_to_varbin(&CONFIG_LINES);
    let ts_log_varbin = strings_to_varbin(&TIMESTAMP_LOGS);

    let datasets: Vec<(&str, &[u8], &VarBinArray)> = vec![
        ("random_binary", &random_bytes, &random_varbin),
        ("struct_binary", &struct_bin_bytes, &struct_bin_varbin),
        ("repeat_binary", &repeated_bin_bytes, &repeated_bin_varbin),
        ("urls", &url_bytes, &url_varbin),
        ("log_lines", &log_bytes, &log_varbin),
        ("json", &json_bytes, &json_varbin),
        ("emails", &email_bytes, &email_varbin),
        ("file_paths", &path_bytes, &path_varbin),
        ("csv_rows", &csv_bytes, &csv_varbin),
        ("sql_queries", &sql_bytes, &sql_varbin),
        ("xml", &xml_bytes, &xml_varbin),
        ("config_kv", &config_bytes, &config_varbin),
        ("ts_logs", &ts_log_bytes, &ts_log_varbin),
    ];

    println!(
        "{:<16} {:>10} {:>10} {:>10} {:>10} {:>8} {:>8} {:>8}",
        "dataset", "raw_bytes", "fsst", "zstd", "snappy", "f_ratio", "z_ratio", "s_ratio"
    );
    println!("{}", "-".repeat(90));

    for (name, raw, varbin) in &datasets {
        let raw_size = raw.len();
        let stats = fsst_analyze(varbin);
        let zstd_size = zstd_compressed_size(raw);
        let snappy_size = snappy_compressed_size(raw);

        println!(
            "{:<16} {:>10} {:>10} {:>10} {:>10} {:>8.2} {:>8.2} {:>8.2}",
            name,
            raw_size,
            stats.compressed_size,
            zstd_size,
            snappy_size,
            raw_size as f64 / stats.compressed_size as f64,
            raw_size as f64 / zstd_size as f64,
            raw_size as f64 / snappy_size as f64,
        );
    }

    println!("{}", "=".repeat(90));

    // Escape analysis
    println!("\nFSST Escape Analysis:");
    println!(
        "{:<16} {:>8} {:>8} {:>10} {:>8} {:>40}",
        "dataset", "n_syms", "escapes", "sym_codes", "esc_%", "len_histogram[1..8]"
    );
    println!("{}", "-".repeat(96));

    for (name, _raw, varbin) in &datasets {
        let stats = fsst_analyze(varbin);
        let total_codes = stats.escape_count + stats.symbol_count;
        let esc_pct = if total_codes > 0 {
            100.0 * stats.escape_count as f64 / total_codes as f64
        } else {
            0.0
        };

        let hist = stats.length_histogram;
        let hist_str = format!(
            "[{}, {}, {}, {}, {}, {}, {}, {}]",
            hist[0], hist[1], hist[2], hist[3], hist[4], hist[5], hist[6], hist[7]
        );

        println!(
            "{:<16} {:>8} {:>8} {:>10} {:>7.1}% {:>40}",
            name, stats.num_symbols, stats.escape_count, stats.symbol_count, esc_pct, hist_str,
        );
    }

    println!("{}", "=".repeat(96));
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

#[divan::bench]
fn compress_fsst_file_paths(bencher: Bencher) {
    let varbin = strings_to_varbin(&FILE_PATHS);
    let compressor = fsst_train_compressor(&varbin);
    bencher
        .with_inputs(|| (&varbin, &compressor))
        .bench_refs(|(v, c)| fsst_compress(*v, c));
}

#[divan::bench]
fn compress_fsst_csv(bencher: Bencher) {
    let varbin = strings_to_varbin(&CSV_ROWS);
    let compressor = fsst_train_compressor(&varbin);
    bencher
        .with_inputs(|| (&varbin, &compressor))
        .bench_refs(|(v, c)| fsst_compress(*v, c));
}

#[divan::bench]
fn compress_fsst_struct_binary(bencher: Bencher) {
    let varbin = bytes_to_varbin(&STRUCTURED_BINARY);
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

#[divan::bench]
fn train_fsst_file_paths(bencher: Bencher) {
    let varbin = strings_to_varbin(&FILE_PATHS);
    bencher
        .with_inputs(|| &varbin)
        .bench_refs(|v| fsst_train_compressor(v));
}

#[divan::bench]
fn train_fsst_csv(bencher: Bencher) {
    let varbin = strings_to_varbin(&CSV_ROWS);
    bencher
        .with_inputs(|| &varbin)
        .bench_refs(|v| fsst_train_compressor(v));
}

#[divan::bench]
fn train_fsst_struct_binary(bencher: Bencher) {
    let varbin = bytes_to_varbin(&STRUCTURED_BINARY);
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

#[divan::bench]
fn compress_zstd_file_paths(bencher: Bencher) {
    let data = concat_strings(&FILE_PATHS);
    bencher
        .with_inputs(|| data.as_slice())
        .bench_refs(|d| zstd::encode_all(*d, 3).unwrap());
}

#[divan::bench]
fn compress_zstd_csv(bencher: Bencher) {
    let data = concat_strings(&CSV_ROWS);
    bencher
        .with_inputs(|| data.as_slice())
        .bench_refs(|d| zstd::encode_all(*d, 3).unwrap());
}

#[divan::bench]
fn compress_zstd_struct_binary(bencher: Bencher) {
    let data = concat_bytes(&STRUCTURED_BINARY);
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

#[divan::bench]
fn compress_snappy_file_paths(bencher: Bencher) {
    let data = concat_strings(&FILE_PATHS);
    bencher
        .with_inputs(|| (snap::raw::Encoder::new(), data.as_slice()))
        .bench_refs(|(enc, d)| enc.compress_vec(d).unwrap());
}

#[divan::bench]
fn compress_snappy_csv(bencher: Bencher) {
    let data = concat_strings(&CSV_ROWS);
    bencher
        .with_inputs(|| (snap::raw::Encoder::new(), data.as_slice()))
        .bench_refs(|(enc, d)| enc.compress_vec(d).unwrap());
}

#[divan::bench]
fn compress_snappy_struct_binary(bencher: Bencher) {
    let data = concat_bytes(&STRUCTURED_BINARY);
    bencher
        .with_inputs(|| (snap::raw::Encoder::new(), data.as_slice()))
        .bench_refs(|(enc, d)| enc.compress_vec(d).unwrap());
}
