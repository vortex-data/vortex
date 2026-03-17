// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Comprehensive benchmarks comparing FSST, FSST-12, Zstd, and Snappy
//! across 10 diverse string datasets.
//!
//! Measures compression ratio, compression throughput, and decompression throughput.

#![allow(clippy::unwrap_used, clippy::cast_possible_truncation)]

use std::sync::LazyLock;

use divan::Bencher;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::arrays::VarBinArray;
use vortex_array::compute::warm_up_vtables;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_fsst::fsst_train_compressor;
use vortex_fsst::fsst12;
use vortex_fsst::test_utils;

fn main() {
    warm_up_vtables();
    // Print compression ratios for all datasets before running benchmarks
    print_compression_summary();
    divan::main();
}

// ---------------------------------------------------------------------------
// 10 Diverse datasets
// ---------------------------------------------------------------------------

/// 1. Short emails (~25 bytes avg)
fn gen_emails(n: usize) -> Vec<String> {
    test_utils::generate_emails(n)
}

/// 2. Medium URLs (~50 bytes avg)
fn gen_urls(n: usize) -> Vec<String> {
    test_utils::generate_short_urls(n)
}

/// 3. Long log lines (~150 bytes avg)
fn gen_logs(n: usize) -> Vec<String> {
    test_utils::generate_log_lines(n)
}

/// 4. Highly repetitive JSON (~80 bytes avg, template-based)
fn gen_json(n: usize) -> Vec<String> {
    test_utils::generate_json_strings(n)
}

/// 5. File paths (~40 bytes avg, hierarchical)
fn gen_paths(n: usize) -> Vec<String> {
    test_utils::generate_file_paths(n)
}

/// 6. ClickBench URLs (~100 bytes avg, long with query params)
fn gen_clickbench_urls(n: usize) -> Vec<String> {
    test_utils::generate_clickbench_urls(n)
}

/// 7. UUIDs - high cardinality, low compressibility
fn gen_uuids(n: usize) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(777);
    (0..n)
        .map(|_| {
            let bytes: [u8; 16] = rng.random();
            format!(
                "{:08x}-{:04x}-{:04x}-{:04x}-{:012x}",
                u32::from_le_bytes(bytes[0..4].try_into().unwrap()),
                u16::from_le_bytes(bytes[4..6].try_into().unwrap()),
                u16::from_le_bytes(bytes[6..8].try_into().unwrap()),
                u16::from_le_bytes(bytes[8..10].try_into().unwrap()),
                u64::from_le_bytes({
                    let mut buf = [0u8; 8];
                    buf[..6].copy_from_slice(&bytes[10..16]);
                    buf
                }),
            )
        })
        .collect()
}

/// 8. Enum-like status strings - very low cardinality
fn gen_status_strings(n: usize) -> Vec<String> {
    let statuses = [
        "PENDING",
        "ACTIVE",
        "COMPLETED",
        "FAILED",
        "CANCELLED",
        "IN_PROGRESS",
        "WAITING_FOR_APPROVAL",
        "ARCHIVED",
    ];
    let mut rng = StdRng::seed_from_u64(888);
    (0..n)
        .map(|_| statuses[rng.random_range(0..statuses.len())].to_string())
        .collect()
}

/// 9. Natural language (English-like sentences)
fn gen_english_text(n: usize) -> Vec<String> {
    let subjects = [
        "The quick brown fox",
        "A lazy dog",
        "The system administrator",
        "An unexpected error",
        "The database connection",
        "A new user",
        "The API endpoint",
        "Our monitoring system",
    ];
    let verbs = [
        "jumped over",
        "encountered",
        "processed",
        "failed to connect to",
        "successfully completed",
        "was unable to handle",
        "quickly resolved",
        "reported issues with",
    ];
    let objects = [
        "the production server",
        "multiple requests",
        "the authentication module",
        "the network interface",
        "several database queries",
        "the configuration file",
        "incoming traffic spikes",
        "the deployment pipeline",
    ];
    let mut rng = StdRng::seed_from_u64(999);
    (0..n)
        .map(|_| {
            format!(
                "{} {} {} at {:02}:{:02}:{:02}.",
                subjects[rng.random_range(0..subjects.len())],
                verbs[rng.random_range(0..verbs.len())],
                objects[rng.random_range(0..objects.len())],
                rng.random_range(0..24u32),
                rng.random_range(0..60u32),
                rng.random_range(0..60u32),
            )
        })
        .collect()
}

/// 10. Base64-encoded data - high entropy, challenging for pattern matchers
fn gen_base64(n: usize) -> Vec<String> {
    let charset = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut rng = StdRng::seed_from_u64(1010);
    (0..n)
        .map(|_| {
            let len = rng.random_range(20..60);
            let s: String = (0..len)
                .map(|_| charset[rng.random_range(0..charset.len())] as char)
                .collect();
            format!("{s}==")
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Dataset wrapper
// ---------------------------------------------------------------------------

const NUM_STRINGS: usize = 50_000;

struct Dataset {
    name: &'static str,
    raw_bytes: Vec<Vec<u8>>,
    total_raw_size: usize,
}

impl Dataset {
    fn new(name: &'static str, strings: Vec<String>) -> Self {
        let raw_bytes: Vec<Vec<u8>> = strings.into_iter().map(|s| s.into_bytes()).collect();
        let total_raw_size: usize = raw_bytes.iter().map(|b| b.len()).sum();
        Self {
            name,
            raw_bytes,
            total_raw_size,
        }
    }

    fn as_refs(&self) -> Vec<&[u8]> {
        self.raw_bytes.iter().map(|v| v.as_slice()).collect()
    }

    fn to_varbin(&self) -> VarBinArray {
        VarBinArray::from_iter(
            self.raw_bytes
                .iter()
                .map(|s| Some(s.clone().into_boxed_slice())),
            DType::Utf8(Nullability::NonNullable),
        )
    }
}

static DATASETS: LazyLock<Vec<Dataset>> = LazyLock::new(|| {
    vec![
        Dataset::new("emails", gen_emails(NUM_STRINGS)),
        Dataset::new("urls", gen_urls(NUM_STRINGS)),
        Dataset::new("logs", gen_logs(NUM_STRINGS)),
        Dataset::new("json", gen_json(NUM_STRINGS)),
        Dataset::new("paths", gen_paths(NUM_STRINGS)),
        Dataset::new("clickbench_urls", gen_clickbench_urls(NUM_STRINGS)),
        Dataset::new("uuids", gen_uuids(NUM_STRINGS)),
        Dataset::new("status_strings", gen_status_strings(NUM_STRINGS)),
        Dataset::new("english_text", gen_english_text(NUM_STRINGS)),
        Dataset::new("base64", gen_base64(NUM_STRINGS)),
    ]
});

// ---------------------------------------------------------------------------
// Compression summary (printed before benchmarks)
// ---------------------------------------------------------------------------

fn print_compression_summary() {
    println!("\n{}", "=".repeat(80));
    println!("COMPRESSION RATIO SUMMARY (compressed/original, lower is better)");
    println!(
        "{:>20} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Dataset", "RawSize", "FSST", "FSST-12", "Zstd", "Snappy"
    );
    println!("{}", "-".repeat(80));

    for ds in DATASETS.iter() {
        let raw_size = ds.total_raw_size;

        // FSST
        let varbin = ds.to_varbin();
        let compressor = fsst_train_compressor(&varbin);
        let mut fsst_size = 0;
        for bytes in &ds.raw_bytes {
            fsst_size += compressor.compress(bytes).len();
        }
        // Add symbol table overhead
        fsst_size += compressor.symbol_table().len() * 8 + compressor.symbol_lengths().len();

        // FSST-12
        let refs = ds.as_refs();
        let compressor12 = fsst12::Compressor12::train(&refs);
        let mut fsst12_size = 0;
        for bytes in &ds.raw_bytes {
            fsst12_size += compressor12.compress(bytes).len();
        }
        // Add symbol table overhead
        fsst12_size += compressor12.symbols().len() * 9; // 8 bytes value + 1 byte len

        // Zstd (level 3)
        let all_data: Vec<u8> = ds
            .raw_bytes
            .iter()
            .flat_map(|b| {
                let len = (b.len() as u32).to_le_bytes();
                len.iter()
                    .copied()
                    .chain(b.iter().copied())
                    .collect::<Vec<u8>>()
            })
            .collect();
        let zstd_compressed = zstd::bulk::compress(&all_data, 3).unwrap();
        let zstd_size = zstd_compressed.len();

        // Snappy
        let mut snappy_encoder = snap::raw::Encoder::new();
        let snappy_compressed = snappy_encoder.compress_vec(&all_data).unwrap();
        let snappy_size = snappy_compressed.len();

        println!(
            "{:>20} {:>9} {:>9.3} {:>9.3} {:>9.3} {:>9.3}",
            ds.name,
            format_size(raw_size),
            fsst_size as f64 / raw_size as f64,
            fsst12_size as f64 / raw_size as f64,
            zstd_size as f64 / raw_size as f64,
            snappy_size as f64 / raw_size as f64,
        );
    }
    println!("{}", "=".repeat(80));
    println!();
}

fn format_size(bytes: usize) -> String {
    if bytes >= 1024 * 1024 {
        format!("{:.1}MB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.1}KB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes}B")
    }
}

// ---------------------------------------------------------------------------
// Dataset index for divan parametrization
// ---------------------------------------------------------------------------

const DATASET_INDICES: &[usize] = &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
/// Dataset names for reference (indices correspond to DATASET_INDICES).
#[allow(dead_code)]
const DATASET_NAMES: &[&str] = &[
    "emails",
    "urls",
    "logs",
    "json",
    "paths",
    "clickbench_urls",
    "uuids",
    "status_strings",
    "english_text",
    "base64",
];

// ---------------------------------------------------------------------------
// FSST benchmarks
// ---------------------------------------------------------------------------

#[divan::bench(args = DATASET_INDICES)]
fn fsst_compress(bencher: Bencher, idx: usize) {
    let ds = &DATASETS[idx];
    let varbin = ds.to_varbin();
    let compressor = fsst_train_compressor(&varbin);

    bencher
        .counter(divan::counter::BytesCount::of_iter(
            ds.raw_bytes.iter().map(|b| b.len()),
        ))
        .with_inputs(|| ds.raw_bytes.clone())
        .bench_refs(|data| {
            let mut total = 0;
            for bytes in data.iter() {
                total += compressor.compress(bytes).len();
            }
            total
        });
}

#[divan::bench(args = DATASET_INDICES)]
fn fsst_decompress(bencher: Bencher, idx: usize) {
    let ds = &DATASETS[idx];
    let varbin = ds.to_varbin();
    let compressor = fsst_train_compressor(&varbin);
    let decompressor = compressor.decompressor();
    let compressed: Vec<Vec<u8>> = ds
        .raw_bytes
        .iter()
        .map(|b| compressor.compress(b))
        .collect();

    bencher
        .counter(divan::counter::BytesCount::of_iter(
            ds.raw_bytes.iter().map(|b| b.len()),
        ))
        .with_inputs(|| compressed.clone())
        .bench_refs(|data| {
            let mut total = 0;
            for c in data.iter() {
                total += decompressor.decompress(c).len();
            }
            total
        });
}

// ---------------------------------------------------------------------------
// FSST-12 benchmarks
// ---------------------------------------------------------------------------

#[divan::bench(args = DATASET_INDICES)]
fn fsst12_compress(bencher: Bencher, idx: usize) {
    let ds = &DATASETS[idx];
    let refs = ds.as_refs();
    let compressor = fsst12::Compressor12::train(&refs);

    bencher
        .counter(divan::counter::BytesCount::of_iter(
            ds.raw_bytes.iter().map(|b| b.len()),
        ))
        .with_inputs(|| ds.raw_bytes.clone())
        .bench_refs(|data| {
            let mut total = 0;
            for bytes in data.iter() {
                total += compressor.compress(bytes).len();
            }
            total
        });
}

#[divan::bench(args = DATASET_INDICES)]
fn fsst12_decompress(bencher: Bencher, idx: usize) {
    let ds = &DATASETS[idx];
    let refs = ds.as_refs();
    let compressor = fsst12::Compressor12::train(&refs);
    let decompressor = compressor.decompressor();
    let compressed: Vec<Vec<u8>> = ds
        .raw_bytes
        .iter()
        .map(|b| compressor.compress(b))
        .collect();

    bencher
        .counter(divan::counter::BytesCount::of_iter(
            ds.raw_bytes.iter().map(|b| b.len()),
        ))
        .with_inputs(|| compressed.clone())
        .bench_refs(|data| {
            let mut total = 0;
            for c in data.iter() {
                total += decompressor.decompress(c).len();
            }
            total
        });
}

// ---------------------------------------------------------------------------
// Zstd benchmarks
// ---------------------------------------------------------------------------

#[divan::bench(args = DATASET_INDICES)]
fn zstd_compress(bencher: Bencher, idx: usize) {
    let ds = &DATASETS[idx];
    // Serialize all strings with length prefix for fair comparison
    let all_data: Vec<u8> = ds
        .raw_bytes
        .iter()
        .flat_map(|b| {
            let len = (b.len() as u32).to_le_bytes();
            len.iter()
                .copied()
                .chain(b.iter().copied())
                .collect::<Vec<u8>>()
        })
        .collect();

    bencher
        .counter(divan::counter::BytesCount::new(all_data.len()))
        .with_inputs(|| all_data.clone())
        .bench_refs(|data| zstd::bulk::compress(data, 3).unwrap());
}

#[divan::bench(args = DATASET_INDICES)]
fn zstd_decompress(bencher: Bencher, idx: usize) {
    let ds = &DATASETS[idx];
    let all_data: Vec<u8> = ds
        .raw_bytes
        .iter()
        .flat_map(|b| {
            let len = (b.len() as u32).to_le_bytes();
            len.iter()
                .copied()
                .chain(b.iter().copied())
                .collect::<Vec<u8>>()
        })
        .collect();
    let compressed = zstd::bulk::compress(&all_data, 3).unwrap();

    bencher
        .counter(divan::counter::BytesCount::new(all_data.len()))
        .with_inputs(|| compressed.clone())
        .bench_refs(|data| zstd::bulk::decompress(data, all_data.len() * 2).unwrap());
}

// ---------------------------------------------------------------------------
// Snappy benchmarks
// ---------------------------------------------------------------------------

#[divan::bench(args = DATASET_INDICES)]
fn snappy_compress(bencher: Bencher, idx: usize) {
    let ds = &DATASETS[idx];
    let all_data: Vec<u8> = ds
        .raw_bytes
        .iter()
        .flat_map(|b| {
            let len = (b.len() as u32).to_le_bytes();
            len.iter()
                .copied()
                .chain(b.iter().copied())
                .collect::<Vec<u8>>()
        })
        .collect();

    bencher
        .counter(divan::counter::BytesCount::new(all_data.len()))
        .with_inputs(|| all_data.clone())
        .bench_refs(|data| {
            let mut encoder = snap::raw::Encoder::new();
            encoder.compress_vec(data).unwrap()
        });
}

#[divan::bench(args = DATASET_INDICES)]
fn snappy_decompress(bencher: Bencher, idx: usize) {
    let ds = &DATASETS[idx];
    let all_data: Vec<u8> = ds
        .raw_bytes
        .iter()
        .flat_map(|b| {
            let len = (b.len() as u32).to_le_bytes();
            len.iter()
                .copied()
                .chain(b.iter().copied())
                .collect::<Vec<u8>>()
        })
        .collect();
    let mut encoder = snap::raw::Encoder::new();
    let compressed = encoder.compress_vec(&all_data).unwrap();

    bencher
        .counter(divan::counter::BytesCount::new(all_data.len()))
        .with_inputs(|| compressed.clone())
        .bench_refs(|data| {
            let mut decoder = snap::raw::Decoder::new();
            decoder.decompress_vec(data).unwrap()
        });
}

// ---------------------------------------------------------------------------
// FSST-12 training benchmarks
// ---------------------------------------------------------------------------

#[divan::bench(args = DATASET_INDICES)]
fn fsst12_train(bencher: Bencher, idx: usize) {
    let ds = &DATASETS[idx];
    let refs = ds.as_refs();

    bencher
        .with_inputs(|| refs.clone())
        .bench_refs(|data| fsst12::Compressor12::train(data));
}

#[divan::bench(args = DATASET_INDICES)]
fn fsst_train(bencher: Bencher, idx: usize) {
    let ds = &DATASETS[idx];
    let varbin = ds.to_varbin();

    bencher
        .with_inputs(|| &varbin)
        .bench_refs(|vb| fsst_train_compressor(vb));
}
