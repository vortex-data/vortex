// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#![allow(
    clippy::cast_possible_truncation,
    clippy::clone_on_ref_ptr,
    clippy::expect_used,
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::unwrap_in_result,
    clippy::unwrap_used,
    clippy::use_debug
)]
//
// Standalone microbenchmark for the OnPair compressor's two hot phases:
//   * dictionary training  (`trainer::train`)
//   * parse / encode        (`parser::parse`)
//
// Reports throughput (MiB/s) for each phase plus the achieved compression
// ratio, for bit widths 12 and 16. Designed to run without the C++ FFI crate
// so it can iterate quickly.
//
// Data source:
//   * env `ONPAIR_BENCH_PARQUET` (+ optional `ONPAIR_BENCH_COLUMN`) — read a
//     UTF-8 column from a parquet file (e.g. TPC-H lineitem `l_comment`).
//   * else a synthetic TPC-H-comment-shaped English corpus.
//
// Optional env:
//   * `ONPAIR_BENCH_MAX_BYTES` — cap the corpus at N bytes (default 1 GiB).
//   * `ONPAIR_BENCH_ITERS`     — timed iterations per phase (default 3).
//
// Run: cargo run --release -p vortex-onpair-rs --example bench_tpch

use std::env;
use std::fs::File;
use std::path::PathBuf;
use std::time::Instant;

use arrow_array::Array;
use arrow_array::cast::AsArray;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use vortex_onpair_rs::Column;
use vortex_onpair_rs::OnPairTrainingConfig;
use vortex_onpair_rs::Store;
use vortex_onpair_rs::TrainingConfig;
use vortex_onpair_rs::parse;
use vortex_onpair_rs::train;

const BITS: &[u8] = &[12, 16];

fn main() {
    let max_bytes = env::var("ONPAIR_BENCH_MAX_BYTES")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1 << 30);
    let iters = env::var("ONPAIR_BENCH_ITERS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(3);
    let threshold = env::var("ONPAIR_BENCH_THRESHOLD")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.2);

    let (source, bytes, offsets) = load_corpus(max_bytes);
    let n = offsets.len() - 1;
    let total = bytes.len();
    println!(
        "corpus: {source}\n  rows = {n}, bytes = {:.2} MiB",
        total as f64 / (1024.0 * 1024.0)
    );

    let off32: Vec<u32> = offsets.iter().map(|&o| o as u32).collect();

    for &bits in BITS {
        println!("\n=== bits = {bits} ===");
        let cfg = TrainingConfig::from(OnPairTrainingConfig {
            bits: bits as u32,
            threshold,
            seed: 42,
        });

        // ── train ──────────────────────────────────────────────────────────
        let mut train_secs = f64::MAX;
        let mut result = train(&bytes, &off32, n, &cfg);
        for _ in 0..iters {
            let t = Instant::now();
            result = train(&bytes, &off32, n, &cfg);
            train_secs = train_secs.min(t.elapsed().as_secs_f64());
        }

        if env::var("ONPAIR_BENCH_HISTO").is_ok() {
            let mut hist = [0usize; 17];
            for i in 0..result.dict.num_tokens() {
                hist[result.dict.token_size(i as u16)] += 1;
            }
            let long: usize = hist[9..].iter().sum();
            eprintln!("  [histo] len>8 tokens = {long}; by len = {hist:?}");
        }

        // ── parse ──────────────────────────────────────────────────────────
        let mut parse_secs = f64::MAX;
        let mut store = Store::default();
        for _ in 0..iters {
            let t = Instant::now();
            parse(&bytes, &off32, n, &result.lpm, bits, &mut store);
            parse_secs = parse_secs.min(t.elapsed().as_secs_f64());
        }

        // ── ratio ────────────────────────────────────────────────────────────
        let dict_bytes = *result.dict.offsets.last().unwrap() as usize;
        let dict_offsets = result.dict.offsets.len() * 4;
        let packed_bytes = (store.num_tokens() * bits as usize).div_ceil(8);
        let boundaries = store.boundaries.len() * 4;
        let compressed = dict_bytes + dict_offsets + packed_bytes + boundaries;

        let mib = total as f64 / (1024.0 * 1024.0);
        println!("  train: {:.3}s  {:.1} MiB/s", train_secs, mib / train_secs);
        println!("  parse: {:.3}s  {:.1} MiB/s", parse_secs, mib / parse_secs);
        println!(
            "  total: {:.3}s  {:.1} MiB/s",
            train_secs + parse_secs,
            mib / (train_secs + parse_secs)
        );
        println!(
            "  dict tokens = {}, dict bytes = {}, packed = {} bytes, tokens = {}",
            result.dict.num_tokens(),
            dict_bytes,
            packed_bytes,
            store.num_tokens()
        );
        println!(
            "  compressed = {:.2} MiB, ratio = {:.3}x",
            compressed as f64 / (1024.0 * 1024.0),
            total as f64 / compressed as f64
        );

        // ── correctness: full train+parse+decode roundtrip on this corpus ───
        let col = Column::compress(
            &bytes,
            &offsets,
            OnPairTrainingConfig {
                bits: bits as u32,
                threshold,
                seed: 42,
            },
        )
        .unwrap();
        let mut decode_secs = f64::MAX;
        for _ in 0..iters {
            let t = Instant::now();
            let _d = col.decode_all();
            decode_secs = decode_secs.min(t.elapsed().as_secs_f64());
        }
        println!(
            "  decode: {:.3}s  {:.1} MiB/s",
            decode_secs,
            mib / decode_secs
        );
        let (dbytes, doffsets) = col.decode_all();
        let bytes_ok = dbytes == bytes;
        let offsets_ok = doffsets == off32;
        println!(
            "  roundtrip: {} (bytes_match={bytes_ok}, offsets_match={offsets_ok}, decoded={} MiB)",
            if bytes_ok && offsets_ok {
                "PASS"
            } else {
                "FAIL"
            },
            dbytes.len() as f64 / (1024.0 * 1024.0),
        );
        assert!(bytes_ok && offsets_ok, "roundtrip mismatch at bits={bits}");

        if env::var("ONPAIR_BENCH_CPP").is_ok() {
            // Apples-to-apples: time Rust's full Column::compress (which, like
            // the C++ shim, repacks u64->u32 offsets and builds the column)
            // against the C++ Column::compress.
            let rcfg = OnPairTrainingConfig {
                bits: bits as u32,
                threshold,
                seed: 42,
            };
            let mut rsecs = f64::MAX;
            for _ in 0..iters {
                let t = Instant::now();
                let c = Column::compress(&bytes, &offsets, rcfg).unwrap();
                rsecs = rsecs.min(t.elapsed().as_secs_f64());
                std::hint::black_box(&c);
            }
            println!(
                "  [Rust  Column::compress] {:.3}s  {:.1} MiB/s",
                rsecs,
                mib / rsecs
            );
            cpp_compare(&bytes, &offsets, bits, threshold, mib, iters);
        }
    }
}

fn cpp_compare(bytes: &[u8], offsets: &[u64], bits: u8, threshold: f64, mib: f64, iters: usize) {
    use vortex_onpair_sys::Column as CppColumn;
    use vortex_onpair_sys::OnPairTrainingConfig as CppCfg;
    let cfg = CppCfg {
        bits: bits as u32,
        threshold,
        seed: 42,
    };
    let mut secs = f64::MAX;
    let mut col = CppColumn::compress(bytes, offsets, cfg).unwrap();
    for _ in 0..iters {
        let t = Instant::now();
        col = CppColumn::compress(bytes, offsets, cfg).unwrap();
        secs = secs.min(t.elapsed().as_secs_f64());
    }
    let parts = col.parts().unwrap();
    let packed_bytes = parts.codes_packed.len() * 8;
    let compressed = parts.dict_bytes.len()
        + parts.dict_offsets.len() * 4
        + packed_bytes
        + parts.codes_boundaries.len() * 4;
    println!(
        "  [C++] compress: {:.3}s  {:.1} MiB/s | dict tokens = {}, ratio = {:.3}x",
        secs,
        mib / secs,
        parts.dict_offsets.len().saturating_sub(1),
        bytes.len() as f64 / compressed as f64
    );
}

fn load_corpus(max_bytes: usize) -> (String, Vec<u8>, Vec<u64>) {
    if let Ok(path) = env::var("ONPAIR_BENCH_PARQUET")
        && let Some((bytes, offsets)) = read_parquet(&PathBuf::from(&path), max_bytes)
    {
        return (format!("{path} (parquet)"), bytes, offsets);
    }
    let (bytes, offsets) = synthetic(max_bytes);
    (
        "synthetic TPC-H-comment-shaped corpus".to_string(),
        bytes,
        offsets,
    )
}

fn read_parquet(path: &PathBuf, max_bytes: usize) -> Option<(Vec<u8>, Vec<u64>)> {
    let file = File::open(path).ok()?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).ok()?;
    let schema = builder.schema().clone();
    let col_name = env::var("ONPAIR_BENCH_COLUMN").ok();
    let picked = match col_name.as_deref() {
        Some(name) => schema.fields().iter().position(|f| f.name() == name)?,
        None => schema.fields().iter().position(|f| {
            use arrow_schema::DataType::*;
            matches!(f.data_type(), Utf8 | LargeUtf8 | Utf8View)
        })?,
    };
    eprintln!(
        "[bench] column #{picked} `{}`",
        schema.fields().get(picked).unwrap().name()
    );

    let mut bytes = Vec::new();
    let mut offsets = vec![0u64];
    let reader = builder.build().ok()?;
    'outer: for batch in reader.flatten() {
        let arr = batch.column(picked);
        use arrow_schema::DataType::*;
        macro_rules! push_iter {
            ($it:expr) => {
                for s in $it {
                    let b = s.unwrap_or("").as_bytes();
                    bytes.extend_from_slice(b);
                    offsets.push(bytes.len() as u64);
                    if bytes.len() >= max_bytes {
                        break 'outer;
                    }
                }
            };
        }
        match arr.data_type() {
            Utf8 => push_iter!(arr.as_string::<i32>().iter()),
            LargeUtf8 => push_iter!(arr.as_string::<i64>().iter()),
            Utf8View => push_iter!(arr.as_string_view().iter()),
            _ => return None,
        }
    }
    Some((bytes, offsets))
}

/// Build a corpus shaped like TPC-H `l_comment`: short phrases of common
/// English words separated by spaces, ~27 bytes each, with heavy word reuse.
fn synthetic(max_bytes: usize) -> (Vec<u8>, Vec<u64>) {
    const WORDS: &[&str] = &[
        "the",
        "quickly",
        "final",
        "regular",
        "ironic",
        "express",
        "packages",
        "accounts",
        "deposits",
        "foxes",
        "requests",
        "blithely",
        "carefully",
        "furiously",
        "slyly",
        "pending",
        "unusual",
        "even",
        "bold",
        "silent",
        "theodolites",
        "instructions",
        "asymptotes",
        "across",
        "above",
        "after",
        "among",
        "around",
        "thinly",
        "sometimes",
        "boldly",
        "fluffily",
    ];
    let mut bytes = Vec::with_capacity(max_bytes.min(1 << 28));
    let mut offsets = vec![0u64];
    let mut x = 0x9E3779B97F4A7C15u64;
    while bytes.len() < max_bytes {
        let nwords = 3 + (x >> 60) as usize % 7;
        let start = bytes.len();
        for w in 0..nwords {
            x = x.wrapping_add(0x9E3779B97F4A7C15);
            x ^= x >> 31;
            if w > 0 {
                bytes.push(b' ');
            }
            bytes.extend_from_slice(WORDS[(x as usize) % WORDS.len()].as_bytes());
        }
        if bytes.len() == start {
            bytes.push(b' ');
        }
        offsets.push(bytes.len() as u64);
    }
    (bytes, offsets)
}
