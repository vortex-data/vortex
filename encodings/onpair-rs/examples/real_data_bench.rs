// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_lossless,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::missing_panics_doc,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::print_stderr,
    clippy::print_stdout,
    clippy::use_debug,
    clippy::float_arithmetic,
    clippy::panic,
    clippy::clone_on_ref_ptr
)]
//
// Quick bench-style driver that runs `Column::compress` (single-shot) and
// `Column::compress_auto` on a parquet file, with optional row cap. Reports
// raw bytes, compressed bytes, ratio, train time, and decode throughput.
//
// Usage:
//   PARQUET=path/to.parquet COLUMN=URL ROWS=200000 cargo run --release \
//       -p onpair-lib --example real_data_bench
//
// If PARQUET is unset, falls back to the same synthetic ClickBench-shaped URL
// corpus the divan bench uses, so the example is always self-contained.

use std::env;
use std::fs::File;
use std::path::PathBuf;
use std::time::Instant;

use arrow_array::cast::AsArray;
use onpair_lib::{Column, OnPairTrainingConfig};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

fn load_parquet(path: &str, col_name: Option<&str>, cap: Option<usize>) -> Vec<Vec<u8>> {
    let file = File::open(path).expect("open parquet");
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).expect("parquet builder");
    let schema = builder.schema().clone();
    let picked = match col_name {
        Some(name) => schema
            .fields()
            .iter()
            .position(|f| f.name() == name)
            .expect("named column not found"),
        None => schema
            .fields()
            .iter()
            .position(|f| {
                use arrow_schema::DataType::*;
                matches!(
                    f.data_type(),
                    Utf8 | LargeUtf8 | Utf8View | Binary | LargeBinary | BinaryView
                )
            })
            .expect("no string/binary column"),
    };
    let f = schema.fields().get(picked).unwrap().clone();
    eprintln!(
        "[real_data] reading column #{picked} `{}` ({})",
        f.name(),
        f.data_type()
    );
    let mut rows: Vec<Vec<u8>> = Vec::new();
    let reader = builder.build().expect("parquet reader");
    'outer: for batch in reader.flatten() {
        let arr = batch.column(picked);
        use arrow_schema::DataType::*;
        match arr.data_type() {
            Utf8 => {
                for s in arr.as_string::<i32>().iter() {
                    rows.push(s.unwrap_or("").as_bytes().to_vec());
                    if let Some(c) = cap
                        && rows.len() >= c
                    {
                        break 'outer;
                    }
                }
            }
            LargeUtf8 => {
                for s in arr.as_string::<i64>().iter() {
                    rows.push(s.unwrap_or("").as_bytes().to_vec());
                    if let Some(c) = cap
                        && rows.len() >= c
                    {
                        break 'outer;
                    }
                }
            }
            Utf8View => {
                for s in arr.as_string_view().iter() {
                    rows.push(s.unwrap_or("").as_bytes().to_vec());
                    if let Some(c) = cap
                        && rows.len() >= c
                    {
                        break 'outer;
                    }
                }
            }
            Binary => {
                for s in arr.as_binary::<i32>().iter() {
                    rows.push(s.unwrap_or(&[]).to_vec());
                    if let Some(c) = cap
                        && rows.len() >= c
                    {
                        break 'outer;
                    }
                }
            }
            LargeBinary => {
                for s in arr.as_binary::<i64>().iter() {
                    rows.push(s.unwrap_or(&[]).to_vec());
                    if let Some(c) = cap
                        && rows.len() >= c
                    {
                        break 'outer;
                    }
                }
            }
            BinaryView => {
                for s in arr.as_binary_view().iter() {
                    rows.push(s.unwrap_or(&[]).to_vec());
                    if let Some(c) = cap
                        && rows.len() >= c
                    {
                        break 'outer;
                    }
                }
            }
            t => panic!("unsupported type: {t:?}"),
        }
    }
    rows
}

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
    "/", "/page", "/news", "/search?q=", "/profile", "/login", "/api/v1/data",
    "/static/asset.png", "/blog/post-", "/feed.xml", "/sitemap.xml", "/users/",
    "/admin/dashboard", "/categories/electronics", "/cart/checkout",
];
const TAILS: &[&str] = &["", "alpha", "beta", "gamma", "delta", "001", "002", "003"];

fn synthetic(n: usize) -> Vec<Vec<u8>> {
    let mut out = Vec::with_capacity(n);
    let mut x = 0x9E3779B97F4A7C15u64;
    for _ in 0..n {
        x = x.wrapping_add(0x9E3779B97F4A7C15);
        let h = HOSTS[(x as usize) % HOSTS.len()];
        let p = PATHS[((x >> 16) as usize) % PATHS.len()];
        let t = TAILS[((x >> 32) as usize) % TAILS.len()];
        let nn = (x >> 48) as u16;
        out.push(format!("{h}{p}{t}{nn}").into_bytes());
    }
    out
}

fn pack(rows: &[Vec<u8>]) -> (Vec<u8>, Vec<u64>) {
    let total: usize = rows.iter().map(|r| r.len()).sum();
    let mut bytes = Vec::with_capacity(total);
    let mut offsets = Vec::with_capacity(rows.len() + 1);
    offsets.push(0u64);
    for r in rows {
        bytes.extend_from_slice(r);
        offsets.push(bytes.len() as u64);
    }
    (bytes, offsets)
}

fn time<F: FnMut() -> R, R>(reps: usize, mut f: F) -> (R, std::time::Duration) {
    // Warm up.
    drop(f());
    let t0 = Instant::now();
    let mut last = None;
    for _ in 0..reps {
        last = Some(f());
    }
    (last.unwrap(), t0.elapsed() / reps as u32)
}

fn load_text(path: &str, cap: Option<usize>) -> Vec<Vec<u8>> {
    use std::io::{BufRead, BufReader};
    let f = File::open(path).expect("open text");
    let r = BufReader::new(f);
    let mut out = Vec::new();
    for line in r.lines() {
        let l = line.expect("readline");
        out.push(l.into_bytes());
        if let Some(c) = cap
            && out.len() >= c
        {
            break;
        }
    }
    out
}

fn main() {
    let parquet = env::var("PARQUET").ok();
    let text = env::var("TEXT").ok();
    let column = env::var("COLUMN").ok();
    let cap: Option<usize> = env::var("ROWS").ok().and_then(|s| s.parse().ok());
    let reps: usize = env::var("REPS").ok().and_then(|s| s.parse().ok()).unwrap_or(3);

    let (label, rows) = if let Some(p) = parquet.as_deref() {
        let r = load_parquet(p, column.as_deref(), cap);
        (
            format!(
                "{} col={}",
                PathBuf::from(p).file_name().unwrap().to_string_lossy(),
                column.as_deref().unwrap_or("<auto>")
            ),
            r,
        )
    } else if let Some(p) = text.as_deref() {
        let r = load_text(p, cap);
        (
            PathBuf::from(p).file_name().unwrap().to_string_lossy().to_string(),
            r,
        )
    } else {
        let n = cap.unwrap_or(100_000);
        ("synthetic_urls".to_string(), synthetic(n))
    };

    let (bytes, offsets) = pack(&rows);
    let raw = bytes.len();
    eprintln!(
        "[real_data] {} rows={} raw={} bytes ({:.2} MiB) reps={reps}",
        label,
        rows.len(),
        raw,
        raw as f64 / (1024.0 * 1024.0),
    );

    println!(
        "{:>16}  {:>5}  {:>5}  {:>10}  {:>10}  {:>8}  {:>10}  {:>11}",
        "config", "bits", "thr", "compr_KB", "ratio", "dict#", "train_ms", "throughput",
    );
    let raw_kb = raw / 1024;
    let bits_to_run: Vec<u32> = env::var("BITS_LIST")
        .ok()
        .map(|s| s.split(',').filter_map(|t| t.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![9, 10, 11, 12, 13, 14, 15, 16]);
    let thrs: Vec<f64> = env::var("THRESHOLDS")
        .ok()
        .map(|s| s.split(',').filter_map(|t| t.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![0.5]);
    for &bits in &bits_to_run {
        for &thr in &thrs {
            let cfg = OnPairTrainingConfig { bits, threshold: thr, seed: 42 };
            let (col, dt) = time(reps, || Column::compress(&bytes, &offsets, cfg).unwrap());
            let sz = col.compressed_size();
            let r = raw as f64 / sz as f64;
            let mb_s = (raw as f64 / (1024.0 * 1024.0)) / dt.as_secs_f64();
            println!(
                "{:>16}  {:>5}  {:>5.2}  {:>10}  {:>10.4}  {:>8}  {:>10.2}  {:>8.1} MiB/s",
                "single", bits, thr, sz / 1024, r, col.dict_size(),
                dt.as_secs_f64() * 1000.0, mb_s,
            );
        }
    }
    // Auto
    let (col, dt) = time(reps, || Column::compress_auto(&bytes, &offsets).unwrap());
    let sz = col.compressed_size();
    let r = raw as f64 / sz as f64;
    let mb_s = (raw as f64 / (1024.0 * 1024.0)) / dt.as_secs_f64();
    println!(
        "{:>16}  {:>5}  {:>5}  {:>10}  {:>10.4}  {:>8}  {:>10.2}  {:>8.1} MiB/s",
        "auto", col.bits(), "*", sz / 1024, r, col.dict_size(),
        dt.as_secs_f64() * 1000.0, mb_s,
    );
    // Thorough (auto + multi-seed at the chosen bit width).  Only run if
    // env var `THOROUGH=1` is set — otherwise the bench is too slow on
    // larger corpora.
    if env::var("THOROUGH").is_ok() {
        let (col, dt) = time(1.max(reps / 2), || Column::compress_thorough(&bytes, &offsets).unwrap());
        let sz = col.compressed_size();
        let r = raw as f64 / sz as f64;
        let mb_s = (raw as f64 / (1024.0 * 1024.0)) / dt.as_secs_f64();
        println!(
            "{:>16}  {:>5}  {:>5}  {:>10}  {:>10.4}  {:>8}  {:>10.2}  {:>8.1} MiB/s",
            "thorough", col.bits(), "*", sz / 1024, r, col.dict_size(),
            dt.as_secs_f64() * 1000.0, mb_s,
        );
    }
    let _ = raw_kb;
}
