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
    clippy::float_arithmetic
)]
//
// Sweep training parameters and report compression ratio + train time for
// each. Identifies (1) what the *single-shot* compression on a fixed seed
// produces and (2) whether trying multiple seeds yields a better dictionary.

use std::env;
use std::time::Instant;

use onpair_lib::{Column, OnPairTrainingConfig};

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
        let n = (x >> 48) as u16;
        out.push(format!("{h}{p}{t}{n}").into_bytes());
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

#[derive(Debug, Clone, Copy)]
struct Sized {
    dict_bytes: usize,
    dict_offsets_bytes: usize,
    codes_packed_bytes: usize,
    codes_boundaries_bytes: usize,
}

impl Sized {
    fn total(self) -> usize {
        self.dict_bytes
            + self.dict_offsets_bytes
            + self.codes_packed_bytes
            + self.codes_boundaries_bytes
    }
}

fn measure(col: &Column) -> Sized {
    let p = col.parts().expect("parts");
    Sized {
        dict_bytes: p.dict_bytes.len(),
        dict_offsets_bytes: size_of_val(p.dict_offsets),
        codes_packed_bytes: size_of_val(p.codes_packed),
        codes_boundaries_bytes: size_of_val(p.codes_boundaries),
    }
}

fn main() {
    let rows = env::var("ROWS").ok().and_then(|s| s.parse().ok()).unwrap_or(100_000);
    let seeds: Vec<u64> = match env::var("SEEDS") {
        Ok(s) => s.split(',').filter_map(|t| t.trim().parse().ok()).collect(),
        Err(_) => vec![42],
    };
    let bits_list: Vec<u32> = match env::var("BITS_LIST") {
        Ok(s) => s.split(',').filter_map(|t| t.trim().parse().ok()).collect(),
        Err(_) => vec![12, 14, 16],
    };
    let thresholds: Vec<f64> = match env::var("THRESHOLDS") {
        Ok(s) => s.split(',').filter_map(|t| t.trim().parse().ok()).collect(),
        Err(_) => vec![0.3, 0.5, 0.7],
    };

    let corpus = synthetic(rows);
    let (bytes, offsets) = pack(&corpus);
    let raw = bytes.len();
    eprintln!(
        "[sweep] corpus: {} rows, {} bytes ({:.2} MiB)",
        corpus.len(),
        raw,
        raw as f64 / (1024.0 * 1024.0),
    );
    eprintln!(
        "[sweep] grid: bits={:?} thresholds={:?} seeds={:?} = {} runs\n",
        bits_list,
        thresholds,
        seeds,
        bits_list.len() * thresholds.len() * seeds.len(),
    );

    println!(
        "{:>4} {:>6} {:>6} {:>10} {:>10} {:>10} {:>10} {:>10} {:>7} {:>8}",
        "bits", "thr", "seed",
        "raw_KB", "compr_KB", "dict_KB", "codes_KB", "bound_KB",
        "ratio", "train_ms",
    );

    // Track best-per-(bits, threshold) and best-per-bits.
    use std::collections::BTreeMap;
    let mut best_per_bits: BTreeMap<u32, (usize, u64, f64)> = BTreeMap::new(); // bits -> (size, seed, thr)
    let mut best_per_grid: BTreeMap<(u32, u64), (usize, u64, f64)> = BTreeMap::new();

    for &bits in &bits_list {
        for &thr in &thresholds {
            for &seed in &seeds {
                let cfg = OnPairTrainingConfig { bits, threshold: thr, seed };
                let t0 = Instant::now();
                let col = Column::compress(&bytes, &offsets, cfg).expect("compress");
                let dt = t0.elapsed();
                let s = measure(&col);
                let total = s.total();
                let ratio = raw as f64 / total as f64;
                println!(
                    "{:>4} {:>6.2} {:>6} {:>10} {:>10} {:>10} {:>10} {:>10} {:>7.3} {:>8.2}",
                    bits, thr, seed,
                    raw / 1024,
                    total / 1024,
                    s.dict_bytes / 1024,
                    s.codes_packed_bytes / 1024,
                    s.codes_boundaries_bytes / 1024,
                    ratio,
                    dt.as_secs_f64() * 1000.0,
                );
                let key = (bits, thr.to_bits());
                best_per_grid
                    .entry((bits, thr.to_bits()))
                    .and_modify(|e| {
                        if total < e.0 {
                            *e = (total, seed, thr);
                        }
                    })
                    .or_insert((total, seed, thr));
                best_per_bits
                    .entry(bits)
                    .and_modify(|e| {
                        if total < e.0 {
                            *e = (total, seed, thr);
                        }
                    })
                    .or_insert((total, seed, thr));
                let _ = key;
            }
        }
    }

    println!("\n=== best per bits (raw = {} bytes) ===", raw);
    println!("{:>4} {:>12} {:>10} {:>8} {:>10}", "bits", "best_bytes", "ratio", "seed", "threshold");
    for (&bits, &(size, seed, thr)) in &best_per_bits {
        let ratio = raw as f64 / size as f64;
        println!("{:>4} {:>12} {:>10.4} {:>8} {:>10.3}", bits, size, ratio, seed, thr);
    }
}
