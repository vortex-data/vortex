// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// Standalone driver for profiling `Column::compress` with samply or perf.
// Reproduces the synthetic ClickBench-shaped URL corpus used by
// benches/clickbench.rs and runs train+compress in a tight loop.

use std::env;
use std::hint::black_box;
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

fn main() {
    let bits: u32 = env::var("BITS").ok().and_then(|s| s.parse().ok()).unwrap_or(12);
    let iters: usize = env::var("ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(50);
    let rows = env::var("ROWS").ok().and_then(|s| s.parse().ok()).unwrap_or(100_000);
    let cfg = OnPairTrainingConfig { bits, threshold: 0.5, seed: 42 };

    let corpus = synthetic(rows);
    let (bytes, offsets) = pack(&corpus);
    eprintln!(
        "[profile_train] bits={bits} iters={iters} rows={} bytes={:.2} MiB",
        corpus.len(),
        bytes.len() as f64 / (1024.0 * 1024.0),
    );

    // Warm up.
    for _ in 0..3 {
        black_box(Column::compress(&bytes, &offsets, cfg).unwrap());
    }

    let t0 = Instant::now();
    for _ in 0..iters {
        let col = Column::compress(black_box(&bytes), black_box(&offsets), cfg).unwrap();
        black_box(col);
    }
    let elapsed = t0.elapsed();
    let per_iter = elapsed / iters as u32;
    let mb_per_s = (bytes.len() as f64 / (1024.0 * 1024.0)) / per_iter.as_secs_f64();
    eprintln!(
        "[profile_train] {iters} iters in {:?} ({:?} / iter, {mb_per_s:.1} MiB/s)",
        elapsed, per_iter
    );
}
