// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_lossless,
    clippy::many_single_char_names,
    clippy::missing_panics_doc,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::print_stderr,
    clippy::use_debug,
    clippy::cast_precision_loss
)]
//
// Standalone driver for profiling `Column::decode_all` and `kmp_automaton`
// scans with samply. Uses the same synthetic corpus as profile_train.rs.

use std::env;
use std::hint::black_box;
use std::time::Instant;

use onpair_lib::{Column, KmpAutomaton, OnPairTrainingConfig};

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
    let iters: usize = env::var("ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(2000);
    let rows = env::var("ROWS").ok().and_then(|s| s.parse().ok()).unwrap_or(100_000);
    let mode = env::var("MODE").unwrap_or_else(|_| "decode".to_string());

    let cfg = OnPairTrainingConfig { bits, threshold: 0.5, seed: 42 };
    let corpus = synthetic(rows);
    let (bytes, offsets) = pack(&corpus);
    let col = Column::compress(&bytes, &offsets, cfg).unwrap();
    eprintln!(
        "[profile_{mode}] bits={bits} iters={iters} rows={} bytes={} dict={}",
        col.len(),
        bytes.len(),
        col.dict_size(),
    );

    match mode.as_str() {
        "decode" => {
            for _ in 0..5 {
                black_box(col.decode_all());
            }
            let t0 = Instant::now();
            for _ in 0..iters {
                black_box(col.decode_all());
            }
            let dt = t0.elapsed();
            eprintln!("decode_all {iters}x in {:?} ({:?}/iter)", dt, dt / iters as u32);
        }
        "kmp" => {
            let dict = col.dictionary().clone();
            let needle: &[u8] = b"example";
            for _ in 0..5 {
                let aut = KmpAutomaton::new(needle, &dict);
                black_box(col.scan_bitmap(aut));
            }
            let t0 = Instant::now();
            for _ in 0..iters {
                let aut = KmpAutomaton::new(needle, &dict);
                black_box(col.scan_bitmap(aut));
            }
            let dt = t0.elapsed();
            eprintln!("kmp_automaton {iters}x in {:?} ({:?}/iter)", dt, dt / iters as u32);
        }
        other => {
            eprintln!("unknown MODE={other}; expected decode|kmp");
            std::process::exit(2);
        }
    }
}
