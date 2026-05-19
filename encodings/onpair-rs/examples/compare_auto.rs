// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_lossless,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::missing_panics_doc,
    clippy::unwrap_used,
    clippy::print_stdout,
    clippy::float_arithmetic
)]
// Quick sanity check: compress_auto vs single-config compression sizes.

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
    let rows = std::env::var("ROWS").ok().and_then(|s| s.parse().ok()).unwrap_or(100_000);
    let corpus = synthetic(rows);
    let (bytes, offsets) = pack(&corpus);
    let raw = bytes.len();
    println!("corpus: {} rows, {} bytes ({:.2} MiB)\n", rows, raw, raw as f64 / (1024.0 * 1024.0));

    println!("{:>20} {:>10} {:>10} {:>10}", "config", "size_KB", "ratio", "dict_size");
    for &bits in &[10u32, 11, 12, 14, 16] {
        let cfg = OnPairTrainingConfig { bits, threshold: 0.5, seed: 42 };
        let col = Column::compress(&bytes, &offsets, cfg).unwrap();
        let sz = col.compressed_size();
        let r = raw as f64 / sz as f64;
        println!("{:>20} {:>10} {:>10.4} {:>10}",
                 format!("bits={bits} thr=0.5"), sz/1024, r, col.dict_size());
    }
    println!();
    let auto = Column::compress_auto(&bytes, &offsets).unwrap();
    let sz = auto.compressed_size();
    let r = raw as f64 / sz as f64;
    println!("{:>20} {:>10} {:>10.4} {:>10}  bits={}",
             "compress_auto", sz/1024, r, auto.dict_size(), auto.bits());
}
