// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Empirical study: does OnPair + token-space block front-coding compress
//! lex-sorted string columns better than the standard baselines?
//!
//! Usage:
//!     onpair-bench tpch_l_comment [rows]
//!     onpair-bench clickbench_url [rows]
//!     onpair-bench clickbench_title [rows]
//!     onpair-bench all [rows]

mod compare_fused;
mod datasets;
mod encoders;
mod frontcode;
mod sort_bench;

use std::env;
use std::fs;
use std::time::Instant;

use anyhow::Result;

use crate::encoders::bytes_front_coded;
use crate::encoders::fsst_size;
use crate::encoders::onpair_compress;
use crate::encoders::onpair_front_coded;
use crate::encoders::onpair_size;
use crate::encoders::raw_size;
use crate::encoders::zstd_block;
use crate::encoders::zstd_monolithic;

#[derive(Debug, Clone)]
struct Row {
    encoding: String,
    bytes: usize,
    ratio: f64,
    bits_per_row: f64,
    encode_ms: u128,
}

fn run_on(name: &str, mut rows: Vec<Vec<u8>>) -> Result<Vec<Row>> {
    if rows.is_empty() {
        anyhow::bail!("no rows loaded for {name}");
    }
    let n = rows.len();
    let raw_bytes: usize = rows.iter().map(|r| r.len()).sum();
    eprintln!(
        "[{name}] n={n} raw={:.2} MiB avg_len={:.1}",
        raw_bytes as f64 / 1024.0 / 1024.0,
        raw_bytes as f64 / n as f64,
    );

    // Sort lexicographically. This is the scenario the user asked about.
    let t = Instant::now();
    rows.sort();
    eprintln!("[{name}] lex sort: {} ms", t.elapsed().as_millis());

    let baseline = raw_size(&rows) as f64;
    let mk = |label: &str, bytes: usize, ms: u128| Row {
        encoding: label.to_string(),
        bytes,
        ratio: baseline / bytes as f64,
        bits_per_row: bytes as f64 * 8.0 / n as f64,
        encode_ms: ms,
    };

    let mut results = Vec::new();
    results.push(mk("raw (sorted)", raw_size(&rows), 0));

    let t = Instant::now();
    let s = zstd_monolithic(&rows, 3)?;
    results.push(mk("zstd-3 monolithic", s, t.elapsed().as_millis()));

    let t = Instant::now();
    let s = zstd_monolithic(&rows, 9)?;
    results.push(mk("zstd-9 monolithic", s, t.elapsed().as_millis()));

    let t = Instant::now();
    let s = zstd_block(&rows, 1024, 3)?;
    results.push(mk("zstd-3 block-1024", s, t.elapsed().as_millis()));

    let t = Instant::now();
    let s = fsst_size(&rows);
    results.push(mk("fsst", s, t.elapsed().as_millis()));

    let t = Instant::now();
    let s = bytes_front_coded(&rows, 256);
    results.push(mk("byte front-code 256", s, t.elapsed().as_millis()));

    let t = Instant::now();
    let out = onpair_compress(&rows, 12)?;
    let compress_ms = t.elapsed().as_millis();
    let s = onpair_size(&out)?;
    results.push(mk("onpair (12-bit)", s, compress_ms));

    let s = onpair_front_coded(&out, 64)?;
    results.push(mk("onpair + front-code 64", s, compress_ms));

    let s = onpair_front_coded(&out, 256)?;
    results.push(mk("onpair + front-code 256", s, compress_ms));

    let s = onpair_front_coded(&out, 1024)?;
    results.push(mk("onpair + front-code 1024", s, compress_ms));

    Ok(results)
}

fn print_table(name: &str, rows: &[Row]) {
    println!("\n## {name}");
    println!();
    println!("| Encoding | Bytes | MiB | Ratio vs raw | Bits/row | Encode ms |");
    println!("|---|---:|---:|---:|---:|---:|");
    for r in rows {
        println!(
            "| {} | {} | {:.2} | {:.2}× | {:.2} | {} |",
            r.encoding,
            r.bytes,
            r.bytes as f64 / 1024.0 / 1024.0,
            r.ratio,
            r.bits_per_row,
            r.encode_ms,
        );
    }
}

fn write_markdown(path: &str, sections: &[(String, Vec<Row>)]) -> Result<()> {
    let mut s = String::new();
    s.push_str("# OnPair + token-space block front-coding: empirical results\n\n");
    s.push_str(
        "Reproduce: `cargo run --release -p onpair-bench -- all 1000000 2`.\n\n",
    );
    s.push_str("## Methodology\n\n");
    s.push_str(
        "All input columns are **lex-sorted** before encoding (the scenario under \
         test). Every encoding's reported byte count includes the per-row 4-byte \
         offset table so the comparisons are apples-to-apples for random row \
         access.\n\n",
    );
    s.push_str(
        "- `raw (sorted)` — sum of sorted string lengths + offsets.\n\
         - `zstd-3 / zstd-9 monolithic` — one zstd of the concatenated bytes. \
         Loses random access (best ratio, baseline for what's achievable).\n\
         - `zstd-3 block-1024` — zstd per 1024-row block; random access at block \
         granularity.\n\
         - `fsst` — `fsst-rs` symbol table + per-row compressed payload + offsets.\n\
         - `byte front-code 256` — classical DELTA_BYTE_ARRAY style: anchor row \
         per 256, others store `(shared_with_prev: u32, suffix_bytes)`.\n\
         - `onpair (12-bit)` — OnPair dict + bit-packed codes. No cross-row \
         delta.\n\
         - `onpair + front-code N` — OnPair codes laid out as block front-coding \
         in **token space**: per block of N, anchor row stores its full token \
         sequence (bit-packed at OnPair's bit width), subsequent rows store \
         `(shared_with_prev_tokens: u16, suffix_tokens)` with the suffix \
         bit-packed at the same width. Random access cost: ≤N token prefix \
         copies per row.\n\n",
    );
    for (name, rows) in sections {
        s.push_str(&format!("## {name}\n\n"));
        s.push_str("| Encoding | Bytes | MiB | Ratio vs raw | Bits/row | Encode ms |\n");
        s.push_str("|---|---:|---:|---:|---:|---:|\n");
        for r in rows {
            s.push_str(&format!(
                "| {} | {} | {:.2} | {:.2}× | {:.2} | {} |\n",
                r.encoding,
                r.bytes,
                r.bytes as f64 / 1024.0 / 1024.0,
                r.ratio,
                r.bits_per_row,
                r.encode_ms,
            ));
        }
        s.push('\n');
    }
    fs::write(path, s)?;
    Ok(())
}

fn run_sort_subcommand(which: &str, max_rows: usize) -> Result<()> {
    let loaders: Vec<(&str, Box<dyn Fn(usize) -> Result<Vec<Vec<u8>>>>)> = match which {
        "tpch_l_comment" => vec![(
            "tpch_l_comment",
            Box::new(|n| datasets::tpch_l_comment(n, 0)),
        )],
        "clickbench_url" => vec![(
            "clickbench_url",
            Box::new(|n| datasets::clickbench_column("URL", n, 0)),
        )],
        "clickbench_title" => vec![(
            "clickbench_title",
            Box::new(|n| datasets::clickbench_column("Title", n, 0)),
        )],
        "all" => vec![
            (
                "tpch_l_comment",
                Box::new(|n| datasets::tpch_l_comment(n, 0)),
            ),
            (
                "clickbench_title",
                Box::new(|n| datasets::clickbench_column("Title", n, 0)),
            ),
            (
                "clickbench_url",
                Box::new(|n| datasets::clickbench_column("URL", n, 0)),
            ),
        ],
        other => anyhow::bail!("unknown dataset '{other}'"),
    };
    let mut sections: Vec<(String, Vec<sort_bench::SortRow>)> = Vec::new();
    for (name, loader) in loaders {
        let rows = loader(max_rows)?;
        // Shuffled and almost-sorted variants.
        let r2 = rows.clone();
        let (sec_name, results) =
            sort_bench::run_sort_bench_with(name, rows, sort_bench::Order::Shuffled)?;
        sort_bench::print_sort_table(&sec_name, &results);
        sections.push((sec_name, results));

        let (sec_name, results) = sort_bench::run_sort_bench_with(
            &format!("{name} almost-sorted"),
            r2,
            sort_bench::Order::AlmostSorted,
        )?;
        sort_bench::print_sort_table(&sec_name, &results);
        sections.push((sec_name, results));
    }

    // Append sort-bench results to results.md
    let mut s = String::new();
    s.push_str("\n# sort_bench: compare_fused vs decode-then-byte-compare\n\n");
    s.push_str(
        "All three methods sort the same shuffled column and produce the same \
         permutation (asserted in code). Method 1 sorts u16 token sequences via \
         `compare_fused`. Method 2 sorts the pre-decoded `Vec<Vec<u8>>` directly \
         (best case for the byte-compare baseline — decode cost is not charged). \
         Method 3 decodes from the OnPair-encoded column and then sorts (realistic \
         end-to-end cost when your storage form is encoded).\n\n",
    );
    for (name, rs) in &sections {
        s.push_str(&format!("## {name}\n\n"));
        s.push_str("| Method | Time (ms) | MB/s (raw) | ns/row |\n");
        s.push_str("|---|---:|---:|---:|\n");
        for r in rs {
            s.push_str(&format!(
                "| {} | {} | {:.1} | {:.0} |\n",
                r.method, r.elapsed_ms, r.mb_per_s, r.ns_per_row
            ));
        }
        s.push('\n');
    }
    use std::io::Write;
    let mut f = fs::OpenOptions::new()
        .append(true)
        .create(true)
        .open("experiments/onpair-bench/results.md")?;
    f.write_all(s.as_bytes())?;
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.get(1).map(|s| s.as_str()) == Some("sort_bench") {
        let which = args.get(2).map(|s| s.as_str()).unwrap_or("all");
        let max_rows: usize = args
            .get(3)
            .and_then(|s| s.parse().ok())
            .unwrap_or(500_000);
        return run_sort_subcommand(which, max_rows);
    }

    let which = args.get(1).map(|s| s.as_str()).unwrap_or("all");
    let max_rows: usize = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_000_000);
    let n_slices: usize = args
        .get(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);

    // Loader signature: (max_rows, slice_index) -> rows.
    // `slice_index` is interpreted differently per dataset (TPC-H: skip that
    // many rows; ClickBench: select that partition file).
    type Loader = Box<dyn Fn(usize, usize) -> Result<Vec<Vec<u8>>>>;
    let datasets: Vec<(&str, Loader)> = match which {
        "tpch_l_comment" => vec![(
            "tpch_l_comment",
            Box::new(|n, slice| datasets::tpch_l_comment(n, slice * n)),
        )],
        "clickbench_url" => vec![(
            "clickbench_url",
            Box::new(|n, slice| datasets::clickbench_column("URL", n, slice)),
        )],
        "clickbench_title" => vec![(
            "clickbench_title",
            Box::new(|n, slice| datasets::clickbench_column("Title", n, slice)),
        )],
        "all" => vec![
            (
                "tpch_l_comment",
                Box::new(|n, slice| datasets::tpch_l_comment(n, slice * n)),
            ),
            (
                "clickbench_title",
                Box::new(|n, slice| datasets::clickbench_column("Title", n, slice)),
            ),
            (
                "clickbench_url",
                Box::new(|n, slice| datasets::clickbench_column("URL", n, slice)),
            ),
        ],
        other => anyhow::bail!("unknown dataset '{other}'"),
    };

    let mut sections = Vec::new();
    for (name, loader) in datasets {
        for slice in 0..n_slices {
            let t = Instant::now();
            let rows = loader(max_rows, slice)?;
            eprintln!(
                "[{name} slice {slice}] loaded {} rows in {} ms",
                rows.len(),
                t.elapsed().as_millis()
            );
            let section = format!("{name} (slice {slice})");
            let results = run_on(&section, rows)?;
            print_table(&section, &results);
            sections.push((section, results));
        }
    }

    write_markdown("experiments/onpair-bench/results.md", &sections)?;
    eprintln!("Wrote experiments/onpair-bench/results.md");
    Ok(())
}
