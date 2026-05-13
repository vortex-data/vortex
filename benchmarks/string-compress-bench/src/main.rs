// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! `string-compress-report`
//!
//! Reports compressed-size, compress / decompress throughput, and pushdown
//! latency (equality, `LIKE '%needle%'` substring, `LIKE 'prefix%'` prefix)
//! for every linked-in backend across every synthetic dataset.
//!
//! Run it from the repo root with:
//! ```text
//! cargo run --release -p string-compress-bench --bin string-compress-report
//! ```

use clap::Parser;
use string_compress_bench::{
    BackendConfig, BackendKind, MeasureOpts, datasets, run_backend,
};

/// CLI for the synthetic report.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Cli {
    /// Number of rows in each synthetic dataset.
    #[arg(long, default_value_t = 4096)]
    rows: usize,

    /// How many times to repeat each phase. The harness reports the best run.
    #[arg(long, default_value_t = 3)]
    iters: u32,

    /// Restrict the run to one backend name.
    #[arg(long)]
    only_backend: Option<String>,

    /// Restrict the run to one dataset name.
    #[arg(long)]
    only_dataset: Option<String>,

    // ─── Knobs ──────────────────────────────────────────────────────────────
    /// `onpair` / `onpair16` merging threshold (must be > 1).
    #[arg(long, default_value_t = 4)]
    onpair_threshold: u16,

    /// `onpair-cpp` code width in bits. 8..=16. 14 ⇒ 16 384 dictionary slots
    /// (upstream default in the README example).
    #[arg(long, default_value_t = 14)]
    onpair_cpp_bits: u8,

    /// `onpair-cpp` training-shuffle RNG seed.
    #[arg(long, default_value_t = 42)]
    onpair_cpp_seed: u32,

    /// `onpair-cpp` fixed merge threshold; 0 keeps the upstream dynamic
    /// threshold (currently a no-op in this binary; reserved for future use).
    #[arg(long, default_value_t = 0)]
    onpair_cpp_fixed_threshold: u32,
}

fn main() {
    let cli = Cli::parse();
    let opts = MeasureOpts {
        compress_iters: cli.iters,
        decompress_iters: cli.iters,
        pushdown_iters: cli.iters,
    };
    let cfg = BackendConfig {
        onpair_threshold: cli.onpair_threshold,
        onpair_cpp_bits: cli.onpair_cpp_bits,
        onpair_cpp_seed: cli.onpair_cpp_seed,
        onpair_cpp_fixed_threshold: cli.onpair_cpp_fixed_threshold,
    };

    let datasets = datasets::all_datasets(cli.rows);
    let backends = BackendKind::all();

    println!("# string-compress-bench report");
    println!(
        "# rows={} iters={} backends=[{}]",
        cli.rows,
        cli.iters,
        backends.iter().map(|b| b.name()).collect::<Vec<_>>().join(", "),
    );
    println!(
        "# knobs: onpair_threshold={} onpair_cpp_bits={} onpair_cpp_seed={}",
        cli.onpair_threshold, cli.onpair_cpp_bits, cli.onpair_cpp_seed,
    );
    println!(
        "# columns: dataset | backend | rows | raw | payload | total | r_payload | r_total | compress | decompress | eq_ms (hits, pushdown?) | contains_ms (hits, pushdown?) | starts_with_ms (hits) | roundtrip",
    );

    for corpus in &datasets {
        if let Some(name) = &cli.only_dataset
            && corpus.name != name.as_str()
        {
            continue;
        }
        for &kind in backends {
            if let Some(name) = &cli.only_backend
                && kind.name() != name.as_str()
            {
                continue;
            }
            let r = run_backend(kind, &corpus.strings, corpus.name, &corpus.needles, cfg, opts);
            let ratio_payload = r.uncompressed_bytes as f64 / r.compressed_payload_bytes as f64;
            let ratio_total = r.uncompressed_bytes as f64 / r.total_compressed_bytes as f64;
            let compress_ms = r.compress.as_secs_f64() * 1e3;
            let decompress_ms = r.decompress.as_secs_f64() * 1e3;
            let eq_label = format!(
                "{:>8.3} ({:>5}, {})",
                r.equality_pushdown
                    .map(|d| d.as_secs_f64() * 1e3)
                    .unwrap_or(f64::NAN),
                r.equality_hits,
                if r.equality_is_compressed_domain { "PD" } else { "--" },
            );
            let contains_label = format!(
                "{:>8.3} ({:>5}, {})",
                r.contains_pushdown
                    .map(|d| d.as_secs_f64() * 1e3)
                    .unwrap_or(f64::NAN),
                r.contains_hits,
                if r.substring_is_compressed_domain { "PD" } else { "--" },
            );
            let starts_with_label = format!(
                "{:>8.3} ({:>5})",
                r.starts_with_pushdown
                    .map(|d| d.as_secs_f64() * 1e3)
                    .unwrap_or(f64::NAN),
                r.starts_with_hits,
            );

            println!(
                "{:<14} | {:<12} | {:>6} | {:>10} | {:>10} | {:>10} | {:>6.2}x | {:>6.2}x | {:>8.3} ms | {:>8.3} ms | {} ms | {} ms | {} ms | {:<3}",
                r.dataset,
                r.backend,
                r.rows,
                r.uncompressed_bytes,
                r.compressed_payload_bytes,
                r.total_compressed_bytes,
                ratio_payload,
                ratio_total,
                compress_ms,
                decompress_ms,
                eq_label,
                contains_label,
                starts_with_label,
                r.roundtrip_ok,
            );
        }
        println!();
    }
}
