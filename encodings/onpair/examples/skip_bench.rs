// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
//! Per-chunk skip-index benchmark for arbitrary string columns.
//!
//! Reads one or more Parquet files, picks a string column, splits the
//! column into fixed-size chunks, and reports how much of the data a
//! per-chunk **trigram Bloom filter** would skip on a substring /
//! prefix workload.
//!
//! ## Compression independence
//!
//! The trigram Bloom is built from the **raw row bytes**, regardless of
//! how the column happens to be encoded on disk. So this benchmark
//! tells you the pruning ceiling for *any* compression scheme — Parquet
//! dictionary, FSST, OnPair, plain Utf8View, LZ4 raw bytes — they all
//! get the same answer. With `--mode onpair` it additionally trains an
//! OnPair dictionary and reports the OnPair-specific `DictPresence`
//! bitmap as a side-by-side comparison.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --release --example skip_bench -p vortex-onpair -- \
//!     --parquet /path/to/hits_0.parquet \
//!     --parquet /path/to/hits_1.parquet \
//!     --column URL \
//!     --max-rows 5000000 \
//!     --chunk-size 1024 \
//!     --contains google \
//!     --contains youtube
//! ```
//!
//! With no `--contains` / `--starts-with` it auto-samples 200 random
//! substrings + 50 random prefixes from real rows, plus 50 synthetic
//! "rare" needles, giving a representative distribution of selectivity.
//!
//! ## Vortex columns
//!
//! `--vortex` is not supported in this binary; convert your Vortex file
//! back to Parquet (or read it with the Vortex Python bindings and dump
//! a column to Parquet) and point this benchmark at the result.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::print_stdout,
    clippy::use_debug
)]

use std::collections::BTreeMap;
use std::fs::File;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::Context;
use arrow_array::Array as ArrowArray;
use arrow_array::cast::AsArray;
use clap::Parser;
use clap::ValueEnum;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::VarBinArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_onpair::DEFAULT_DICT12_CONFIG;
use vortex_onpair::decode::OwnedDecodeInputs;
use vortex_onpair::lpm::DictIndex;
use vortex_onpair::onpair_compress;
use vortex_onpair::skip::DictPresence;
use vortex_onpair::skip::TrigramBloom;

#[derive(Parser)]
#[command(about = "Chunk-level trigram-Bloom skip-index benchmark for string columns")]
struct Args {
    /// Parquet file path. Repeat for multiple files.
    #[arg(long, num_args = 1.., required = true)]
    parquet: Vec<PathBuf>,

    /// Column name to analyse. Must be Utf8 / Utf8View / Binary /
    /// BinaryView / Large{Utf8,Binary}.
    #[arg(long, default_value = "URL")]
    column: String,

    /// Rows per chunk (the page granularity at which skipping happens).
    #[arg(long, default_value_t = 1024)]
    chunk_size: usize,

    /// Cap on total rows loaded across all files. 0 = no cap.
    #[arg(long, default_value_t = 1_000_000)]
    max_rows: usize,

    /// Trigram Bloom sizing in bits per row. 32 b/row ≈ 4 KB / 1024-row
    /// chunk, ~5% FPR at typical URL trigram density.
    #[arg(long, default_value_t = 32)]
    trigram_bits_per_row: usize,

    /// Add `LIKE '%S%'` to the workload. Can repeat.
    #[arg(long)]
    contains: Vec<String>,

    /// Add `LIKE 'S%'` to the workload. Can repeat.
    #[arg(long)]
    starts_with: Vec<String>,

    /// Auto-generated substring needles sampled from random real rows.
    #[arg(long, default_value_t = 200)]
    auto_substrings: usize,

    /// Auto-generated prefix needles sampled from random real rows.
    #[arg(long, default_value_t = 50)]
    auto_prefixes: usize,

    /// Auto-generated synthetic "rare" substring needles.
    #[arg(long, default_value_t = 50)]
    auto_rare: usize,

    /// PRNG seed for auto needle sampling.
    #[arg(long, default_value_t = 0x9e37_79b9_7f4a_7c15_u64)]
    seed: u64,

    /// Index mode.
    #[arg(long, value_enum, default_value_t = Mode::Raw)]
    mode: Mode,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Mode {
    /// Build trigram Bloom from raw row bytes. Compression-agnostic.
    Raw,
    /// Also OnPair-compress the column and build the OnPair-specific
    /// DictPresence overlay for side-by-side comparison.
    Onpair,
}

#[derive(Clone, Debug)]
enum Pred {
    StartsWith(String),
    Contains(String),
}

impl Pred {
    fn truly_matches(&self, rows: &[String]) -> bool {
        match self {
            Pred::StartsWith(s) => rows.iter().any(|r| r.starts_with(s.as_str())),
            Pred::Contains(s) => rows.iter().any(|r| r.contains(s.as_str())),
        }
    }
    fn bytes(&self) -> &[u8] {
        match self {
            Pred::StartsWith(s) | Pred::Contains(s) => s.as_bytes(),
        }
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let t0 = Instant::now();
    let rows = load_column(&args.parquet, &args.column, args.max_rows)?;
    let n_rows = rows.len();
    anyhow::ensure!(
        n_rows > args.chunk_size,
        "got only {n_rows} rows; need > chunk_size = {}",
        args.chunk_size
    );
    let num_chunks = n_rows / args.chunk_size;
    let n_aligned = num_chunks * args.chunk_size;
    let raw_bytes: usize = rows[..n_aligned].iter().map(String::len).sum();
    println!(
        "loaded {n_rows} rows in {:?}; chunks: {num_chunks} × {} ; raw_bytes: {raw_bytes}",
        t0.elapsed(),
        args.chunk_size,
    );

    // --------------------- build indexes ---------------------
    let mut trigrams: Vec<TrigramBloom> = Vec::with_capacity(num_chunks);
    let mut presence: Vec<DictPresence> = Vec::new();
    // Keep these alive when in onpair mode so DictPresence can reference them.
    let mut onpair_state: Option<OnPairState> = None;

    let t0 = Instant::now();
    match args.mode {
        Mode::Raw => {
            for c in 0..num_chunks {
                let lo = c * args.chunk_size;
                let hi = lo + args.chunk_size;
                trigrams.push(TrigramBloom::build_from_strings(
                    rows[lo..hi].iter().map(String::as_bytes),
                    args.chunk_size,
                    args.trigram_bits_per_row,
                ));
            }
        }
        Mode::Onpair => {
            let state = build_onpair_state(&rows[..n_aligned])?;
            let dv = state.inputs.view();
            for c in 0..num_chunks {
                let lo = c * args.chunk_size;
                let hi = lo + args.chunk_size;
                presence.push(DictPresence::build(&dv, lo, hi));
                trigrams.push(TrigramBloom::build(&dv, lo, hi, args.trigram_bits_per_row));
            }
            onpair_state = Some(state);
        }
    }
    let index_build_elapsed = t0.elapsed();

    let trigram_bytes: usize = trigrams.iter().map(TrigramBloom::byte_size).sum();
    let presence_bytes: usize = presence.iter().map(DictPresence::byte_size).sum();

    println!(
        "built indexes in {index_build_elapsed:?}: TrigramBloom = {} B total ({} B / chunk, {:.2} B / row)",
        trigram_bytes,
        trigram_bytes / num_chunks,
        trigram_bytes as f64 / n_aligned as f64,
    );
    if matches!(args.mode, Mode::Onpair) {
        println!(
            "                   DictPresence = {} B total ({} B / chunk, {:.2} B / row)",
            presence_bytes,
            presence_bytes / num_chunks,
            presence_bytes as f64 / n_aligned as f64,
        );
    }
    println!(
        "index overhead: TrigramBloom = {:.4}% of raw text",
        100.0 * trigram_bytes as f64 / raw_bytes as f64,
    );
    println!();

    // ---------------- generate the workload ----------------
    let mut workload: Vec<(&'static str, Pred)> = Vec::new();
    for s in &args.contains {
        workload.push(("user/contains", Pred::Contains(s.clone())));
    }
    for s in &args.starts_with {
        workload.push(("user/prefix", Pred::StartsWith(s.clone())));
    }
    let mut rng = Splitmix64::new(args.seed);
    for _ in 0..args.auto_substrings {
        if let Some(p) = sample_substring(&rows, n_aligned, &mut rng) {
            workload.push(("auto/substring", p));
        }
    }
    for _ in 0..args.auto_prefixes {
        if let Some(p) = sample_prefix(&rows, n_aligned, &mut rng) {
            workload.push(("auto/prefix", p));
        }
    }
    for _ in 0..args.auto_rare {
        workload.push(("auto/rare", sample_rare(&mut rng)));
    }

    println!("workload: {} queries", workload.len());
    println!();

    // ---------------- evaluate ----------------
    #[derive(Default)]
    struct Cat {
        n_q: usize,
        n_c: usize,
        real: usize,
        kept_b: usize,
        kept_a: usize,
    }
    let mut by_tag: BTreeMap<&'static str, Cat> = BTreeMap::new();
    let mut total = Cat::default();

    let dv_index = onpair_state
        .as_ref()
        .map(|st| (st.inputs.view(), DictIndex::build(&st.inputs.view())));

    let t0 = Instant::now();
    for (tag, q) in &workload {
        let cat = by_tag.entry(tag).or_default();
        cat.n_q += 1;
        total.n_q += 1;
        for c in 0..num_chunks {
            let lo = c * args.chunk_size;
            let hi = lo + args.chunk_size;
            let real = q.truly_matches(&rows[lo..hi]);
            let keep_b = trigrams[c].might_contain(q.bytes());
            assert!(!real || keep_b, "Trigram false negative on chunk {c} for {q:?}");
            cat.n_c += 1;
            cat.real += real as usize;
            cat.kept_b += keep_b as usize;
            total.n_c += 1;
            total.real += real as usize;
            total.kept_b += keep_b as usize;
            if let Some((dv, idx)) = dv_index.as_ref() {
                let keep_a = match q {
                    Pred::Contains(s) => presence[c].might_contain(dv, s.as_bytes()),
                    Pred::StartsWith(s) => presence[c].might_starts_with(dv, idx, s.as_bytes()),
                };
                assert!(!real || keep_a, "DictPresence false negative on chunk {c} for {q:?}");
                cat.kept_a += keep_a as usize;
                total.kept_a += keep_a as usize;
            }
        }
    }
    let eval_elapsed = t0.elapsed();

    // ---------------- report ----------------
    let mode_has_a = matches!(args.mode, Mode::Onpair);
    println!(
        "{:<24} {:>5} {:>10} {:>10} {:>10} {:>10}",
        "category", "Q", "C", "real%", "B.kept%", "B.vs_floor",
    );
    if mode_has_a {
        println!(
            "{:<24}      {:>10} {:>10} {:>10} {:>10}",
            "                  +A:", "", "", "A.kept%", "A.vs_floor",
        );
    }
    println!("{}", "-".repeat(if mode_has_a { 92 } else { 76 }));
    let print_row = |name: &str, c: &Cat| {
        if c.n_c == 0 {
            return;
        }
        let floor = 100.0 * c.real as f64 / c.n_c as f64;
        let kept_b = 100.0 * c.kept_b as f64 / c.n_c as f64;
        println!(
            "{:<24} {:>5} {:>10} {:>9.2}% {:>9.2}% {:>+9.2}pp",
            name,
            c.n_q,
            c.n_c,
            floor,
            kept_b,
            kept_b - floor,
        );
        if mode_has_a {
            let kept_a = 100.0 * c.kept_a as f64 / c.n_c as f64;
            println!(
                "{:<24}     {:>10} {:>10} {:>9.2}% {:>+9.2}pp",
                "                  +A:", "", "", kept_a, kept_a - floor,
            );
        }
    };
    for (tag, cat) in &by_tag {
        print_row(tag, cat);
    }
    println!("{}", "-".repeat(if mode_has_a { 92 } else { 76 }));
    print_row("TOTAL", &total);

    println!();
    println!("Columns:");
    println!("  Q           = # queries in this category");
    println!("  C           = # (query × chunk) evaluations");
    println!("  real%       = (q,c) pairs with ≥1 actual match — the FLOOR for any sound prefilter");
    println!("  B.kept%     = (q,c) pairs the TrigramBloom still keeps  (lower = more pruning)");
    println!("  B.vs_floor  = B.kept% − real%   (0 = optimal pruning; higher = wasted scans)");
    if mode_has_a {
        println!("  A.kept%     = same for the OnPair DictPresence bitmap");
    }
    println!();
    println!("evaluated in {eval_elapsed:?}");
    Ok(())
}

/// Keeps `OwnedDecodeInputs` alive so a `DecodeView` borrowed from it
/// stays valid for the lifetime of `main`.
struct OnPairState {
    inputs: OwnedDecodeInputs,
}

fn build_onpair_state(rows: &[String]) -> anyhow::Result<OnPairState> {
    let varbin = VarBinArray::from_iter(
        rows.iter().map(|s| Some(s.as_bytes())),
        DType::Utf8(Nullability::NonNullable),
    );
    let arr = onpair_compress(&varbin, varbin.len(), varbin.dtype(), DEFAULT_DICT12_CONFIG)
        .map_err(|e| anyhow::anyhow!("OnPair compress failed: {e}"))?;
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let inputs = OwnedDecodeInputs::collect(arr.as_view(), &mut ctx)
        .map_err(|e| anyhow::anyhow!("collect decode inputs: {e}"))?;
    Ok(OnPairState { inputs })
}

// ---------------------- workload sampling ----------------------

fn sample_substring(rows: &[String], n_aligned: usize, rng: &mut Splitmix64) -> Option<Pred> {
    for _ in 0..16 {
        let i = (rng.next() as usize) % n_aligned;
        let s = rows[i].as_bytes();
        if s.len() < 6 {
            continue;
        }
        let max_len = s.len().min(15);
        let len = 5 + (rng.next() as usize) % (max_len - 4);
        let start = (rng.next() as usize) % (s.len() - len + 1);
        if let Ok(needle) = std::str::from_utf8(&s[start..start + len]) {
            if !needle.is_empty() {
                return Some(Pred::Contains(needle.to_string()));
            }
        }
    }
    None
}

fn sample_prefix(rows: &[String], n_aligned: usize, rng: &mut Splitmix64) -> Option<Pred> {
    for _ in 0..16 {
        let i = (rng.next() as usize) % n_aligned;
        let s = rows[i].as_bytes();
        if s.len() < 12 {
            continue;
        }
        let max_len = s.len().min(30);
        let len = 12 + (rng.next() as usize) % (max_len - 11);
        if let Ok(prefix) = std::str::from_utf8(&s[..len]) {
            if !prefix.is_empty() {
                return Some(Pred::StartsWith(prefix.to_string()));
            }
        }
    }
    None
}

fn sample_rare(rng: &mut Splitmix64) -> Pred {
    let len = 6 + (rng.next() as usize) % 7;
    let mut s = String::with_capacity(len + 4);
    for _ in 0..len {
        s.push((((rng.next() % 26) as u8) + b'a') as char);
    }
    s.push_str(&format!("{}", rng.next() % 1000));
    Pred::Contains(s)
}

// ---------------------- parquet column loading ----------------------

fn load_column(paths: &[PathBuf], col_name: &str, max_rows: usize) -> anyhow::Result<Vec<String>> {
    let cap = if max_rows > 0 { max_rows } else { 0 };
    let mut out: Vec<String> = Vec::with_capacity(cap.min(8 * 1024 * 1024));
    'outer: for path in paths {
        let file = File::open(path).with_context(|| format!("open {path:?}"))?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let schema = builder.schema().clone();
        let col_idx = schema
            .fields()
            .iter()
            .position(|f| f.name() == col_name)
            .with_context(|| format!("column {col_name:?} not in {path:?}"))?;
        let mask = ProjectionMask::leaves(builder.parquet_schema(), [col_idx]);
        let mut reader = builder.with_projection(mask).with_batch_size(8192).build()?;
        while let Some(batch) = reader.next() {
            let batch = batch?;
            let col = batch.column(0);
            let want = if max_rows > 0 { max_rows.saturating_sub(out.len()) } else { col.len() };
            if want == 0 {
                break 'outer;
            }
            let pushed = push_strings(col, want, &mut out);
            anyhow::ensure!(pushed > 0, "unexpected column type: {:?}", col.data_type());
            if max_rows > 0 && out.len() >= max_rows {
                break 'outer;
            }
        }
    }
    Ok(out)
}

fn push_strings(col: &dyn ArrowArray, want: usize, out: &mut Vec<String>) -> usize {
    if let Some(s) = col.as_string_opt::<i32>() {
        let n = s.len().min(want);
        for i in 0..n {
            out.push(s.value(i).to_string());
        }
        return n;
    }
    if let Some(s) = col.as_string_opt::<i64>() {
        let n = s.len().min(want);
        for i in 0..n {
            out.push(s.value(i).to_string());
        }
        return n;
    }
    if let Some(s) = col.as_string_view_opt() {
        let n = s.len().min(want);
        for i in 0..n {
            out.push(s.value(i).to_string());
        }
        return n;
    }
    if let Some(b) = col.as_binary_opt::<i32>() {
        let n = b.len().min(want);
        for i in 0..n {
            out.push(String::from_utf8_lossy(b.value(i)).into_owned());
        }
        return n;
    }
    if let Some(b) = col.as_binary_opt::<i64>() {
        let n = b.len().min(want);
        for i in 0..n {
            out.push(String::from_utf8_lossy(b.value(i)).into_owned());
        }
        return n;
    }
    if let Some(b) = col.as_binary_view_opt() {
        let n = b.len().min(want);
        for i in 0..n {
            out.push(String::from_utf8_lossy(b.value(i)).into_owned());
        }
        return n;
    }
    0
}

struct Splitmix64(u64);
impl Splitmix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}
