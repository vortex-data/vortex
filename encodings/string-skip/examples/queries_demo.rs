// SPDX-License-Identifier: Apache-2.0
//! End-to-end demo: load a parquet column, build all the skip indexes,
//! run a synthetic workload covering the full SQL string predicate
//! taxonomy, report pruning rates.
//!
//! ```bash
//! cargo run --release -p string-skip --example queries_demo -- \
//!     --parquet /path/to.parquet --column URL --max-rows 1000000 --sort
//! ```
//!
//! Note: this example uses a **stub** dictionary built from row content
//! rather than a real OnPair/FSST encoding. It exists to validate the
//! API surface and produce comparable pruning numbers to the
//! `vortex-onpair` integration. Real production use plugs in OnPair's
//! dict via the `TokenDict` trait.

#![allow(clippy::print_stdout)]

use std::collections::HashSet;
use std::fs::File;
use std::path::PathBuf;

use anyhow::Context;
use arrow_array::Array as ArrowArray;
use arrow_array::cast::AsArray;
use clap::Parser;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use rand::SeedableRng;
use rand::seq::IndexedRandom;
use rand_xoshiro::Xoshiro256PlusPlus;

use string_skip::{
    BigramTiers, ChunkStats, DictIndex, DictPresence, HybridBloom, Pred,
    UbiquitousBigrams, chunk_might_match,
    dict::{TokenDict, tokenize_needle},
    prune::ChunkSkipState,
};

#[derive(Parser)]
struct Args {
    #[arg(long, num_args = 1.., required = true)]
    parquet: Vec<PathBuf>,
    #[arg(long)]
    column: String,
    #[arg(long, default_value_t = 1_000_000)]
    max_rows: usize,
    #[arg(long, default_value_t = 8192)]
    chunk_size: usize,
    #[arg(long, default_value_t = 16)]
    bits: usize,
    #[arg(long, default_value_t = 75)]
    ubiq_pct: u8,
    #[arg(long)]
    sort: bool,
    #[arg(long, default_value_t = 50)]
    samples: usize,
    #[arg(long, default_value_t = 0x9e37_79b9_7f4a_7c15_u64)]
    seed: u64,
}

/// A simple dict built from byte n-grams seen in the input. Not a real
/// OnPair dict — exists just to drive the demo end-to-end.
///
/// In production, callers plug in OnPair's `DecodeView` (or FSST's
/// equivalent) via the `TokenDict` trait.
struct DemoDict {
    toks: Vec<Vec<u8>>,
}

impl DemoDict {
    /// Build by seeding with all 256 single-byte tokens and the most
    /// frequent 2-3 byte substrings seen in `rows`.
    fn build(rows: &[Vec<u8>], target_size: usize) -> Self {
        let mut counts: std::collections::HashMap<Vec<u8>, u32> =
            std::collections::HashMap::new();
        for r in rows {
            for w in r.windows(2) {
                *counts.entry(w.to_vec()).or_insert(0) += 1;
            }
            for w in r.windows(3) {
                *counts.entry(w.to_vec()).or_insert(0) += 1;
            }
        }
        let mut by_count: Vec<(Vec<u8>, u32)> = counts.into_iter().collect();
        by_count.sort_by(|a, b| b.1.cmp(&a.1));
        let mut toks: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
        for (bytes, _) in by_count.into_iter().take(target_size.saturating_sub(256)) {
            toks.push(bytes);
        }
        toks.sort();
        toks.dedup();
        Self { toks }
    }
}

impl TokenDict for DemoDict {
    fn len(&self) -> usize { self.toks.len() }
    fn token_bytes(&self, id: u16) -> &[u8] { &self.toks[id as usize] }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    eprintln!("Loading column {:?}...", args.column);
    let mut rows = load_column(&args.parquet, &args.column, args.max_rows)?;
    if args.sort {
        rows.sort();
        eprintln!("  sorted {} rows", rows.len());
    }
    let total_raw: usize = rows.iter().map(Vec::len).sum();
    eprintln!("loaded {} rows ({:.1} MB)", rows.len(), total_raw as f64 / 1e6);

    eprintln!("Building demo dict (4096 tokens)...");
    let dict = DemoDict::build(&rows, 4096);
    let index = DictIndex::build(&dict);
    eprintln!("  dict_size = {}", dict.len());

    eprintln!("Tokenizing rows...");
    let mut codes = Vec::new();
    let mut offsets = vec![0u32];
    for r in &rows {
        let toks = tokenize_needle(&dict, &index, r)
            .ok_or_else(|| anyhow::anyhow!("untokenizable row"))?;
        codes.extend(toks);
        offsets.push(codes.len() as u32);
    }
    eprintln!("  {} total tokens, {:.1} per row", codes.len(),
        codes.len() as f64 / rows.len() as f64);

    let cs = args.chunk_size;
    let nch = rows.len() / cs;
    if nch == 0 {
        anyhow::bail!("not enough rows for a single chunk");
    }
    eprintln!("Chunks: {nch} × {cs} rows");

    eprintln!("Building per-chunk stats...");
    let chunk_stats: Vec<ChunkStats> = (0..nch)
        .map(|c| ChunkStats::from_rows(&rows[c * cs..(c + 1) * cs]))
        .collect();

    let presence: Vec<DictPresence> = (0..nch)
        .map(|c| {
            let row_lo = c * cs;
            let row_hi = (c + 1) * cs;
            let tok_lo = offsets[row_lo] as usize;
            let tok_hi = offsets[row_hi] as usize;
            DictPresence::build(&codes[tok_lo..tok_hi], dict.len())
        })
        .collect();

    eprintln!("Building ubiquity table...");
    let ubiq = UbiquitousBigrams::build(&codes, &offsets, cs, args.ubiq_pct);
    let tiers = BigramTiers::empty();
    eprintln!("  ubiq: {} entries ({} B)", ubiq.len(), ubiq.byte_size());

    eprintln!("Building per-chunk blooms...");
    let blooms: Vec<HybridBloom> = (0..nch)
        .map(|c| HybridBloom::build(
            &codes, &offsets, c * cs, (c + 1) * cs, args.bits, &ubiq))
        .collect();

    // Build the workload
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(args.seed);
    let workload = build_workload(&rows, &mut rng, args.samples);

    let total_chunks = nch as f64;
    let avg_chunk_bytes = total_raw / nch;

    println!();
    println!("=== Pruning report ({nch} chunks, {:.1} MB raw, {:.2} MB/chunk avg) ===",
        total_raw as f64 / 1e6, avg_chunk_bytes as f64 / 1e6);
    println!();
    println!("{:<38} {:>8} {:>10} {:>10} {:>6}",
        "Predicate class", "skip%", "avg avoid", "kept/qry", "queries");
    println!("{}", "-".repeat(78));

    for (label, queries) in &workload {
        if queries.is_empty() {
            continue;
        }
        let mut total_kept = 0usize;
        let mut total_bytes_saved = 0usize;
        for q in queries {
            let kept = run_query(q, &chunk_stats, &presence, &blooms, &ubiq, &tiers, &dict, &index, nch);
            // Soundness check
            for c in 0..nch {
                let truly = q.matches_any(&rows[c * cs..(c + 1) * cs]);
                if truly && !kept.contains(&c) {
                    panic!("FN for {label} q={q:?} chunk={c}");
                }
            }
            total_kept += kept.len();
            for c in 0..nch {
                if !kept.contains(&c) {
                    total_bytes_saved += chunk_stats[c].raw_bytes;
                }
            }
        }
        let n_q = queries.len();
        let avg_kept = total_kept as f64 / n_q as f64;
        let avg_skipped = total_chunks - avg_kept;
        let skip_pct = 100.0 * avg_skipped / total_chunks;
        let avg_avoid_mb = total_bytes_saved as f64 / n_q as f64 / 1e6;
        println!("{:<38} {:>7.1}% {:>7.1} MB {:>10.1} {:>6}",
            label, skip_pct, avg_avoid_mb, avg_kept, n_q);
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_query<D: TokenDict>(
    pred: &Pred,
    stats: &[ChunkStats],
    presence: &[DictPresence],
    blooms: &[HybridBloom],
    ubiq: &UbiquitousBigrams,
    tiers: &BigramTiers,
    dict: &D,
    index: &DictIndex,
    nch: usize,
) -> HashSet<usize> {
    let mut kept = HashSet::with_capacity(nch);
    for c in 0..nch {
        let state = ChunkSkipState {
            stats: &stats[c],
            presence: &presence[c],
            bloom: Some(&blooms[c]),
            tiered: None,
            ubiq,
            tiers,
            dict,
            index,
        };
        if chunk_might_match(pred, &state) {
            kept.insert(c);
        }
    }
    kept
}

fn build_workload(
    rows: &[Vec<u8>],
    rng: &mut Xoshiro256PlusPlus,
    n: usize,
) -> Vec<(&'static str, Vec<Pred>)> {
    use rand::Rng;
    let mut w = Vec::new();

    // Eq
    let eq: Vec<Pred> = (0..n).map(|_| Pred::Eq(rows.choose(rng).unwrap().clone())).collect();
    w.push(("Eq: col = 'x'", eq));

    // Lt
    let lt: Vec<Pred> = (0..n).map(|_| Pred::Lt(rows.choose(rng).unwrap().clone())).collect();
    w.push(("Lt: col < 'x'", lt));

    // Between
    let between: Vec<Pred> = (0..n).map(|_| {
        let mut a = rows.choose(rng).unwrap().clone();
        let mut b = rows.choose(rng).unwrap().clone();
        if a > b { std::mem::swap(&mut a, &mut b); }
        Pred::Between(a, b)
    }).collect();
    w.push(("Between: BETWEEN a AND b", between));

    // Prefix
    let prefix: Vec<Pred> = (0..n).filter_map(|_| {
        let r = rows.choose(rng).unwrap();
        if r.len() < 8 { return None; }
        let plen = 3 + rng.gen_range(0..6);
        Some(Pred::Prefix(r[..plen.min(r.len())].to_vec()))
    }).collect();
    w.push(("Prefix: LIKE 'p%'", prefix));

    // Suffix
    let suffix: Vec<Pred> = (0..n).filter_map(|_| {
        let r = rows.choose(rng).unwrap();
        if r.len() < 8 { return None; }
        let slen = 3 + rng.gen_range(0..6);
        let start = r.len().saturating_sub(slen);
        Some(Pred::Suffix(r[start..].to_vec()))
    }).collect();
    w.push(("Suffix: LIKE '%s'", suffix));

    // Contains
    let contains: Vec<Pred> = (0..n).filter_map(|_| {
        let r = rows.choose(rng).unwrap();
        if r.len() < 10 { return None; }
        let slen = 5 + rng.gen_range(0..8);
        let start = rng.gen_range(0..r.len() - slen);
        Some(Pred::Contains(r[start..start + slen].to_vec()))
    }).collect();
    w.push(("Contains: LIKE '%x%'", contains));

    // PrefixSuffix
    let psufx: Vec<Pred> = (0..n).filter_map(|_| {
        let r = rows.choose(rng).unwrap();
        if r.len() < 14 { return None; }
        let plen = 3 + rng.gen_range(0..4);
        let slen = 3 + rng.gen_range(0..4);
        Some(Pred::PrefixSuffix(r[..plen].to_vec(), r[r.len() - slen..].to_vec()))
    }).collect();
    w.push(("PrefixSuffix: LIKE 'a%b'", psufx));

    // SingleWildcard
    let sw: Vec<Pred> = (0..n).filter_map(|_| {
        let r = rows.choose(rng).unwrap();
        if r.len() < 12 { return None; }
        let pos = 3 + rng.gen_range(0..r.len() - 10);
        let plen = 3;
        let slen = 3;
        Some(Pred::SingleWildcard(
            r[pos..pos + plen].to_vec(),
            r[pos + plen + 1..pos + plen + 1 + slen].to_vec(),
        ))
    }).collect();
    w.push(("Wildcard: LIKE '%a_b%'", sw));

    // MultiFragment
    let mf: Vec<Pred> = (0..n).filter_map(|_| {
        let r = rows.choose(rng).unwrap();
        if r.len() < 16 { return None; }
        let len1 = 3 + rng.gen_range(0..4);
        let start1 = rng.gen_range(0..r.len() / 3);
        let start2 = start1 + len1 + 2 + rng.gen_range(0..4);
        let len2 = 3 + rng.gen_range(0..4);
        if start2 + len2 > r.len() { return None; }
        Some(Pred::MultiFragment(vec![
            r[start1..start1 + len1].to_vec(),
            r[start2..start2 + len2].to_vec(),
        ]))
    }).collect();
    w.push(("MultiFragment: LIKE '%a%b%'", mf));

    // Length predicates
    let avg_len = rows.iter().map(Vec::len).sum::<usize>() / rows.len().max(1);
    let lg: Vec<Pred> = (0..n).map(|i| Pred::LengthGt(avg_len + (i % 20))).collect();
    w.push(("LengthGt: LENGTH > k", lg));

    let lb: Vec<Pred> = (0..n).map(|_| {
        let lo = rng.gen_range(0..avg_len);
        let hi = lo + 10 + rng.gen_range(0..50);
        Pred::LengthBetween(lo, hi)
    }).collect();
    w.push(("LengthBetween: LENGTH BETWEEN", lb));

    // IN set
    let in_set: Vec<Pred> = (0..n).map(|_| {
        let k = 3 + rng.gen_range(0..5);
        let xs: Vec<Vec<u8>> = (0..k).map(|_| rows.choose(rng).unwrap().clone()).collect();
        Pred::InSet(xs)
    }).collect();
    w.push(("InSet: IN (3-7 values)", in_set));

    w.push(("IsNull: IS NULL", vec![Pred::IsNull]));
    w.push(("IsNotNull: IS NOT NULL", vec![Pred::IsNotNull]));

    w
}

fn load_column(paths: &[PathBuf], col_name: &str, max_rows: usize) -> anyhow::Result<Vec<Vec<u8>>> {
    let mut out = Vec::with_capacity(max_rows.max(1));
    'outer: for path in paths {
        let file = File::open(path).with_context(|| format!("open {path:?}"))?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let schema = builder.schema().clone();
        let col_idx = schema.fields().iter().position(|f| f.name() == col_name)
            .with_context(|| format!("column {col_name:?} not in {path:?}"))?;
        let mask = ProjectionMask::leaves(builder.parquet_schema(), [col_idx]);
        let mut reader = builder.with_projection(mask).with_batch_size(8192).build()?;
        while let Some(batch) = reader.next() {
            let batch = batch?;
            let col = batch.column(0);
            let want = max_rows.saturating_sub(out.len());
            if want == 0 { break 'outer; }
            let pushed = push_strings(col, want, &mut out);
            anyhow::ensure!(pushed > 0, "unexpected column type: {:?}", col.data_type());
            if out.len() >= max_rows { break 'outer; }
        }
    }
    Ok(out)
}

fn push_strings(col: &dyn ArrowArray, want: usize, out: &mut Vec<Vec<u8>>) -> usize {
    if let Some(s) = col.as_string_opt::<i32>() {
        let n = s.len().min(want);
        for i in 0..n { out.push(s.value(i).as_bytes().to_vec()); }
        return n;
    }
    if let Some(s) = col.as_string_opt::<i64>() {
        let n = s.len().min(want);
        for i in 0..n { out.push(s.value(i).as_bytes().to_vec()); }
        return n;
    }
    if let Some(s) = col.as_string_view_opt() {
        let n = s.len().min(want);
        for i in 0..n { out.push(s.value(i).as_bytes().to_vec()); }
        return n;
    }
    if let Some(b) = col.as_binary_opt::<i32>() {
        let n = b.len().min(want);
        for i in 0..n { out.push(b.value(i).to_vec()); }
        return n;
    }
    if let Some(b) = col.as_binary_opt::<i64>() {
        let n = b.len().min(want);
        for i in 0..n { out.push(b.value(i).to_vec()); }
        return n;
    }
    if let Some(b) = col.as_binary_view_opt() {
        let n = b.len().min(want);
        for i in 0..n { out.push(b.value(i).to_vec()); }
        return n;
    }
    0
}
