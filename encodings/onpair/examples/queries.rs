// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
//! `queries` — demonstrates pruning rates for the full taxonomy of SQL
//! string predicates on real columns, using OnPair + the skip-index
//! stack (sorted min/max, DictPresence, HybridBloom).
//!
//! For each query type and each test column, prints:
//!   * total chunks scanned (before pruning)
//!   * chunks proven skippable
//!   * skip% (chunks skipped / total)
//!   * raw bytes saved per query
//!
//! ```bash
//! cargo run --release --example queries -p vortex-onpair -- \
//!     --parquet /path/to.parquet --column URL --max-rows 1000000 \
//!     --sort
//! ```

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::print_stdout,
    clippy::use_debug
)]

use std::collections::HashSet;
use std::fs::File;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::Context;
use arrow_array::Array as ArrowArray;
use arrow_array::cast::AsArray;
use clap::Parser;
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
use vortex_onpair::skip::HybridBloom;
use vortex_onpair::skip::UbiquitousBigrams;

#[derive(Parser)]
struct Args {
    #[arg(long, num_args = 1.., required = true)]
    parquet: Vec<PathBuf>,
    #[arg(long, default_value = "URL")]
    column: String,
    #[arg(long, default_value_t = 1_000_000)]
    max_rows: usize,
    #[arg(long, default_value_t = 8192)]
    chunk_size: usize,
    #[arg(long, default_value_t = 16)]
    bits: usize,
    #[arg(long, default_value_t = 75)]
    ubiq_pct: u8,
    /// Sort rows lexicographically before chunking.
    #[arg(long)]
    sort: bool,
    /// Number of auto-sampled queries per query type.
    #[arg(long, default_value_t = 50)]
    samples: usize,
    #[arg(long, default_value_t = 0x9e37_79b9_7f4a_7c15_u64)]
    seed: u64,
}

/// Per-chunk skip statistics.
struct ChunkStats {
    min: Vec<u8>,
    max: Vec<u8>,
    min_len: usize,
    max_len: usize,
    null_count: usize,
    raw_bytes: usize,
}

/// Result of pruning a single query: how many chunks remain, bytes saved.
#[derive(Default, Clone, Copy)]
struct PruneResult {
    chunks_kept: usize,
    chunks_skipped: usize,
    bytes_saved: usize,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // -------- Load + sort + compress ----------------------------------------
    eprintln!("Loading column {:?} ...", args.column);
    let mut rows = load_column(&args.parquet, &args.column, args.max_rows)?;
    if args.sort {
        rows.sort();
        eprintln!("  sorted {} rows", rows.len());
    }
    let total_raw_bytes: usize = rows.iter().map(String::len).sum();
    eprintln!(
        "loaded {} rows ({:.1} MB raw)",
        rows.len(),
        total_raw_bytes as f64 / 1e6
    );

    let varbin = VarBinArray::from_iter(
        rows.iter().map(|s| Some(s.as_bytes())),
        DType::Utf8(Nullability::NonNullable),
    );
    let t0 = Instant::now();
    let arr =
        onpair_compress(&varbin, varbin.len(), varbin.dtype(), DEFAULT_DICT12_CONFIG).unwrap();
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let inputs = OwnedDecodeInputs::collect(arr.as_view(), &mut ctx).unwrap();
    let dv = inputs.view();
    let index = DictIndex::build(&dv);
    eprintln!(
        "OnPair-compressed in {:?}; dict_size={}",
        t0.elapsed(),
        dv.dict_table.len()
    );

    // -------- Build per-chunk state ------------------------------------------
    let cs = args.chunk_size;
    let n_aligned = (rows.len() / cs) * cs;
    let nch = n_aligned / cs;
    eprintln!("chunks: {} of {} rows each", nch, cs);

    let chunk_stats: Vec<ChunkStats> = (0..nch)
        .map(|c| {
            let slice = &rows[c * cs..(c + 1) * cs];
            let min = slice.iter().map(|s| s.as_bytes()).min().unwrap().to_vec();
            let max = slice.iter().map(|s| s.as_bytes()).max().unwrap().to_vec();
            let min_len = slice.iter().map(String::len).min().unwrap();
            let max_len = slice.iter().map(String::len).max().unwrap();
            let raw_bytes = slice.iter().map(String::len).sum();
            ChunkStats {
                min,
                max,
                min_len,
                max_len,
                null_count: 0,
                raw_bytes,
            }
        })
        .collect();

    let presence: Vec<DictPresence> = (0..nch)
        .map(|c| DictPresence::build(&dv, c * cs, (c + 1) * cs))
        .collect();

    let ubiq = UbiquitousBigrams::build(dv.codes, dv.codes_offsets, cs, args.ubiq_pct);
    let blooms: Vec<HybridBloom> = (0..nch)
        .map(|c| HybridBloom::build(&dv, c * cs, (c + 1) * cs, args.bits, &ubiq))
        .collect();
    eprintln!(
        "ubiq table: {} entries ({} B)",
        ubiq.len(),
        ubiq.byte_size()
    );

    // -------- Build the workload --------------------------------------------
    let mut rng = Splitmix64::new(args.seed);
    let workload = build_workload(&rows, &mut rng, args.samples);

    // -------- Run all queries -----------------------------------------------
    println!();
    println!(
        "=== Query pruning report ({} chunks, {:.1} MB raw, {:.1} MB/chunk avg) ===",
        nch,
        total_raw_bytes as f64 / 1e6,
        chunk_stats.iter().map(|s| s.raw_bytes).sum::<usize>() as f64 / nch as f64 / 1e6
    );
    println!();
    println!(
        "{:<35} {:>10} {:>10} {:>10} {:>10}",
        "Query class (sample size)", "skip%", "bytes_avoid", "chunks_kept", "queries"
    );
    println!("{}", "-".repeat(80));

    let kept_total = nch as f64;

    for (label, queries) in workload {
        if queries.is_empty() {
            continue;
        }
        let n_q = queries.len();
        let mut sum = PruneResult::default();
        for q in &queries {
            let pr = prune(
                q,
                &chunk_stats,
                &presence,
                &blooms,
                &ubiq,
                &dv,
                &index,
                &rows,
                cs,
            );
            // Sanity: assert no false negatives
            for c in 0..nch {
                let truly_matches = q.truly_matches(&rows[c * cs..(c + 1) * cs]);
                if truly_matches && pr.kept_chunks.contains(&c) == false {
                    panic!("FN for query {label}, query {q:?}, chunk {c}");
                }
            }
            sum.chunks_kept += pr.kept_chunks.len();
            sum.chunks_skipped += nch - pr.kept_chunks.len();
            sum.bytes_saved += (0..nch)
                .filter(|c| !pr.kept_chunks.contains(c))
                .map(|c| chunk_stats[c].raw_bytes)
                .sum::<usize>();
        }
        let avg_kept = sum.chunks_kept as f64 / n_q as f64;
        let avg_skipped = sum.chunks_skipped as f64 / n_q as f64;
        let skip_pct = 100.0 * avg_skipped / kept_total;
        let avg_saved_mb = sum.bytes_saved as f64 / n_q as f64 / 1e6;
        println!(
            "{:<35} {:>9.1}% {:>8.1} MB {:>10.1} {:>10}",
            label, skip_pct, avg_saved_mb, avg_kept, n_q
        );
    }

    Ok(())
}

// ============================================================================
//                              Predicates
// ============================================================================

#[derive(Debug, Clone)]
enum Pred {
    /// col = 'x'
    Eq(String),
    /// col < 'x'
    Lt(String),
    /// col >= 'x'
    Gte(String),
    /// col BETWEEN 'a' AND 'b'
    Between(String, String),
    /// LIKE 'p%'
    Prefix(String),
    /// LIKE '%s'
    Suffix(String),
    /// LIKE '%x%'
    Contains(String),
    /// LIKE 'a%b' (anchored prefix + suffix)
    PrefixSuffix(String, String),
    /// LIKE 'a_b' anchored, single-char wildcard at position `pos`
    /// Stored as: prefix + suffix (both fixed); pos is implicit (prefix.len())
    SingleWildcard(String, String),
    /// LIKE '%a%b%' (two fragments anywhere)
    MultiFragment(Vec<String>),
    /// LENGTH(col) > k
    LengthGt(usize),
    /// LENGTH(col) BETWEEN lo AND hi
    LengthBetween(usize, usize),
    /// IS NULL
    IsNull,
    /// col IN (...)
    InSet(Vec<String>),
}

impl Pred {
    fn truly_matches(&self, rows: &[String]) -> bool {
        rows.iter().any(|r| self.matches_one(r))
    }
    fn matches_one(&self, r: &str) -> bool {
        match self {
            Pred::Eq(x) => r == x,
            Pred::Lt(x) => r < x.as_str(),
            Pred::Gte(x) => r >= x.as_str(),
            Pred::Between(a, b) => r >= a.as_str() && r <= b.as_str(),
            Pred::Prefix(p) => r.starts_with(p.as_str()),
            Pred::Suffix(s) => r.ends_with(s.as_str()),
            Pred::Contains(s) => r.contains(s.as_str()),
            Pred::PrefixSuffix(p, s) => r.starts_with(p.as_str()) && r.ends_with(s.as_str()),
            Pred::SingleWildcard(p, s) => {
                let r = r.as_bytes();
                let need = p.len() + 1 + s.len();
                if r.len() < need {
                    return false;
                }
                // try every offset (this is for substring `%a_b%` mode)
                for i in 0..=r.len() - need {
                    if &r[i..i + p.len()] == p.as_bytes()
                        && &r[i + p.len() + 1..i + p.len() + 1 + s.len()] == s.as_bytes()
                    {
                        return true;
                    }
                }
                false
            }
            Pred::MultiFragment(frags) => {
                let mut pos = 0;
                for f in frags {
                    match r[pos..].find(f.as_str()) {
                        Some(off) => pos = pos + off + f.len(),
                        None => return false,
                    }
                }
                true
            }
            Pred::LengthGt(k) => r.len() > *k,
            Pred::LengthBetween(lo, hi) => r.len() >= *lo && r.len() <= *hi,
            Pred::IsNull => false, // we don't have nulls in this demo
            Pred::InSet(xs) => xs.iter().any(|x| r == x.as_str()),
        }
    }
}

// ============================================================================
//                              Pruning
// ============================================================================

struct PruneOut {
    kept_chunks: HashSet<usize>,
}

fn prune(
    pred: &Pred,
    stats: &[ChunkStats],
    presence: &[DictPresence],
    blooms: &[HybridBloom],
    ubiq: &UbiquitousBigrams,
    dv: &vortex_onpair::decode::DecodeView<'_>,
    index: &DictIndex,
    _rows: &[String],
    _cs: usize,
) -> PruneOut {
    let nch = stats.len();
    let mut kept = HashSet::with_capacity(nch);

    for c in 0..nch {
        let keep = chunk_might_match(pred, &stats[c], &presence[c], &blooms[c], ubiq, dv, index);
        if keep {
            kept.insert(c);
        }
    }
    PruneOut { kept_chunks: kept }
}

fn chunk_might_match(
    pred: &Pred,
    stats: &ChunkStats,
    presence: &DictPresence,
    bloom: &HybridBloom,
    ubiq: &UbiquitousBigrams,
    dv: &vortex_onpair::decode::DecodeView<'_>,
    index: &DictIndex,
) -> bool {
    match pred {
        Pred::Eq(x) => {
            let xb = x.as_bytes();
            stats.min.as_slice() <= xb
                && xb <= stats.max.as_slice()
                && presence.might_eq(dv, index, xb)
        }
        Pred::Lt(x) => {
            // chunk must have some row < x: min < x
            stats.min.as_slice() < x.as_bytes()
        }
        Pred::Gte(x) => stats.max.as_slice() >= x.as_bytes(),
        Pred::Between(a, b) => {
            !(stats.max.as_slice() < a.as_bytes() || stats.min.as_slice() > b.as_bytes())
        }
        Pred::Prefix(p) => {
            // Range [p, p + 0xff*]: a chunk overlaps iff min <= p+ff... and max >= p
            let pb = p.as_bytes();
            // chunk's max must be >= p, and chunk's min must be <= p+ff..ff
            // simpler: any string in [p, p+'~') overlaps
            if stats.max.as_slice() < pb {
                return false;
            }
            // Construct upper bound: p with 0xff appended max_len times
            let mut upper = p.clone().into_bytes();
            upper.extend(std::iter::repeat(0xffu8).take(stats.max_len));
            if stats.min.as_slice() > upper.as_slice() {
                return false;
            }
            // Fallback: also check DictPresence prefix
            presence.might_starts_with(dv, index, pb)
        }
        Pred::Suffix(s) => {
            // No reliable min/max pruning for suffixes — fall back to bloom
            bloom.might_contain(dv, index, presence, ubiq, s.as_bytes())
        }
        Pred::Contains(s) => bloom.might_contain(dv, index, presence, ubiq, s.as_bytes()),
        Pred::PrefixSuffix(p, s) => {
            // Combine: prefix range check + bloom for suffix
            let pb = p.as_bytes();
            if stats.max.as_slice() < pb {
                return false;
            }
            let mut upper = p.clone().into_bytes();
            upper.extend(std::iter::repeat(0xffu8).take(stats.max_len));
            if stats.min.as_slice() > upper.as_slice() {
                return false;
            }
            // Suffix check via bloom (treat as contains)
            bloom.might_contain(dv, index, presence, ubiq, s.as_bytes())
        }
        Pred::SingleWildcard(p, s) => {
            // Use bloom for both anchored parts (treat as substring needle for each)
            if !p.is_empty() && !bloom.might_contain(dv, index, presence, ubiq, p.as_bytes()) {
                return false;
            }
            if !s.is_empty() && !bloom.might_contain(dv, index, presence, ubiq, s.as_bytes()) {
                return false;
            }
            true
        }
        Pred::MultiFragment(frags) => {
            // Each fragment must be in the chunk
            for f in frags {
                if !bloom.might_contain(dv, index, presence, ubiq, f.as_bytes()) {
                    return false;
                }
            }
            true
        }
        Pred::LengthGt(k) => stats.max_len > *k,
        Pred::LengthBetween(lo, hi) => !(stats.max_len < *lo || stats.min_len > *hi),
        Pred::IsNull => stats.null_count > 0,
        Pred::InSet(xs) => xs.iter().any(|x| {
            let xb = x.as_bytes();
            stats.min.as_slice() <= xb
                && xb <= stats.max.as_slice()
                && presence.might_eq(dv, index, xb)
        }),
    }
}

// ============================================================================
//                              Workload
// ============================================================================

fn build_workload(
    rows: &[String],
    rng: &mut Splitmix64,
    n: usize,
) -> Vec<(&'static str, Vec<Pred>)> {
    let mut w = Vec::<(&'static str, Vec<Pred>)>::new();

    // Equality: pick real rows
    let mut eq = Vec::new();
    for _ in 0..n {
        let r = &rows[(rng.next() as usize) % rows.len()];
        eq.push(Pred::Eq(r.clone()));
    }
    w.push(("Eq: col = 'x' (real value)", eq));

    // Range comparisons
    let mut lt = Vec::new();
    for _ in 0..n {
        let r = &rows[(rng.next() as usize) % rows.len()];
        lt.push(Pred::Lt(r.clone()));
    }
    w.push(("Lt: col < 'x'", lt));

    let mut between = Vec::new();
    for _ in 0..n {
        let mut a = rows[(rng.next() as usize) % rows.len()].clone();
        let mut b = rows[(rng.next() as usize) % rows.len()].clone();
        if a > b {
            std::mem::swap(&mut a, &mut b);
        }
        between.push(Pred::Between(a, b));
    }
    w.push(("Between: col BETWEEN a AND b", between));

    // Prefix
    let mut prefix = Vec::new();
    for _ in 0..n {
        let r = rows[(rng.next() as usize) % rows.len()].as_bytes();
        if r.len() < 8 {
            continue;
        }
        let plen = 3 + (rng.next() as usize) % 6;
        if let Ok(p) = std::str::from_utf8(&r[..plen.min(r.len())]) {
            prefix.push(Pred::Prefix(p.to_string()));
        }
    }
    w.push(("Prefix: LIKE 'p%'", prefix));

    // Suffix
    let mut suffix = Vec::new();
    for _ in 0..n {
        let r = rows[(rng.next() as usize) % rows.len()].as_bytes();
        if r.len() < 8 {
            continue;
        }
        let slen = 3 + (rng.next() as usize) % 6;
        let start = r.len().saturating_sub(slen);
        if let Ok(s) = std::str::from_utf8(&r[start..]) {
            suffix.push(Pred::Suffix(s.to_string()));
        }
    }
    w.push(("Suffix: LIKE '%s'", suffix));

    // Contains (substring)
    let mut contains = Vec::new();
    for _ in 0..n {
        let r = rows[(rng.next() as usize) % rows.len()].as_bytes();
        if r.len() < 10 {
            continue;
        }
        let slen = 5 + (rng.next() as usize) % 8;
        let start = (rng.next() as usize) % (r.len() - slen);
        if let Ok(s) = std::str::from_utf8(&r[start..start + slen]) {
            contains.push(Pred::Contains(s.to_string()));
        }
    }
    w.push(("Contains: LIKE '%x%'", contains));

    // Prefix + suffix
    let mut psufx = Vec::new();
    for _ in 0..n {
        let r = rows[(rng.next() as usize) % rows.len()].as_bytes();
        if r.len() < 14 {
            continue;
        }
        let plen = 3 + (rng.next() as usize) % 4;
        let slen = 3 + (rng.next() as usize) % 4;
        if let (Ok(p), Ok(s)) = (
            std::str::from_utf8(&r[..plen]),
            std::str::from_utf8(&r[r.len() - slen..]),
        ) {
            psufx.push(Pred::PrefixSuffix(p.to_string(), s.to_string()));
        }
    }
    w.push(("PrefixSuffix: LIKE 'a%b'", psufx));

    // Single wildcard (substring with _)
    let mut sw = Vec::new();
    for _ in 0..n {
        let r = rows[(rng.next() as usize) % rows.len()].as_bytes();
        if r.len() < 12 {
            continue;
        }
        let pos = 3 + (rng.next() as usize) % (r.len() - 10);
        let plen = 3;
        let slen = 3;
        if let (Ok(p), Ok(s)) = (
            std::str::from_utf8(&r[pos..pos + plen]),
            std::str::from_utf8(&r[pos + plen + 1..pos + plen + 1 + slen]),
        ) {
            sw.push(Pred::SingleWildcard(p.to_string(), s.to_string()));
        }
    }
    w.push(("Wildcard: LIKE '%a_b%' (1 char)", sw));

    // Multi-fragment substring
    let mut mf = Vec::new();
    for _ in 0..n {
        let r = rows[(rng.next() as usize) % rows.len()].as_bytes();
        if r.len() < 16 {
            continue;
        }
        let plen1 = 3 + (rng.next() as usize) % 4;
        let start1 = (rng.next() as usize) % (r.len() / 3);
        let start2 = start1 + plen1 + 2 + (rng.next() as usize) % 4;
        let plen2 = 3 + (rng.next() as usize) % 4;
        if start2 + plen2 > r.len() {
            continue;
        }
        if let (Ok(f1), Ok(f2)) = (
            std::str::from_utf8(&r[start1..start1 + plen1]),
            std::str::from_utf8(&r[start2..start2 + plen2]),
        ) {
            mf.push(Pred::MultiFragment(vec![f1.to_string(), f2.to_string()]));
        }
    }
    w.push(("MultiFragment: LIKE '%a%b%'", mf));

    // Length predicates
    let avg_len = rows.iter().map(String::len).sum::<usize>() / rows.len().max(1);
    let lg: Vec<Pred> = (0..n).map(|i| Pred::LengthGt(avg_len + (i % 20))).collect();
    w.push(("LengthGt: LENGTH(col) > k", lg));

    let lb: Vec<Pred> = (0..n)
        .map(|_| {
            let lo = (rng.next() as usize) % avg_len;
            let hi = lo + 10 + (rng.next() as usize) % 50;
            Pred::LengthBetween(lo, hi)
        })
        .collect();
    w.push(("LengthBetween: LENGTH BETWEEN lo AND hi", lb));

    // IN set
    let mut in_set = Vec::new();
    for _ in 0..n {
        let k = 3 + (rng.next() as usize) % 5;
        let xs: Vec<String> = (0..k)
            .map(|_| rows[(rng.next() as usize) % rows.len()].clone())
            .collect();
        in_set.push(Pred::InSet(xs));
    }
    w.push(("InSet: col IN (3-7 values)", in_set));

    // IS NULL (will always be 0% pruned since we have no nulls)
    w.push(("IsNull: col IS NULL", vec![Pred::IsNull]));

    w
}

// ============================================================================
//                              Helpers
// ============================================================================

fn load_column(paths: &[PathBuf], col_name: &str, max_rows: usize) -> anyhow::Result<Vec<String>> {
    let mut out: Vec<String> = Vec::with_capacity(max_rows.max(1));
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
        let mut reader = builder
            .with_projection(mask)
            .with_batch_size(8192)
            .build()?;
        while let Some(batch) = reader.next() {
            let batch = batch?;
            let col = batch.column(0);
            let want = if max_rows > 0 {
                max_rows.saturating_sub(out.len())
            } else {
                col.len()
            };
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
