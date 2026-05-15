// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
//! Parameter sweep across chunk skip-index variants. Emits a CSV with
//! one row per `(variant, chunk_size, bits_per_row, workload_category)`
//! tuple plus a stdout summary so the operator can find the
//! Pareto-optimal configuration for their data.
//!
//! Run:
//!
//! ```bash
//! cargo run --release --example skip_sweep -p vortex-onpair -- \
//!     --parquet hits_0.parquet --column URL --max-rows 1000000 \
//!     --chunks 256,1024,4096 --bits 8,16,32,64,128 \
//!     --out skip_sweep_results.csv
//! ```
//!
//! Variants compared:
//!
//! * **A**  — `DictPresence` (OnPair-specific, 1 bit per dict id)
//! * **B**  — `TrigramBloom` (compression-agnostic byte trigrams)
//! * **C**  — `SeamBloom`     (OnPair-specific seam trigrams) + A
//! * **D**  — `TokenPairBloom` (OnPair-specific code pairs) + A
//! * **AB** — A AND B (the practical pair tier-1 + tier-2)
//!
//! Workload categories: `clickbench` (ClickBench queries Q21–Q24),
//! `auto/substring` (random substrings from real rows),
//! `auto/prefix` (random prefixes), `auto/rare` (synthetic rare).

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::print_stdout,
    clippy::use_debug
)]

use std::collections::BTreeMap;
use std::fs::File;
use std::io::BufWriter;
use std::io::Write;
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
use vortex_onpair::skip::SeamBloom;
use vortex_onpair::skip::TokenPairBloom;
use vortex_onpair::skip::TrigramBloom;

#[derive(Parser)]
#[command(about = "Sweep skip-index variants and emit CSV")]
struct Args {
    #[arg(long, num_args = 1.., required = true)]
    parquet: Vec<PathBuf>,
    #[arg(long, default_value = "URL")]
    column: String,
    /// Comma-separated chunk sizes to sweep, e.g. `256,1024,4096`.
    #[arg(long, default_value = "1024")]
    chunks: String,
    /// Comma-separated bits-per-row for Blooms, e.g. `8,16,32,64`.
    #[arg(long, default_value = "8,16,32,64,128")]
    bits: String,
    #[arg(long, default_value_t = 1_000_000)]
    max_rows: usize,
    #[arg(long, default_value_t = 200)]
    auto_substrings: usize,
    #[arg(long, default_value_t = 50)]
    auto_prefixes: usize,
    #[arg(long, default_value_t = 50)]
    auto_rare: usize,
    #[arg(long, default_value_t = 0x9e37_79b9_7f4a_7c15_u64)]
    seed: u64,
    /// CSV output path. If unset, only print the stdout summary.
    #[arg(long)]
    out: Option<PathBuf>,
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

#[derive(Default, Clone, Copy)]
struct Stats {
    n_q: usize,
    n_c: usize,
    real: usize,
    kept: usize,
    build_ns: u128,
    eval_ns: u128,
    bytes: usize,
}

fn pct(n: usize, d: usize) -> f64 {
    if d == 0 { 0.0 } else { 100.0 * n as f64 / d as f64 }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let chunk_sizes: Vec<usize> = args
        .chunks
        .split(',')
        .map(|s| s.trim().parse::<usize>())
        .collect::<Result<_, _>>()?;
    let bit_settings: Vec<usize> = args
        .bits
        .split(',')
        .map(|s| s.trim().parse::<usize>())
        .collect::<Result<_, _>>()?;

    eprintln!("Loading column {:?} ...", args.column);
    let t0 = Instant::now();
    let rows = load_column(&args.parquet, &args.column, args.max_rows)?;
    eprintln!("loaded {} rows in {:?}", rows.len(), t0.elapsed());

    // OnPair-compress once. The same compressed array is sliced into
    // chunks of different sizes below; the column dict is shared.
    let t0 = Instant::now();
    let varbin = VarBinArray::from_iter(
        rows.iter().map(|s| Some(s.as_bytes())),
        DType::Utf8(Nullability::NonNullable),
    );
    let arr =
        onpair_compress(&varbin, varbin.len(), varbin.dtype(), DEFAULT_DICT12_CONFIG).unwrap();
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let inputs = OwnedDecodeInputs::collect(arr.as_view(), &mut ctx).unwrap();
    let dv = inputs.view();
    let index = DictIndex::build(&dv);
    eprintln!(
        "OnPair-compressed in {:?}; dict_size={}",
        t0.elapsed(),
        dv.dict_table.len(),
    );

    // Build workload (ClickBench + auto-sampled).
    let mut rng = Splitmix64::new(args.seed);
    let mut workload: Vec<(&'static str, Pred)> = Vec::new();
    workload.push(("clickbench", Pred::Contains("google".to_string())));
    workload.push(("clickbench", Pred::Contains(".google.".to_string())));
    for _ in 0..args.auto_substrings {
        if let Some(p) = sample_substring(&rows, &mut rng) {
            workload.push(("auto/substring", p));
        }
    }
    for _ in 0..args.auto_prefixes {
        if let Some(p) = sample_prefix(&rows, &mut rng) {
            workload.push(("auto/prefix", p));
        }
    }
    for _ in 0..args.auto_rare {
        workload.push(("auto/rare", sample_rare(&mut rng)));
    }
    eprintln!("workload: {} queries", workload.len());

    let mut csv = args.out.as_ref().map(|p| -> anyhow::Result<_> {
        let f = BufWriter::new(File::create(p)?);
        Ok(f)
    }).transpose()?;
    if let Some(w) = csv.as_mut() {
        writeln!(w, "variant,chunk_size,bits_per_row,category,bytes_per_row,real_pct,kept_pct,vs_floor_pp,build_ns_per_row,eval_ns_per_q")?;
    }

    println!();
    println!(
        "{:<6} {:>10} {:>9} {:<14} {:>9} {:>9} {:>9} {:>11} {:>11} {:>11}",
        "var", "chunk", "bits/row", "category",
        "b/row", "real%", "kept%", "vs_floor",
        "build_ns", "eval_ns",
    );
    println!("{}", "-".repeat(110));

    for &cs in &chunk_sizes {
        let nrows = (rows.len() / cs) * cs;
        let nch = nrows / cs;
        if nch == 0 {
            continue;
        }

        // A is independent of `bits_per_row`; build it once per chunk_size.
        let t0 = Instant::now();
        let presence: Vec<DictPresence> = (0..nch)
            .map(|c| DictPresence::build(&dv, c * cs, (c + 1) * cs))
            .collect();
        let a_build_ns = t0.elapsed().as_nanos();
        let a_bytes: usize = presence.iter().map(DictPresence::byte_size).sum();

        // ---- evaluate A (constant across `bits` axis) ----
        evaluate_variant(
            "A", cs, 0, &workload, nch, cs, a_bytes, a_build_ns, &rows, csv.as_mut(),
            |q, c| match q {
                Pred::Contains(s) => presence[c].might_contain(&dv, s.as_bytes()),
                Pred::StartsWith(s) => presence[c].might_starts_with(&dv, &index, s.as_bytes()),
            },
        )?;

        for &bits in &bit_settings {
            // ---- B = TrigramBloom (raw bytes, compression-agnostic) ----
            let t0 = Instant::now();
            let bs: Vec<TrigramBloom> = (0..nch)
                .map(|c| {
                    TrigramBloom::build_from_strings(
                        rows[c * cs..(c + 1) * cs].iter().map(String::as_bytes),
                        cs,
                        bits,
                    )
                })
                .collect();
            let b_build_ns = t0.elapsed().as_nanos();
            let b_bytes: usize = bs.iter().map(TrigramBloom::byte_size).sum();

            evaluate_variant(
                "B", cs, bits, &workload, nch, cs, b_bytes, b_build_ns, &rows, csv.as_mut(),
                |q, c| bs[c].might_contain(q.bytes()),
            )?;

            // ---- C = SeamBloom + DictPresence ----
            let t0 = Instant::now();
            let cs_idx: Vec<SeamBloom> = (0..nch)
                .map(|c| SeamBloom::build(&dv, c * cs, (c + 1) * cs, bits))
                .collect();
            let c_build_ns = t0.elapsed().as_nanos() + a_build_ns;
            let c_bytes: usize = cs_idx.iter().map(SeamBloom::byte_size).sum::<usize>() + a_bytes;

            evaluate_variant(
                "C", cs, bits, &workload, nch, cs, c_bytes, c_build_ns, &rows, csv.as_mut(),
                |q, c| cs_idx[c].might_contain(&dv, &presence[c], q.bytes()),
            )?;

            // ---- D = TokenPairBloom + DictPresence ----
            let t0 = Instant::now();
            let ds: Vec<TokenPairBloom> = (0..nch)
                .map(|c| TokenPairBloom::build(&dv, c * cs, (c + 1) * cs, bits))
                .collect();
            let d_build_ns = t0.elapsed().as_nanos() + a_build_ns;
            let d_bytes: usize = ds.iter().map(TokenPairBloom::byte_size).sum::<usize>() + a_bytes;

            evaluate_variant(
                "D", cs, bits, &workload, nch, cs, d_bytes, d_build_ns, &rows, csv.as_mut(),
                |q, c| match q {
                    Pred::Contains(s) => {
                        ds[c].might_contain(&dv, &index, &presence[c], s.as_bytes())
                    }
                    Pred::StartsWith(s) => {
                        presence[c].might_starts_with(&dv, &index, s.as_bytes())
                    }
                },
            )?;

            // ---- AB = A AND B ----
            let ab_bytes = a_bytes + b_bytes;
            let ab_build_ns = a_build_ns + b_build_ns;
            evaluate_variant(
                "AB", cs, bits, &workload, nch, cs, ab_bytes, ab_build_ns, &rows, csv.as_mut(),
                |q, c| {
                    let pa = match q {
                        Pred::Contains(s) => presence[c].might_contain(&dv, s.as_bytes()),
                        Pred::StartsWith(s) => {
                            presence[c].might_starts_with(&dv, &index, s.as_bytes())
                        }
                    };
                    pa && bs[c].might_contain(q.bytes())
                },
            )?;
        }
    }

    if let Some(mut w) = csv {
        w.flush()?;
    }
    println!();
    println!("CSV format:");
    println!("  variant            A | B | C | D | AB");
    println!("  chunk_size         rows per chunk");
    println!("  bits_per_row       Bloom sizing in bits / row (0 for A)");
    println!("  category           clickbench | auto/substring | auto/prefix | auto/rare | TOTAL");
    println!("  bytes_per_row      total bytes of this index / number of rows");
    println!("  real_pct           % (q,c) pairs with ≥ 1 real match  — the floor");
    println!("  kept_pct           % (q,c) pairs the prefilter still keeps");
    println!("  vs_floor_pp        kept_pct − real_pct  (0 pp = optimal)");
    println!("  build_ns_per_row   ns spent building the index, normalised by row");
    println!("  eval_ns_per_q      ns spent evaluating one query across all chunks");
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn evaluate_variant<F: FnMut(&Pred, usize) -> bool>(
    variant: &str,
    chunk_size: usize,
    bits: usize,
    workload: &[(&'static str, Pred)],
    nch: usize,
    cs: usize,
    bytes: usize,
    build_ns: u128,
    rows: &[String],
    mut csv: Option<&mut BufWriter<File>>,
    mut keep: F,
) -> anyhow::Result<()> {
    let nrows = nch * cs;
    let mut cats: BTreeMap<&'static str, Stats> = BTreeMap::new();
    let mut total = Stats::default();
    let t0 = Instant::now();
    for (tag, q) in workload {
        for c in 0..nch {
            let lo = c * cs;
            let hi = lo + cs;
            let real = q.truly_matches(&rows[lo..hi]);
            let k = keep(q, c);
            assert!(!real || k, "{variant}/{bits}b/cs={cs}: FN on chunk {c}, {q:?}");
            let cat = cats.entry(tag).or_default();
            cat.n_c += 1;
            cat.real += real as usize;
            cat.kept += k as usize;
            total.n_c += 1;
            total.real += real as usize;
            total.kept += k as usize;
        }
        cats.entry(tag).or_default().n_q += 1;
        total.n_q += 1;
    }
    let eval_ns = t0.elapsed().as_nanos();
    let eval_ns_per_q = eval_ns / workload.len().max(1) as u128;
    let bytes_per_row = bytes as f64 / nrows as f64;
    let build_ns_per_row = build_ns as f64 / nrows as f64;

    let mut print_one = |name: &str, s: &Stats, csv: Option<&mut BufWriter<File>>| -> anyhow::Result<()> {
        let real = pct(s.real, s.n_c);
        let kept = pct(s.kept, s.n_c);
        println!(
            "{:<6} {:>10} {:>9} {:<14} {:>9.3} {:>8.2}% {:>8.2}% {:>+8.2}pp {:>11.1} {:>11}",
            variant, chunk_size, bits, name,
            bytes_per_row, real, kept, kept - real,
            build_ns_per_row, eval_ns_per_q,
        );
        if let Some(w) = csv {
            writeln!(w, "{variant},{chunk_size},{bits},{name},{:.4},{real:.4},{kept:.4},{:.4},{:.4},{eval_ns_per_q}",
                bytes_per_row, kept - real, build_ns_per_row,
            )?;
        }
        Ok(())
    };
    for (tag, st) in &cats {
        print_one(tag, st, csv.as_deref_mut())?;
    }
    print_one("TOTAL", &total, csv.as_deref_mut())?;
    Ok(())
}

fn sample_substring(rows: &[String], rng: &mut Splitmix64) -> Option<Pred> {
    for _ in 0..16 {
        let i = (rng.next() as usize) % rows.len();
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

fn sample_prefix(rows: &[String], rng: &mut Splitmix64) -> Option<Pred> {
    for _ in 0..16 {
        let i = (rng.next() as usize) % rows.len();
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

fn load_column(paths: &[PathBuf], col_name: &str, max_rows: usize) -> anyhow::Result<Vec<String>> {
    let mut out: Vec<String> = Vec::with_capacity(max_rows.max(1));
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
            let want = if max_rows > 0 { max_rows.saturating_sub(out.len()) } else { col.len() };
            if want == 0 { break 'outer; }
            let pushed = push_strings(col, want, &mut out);
            anyhow::ensure!(pushed > 0, "unexpected column type: {:?}", col.data_type());
            if max_rows > 0 && out.len() >= max_rows { break 'outer; }
        }
    }
    Ok(out)
}

fn push_strings(col: &dyn ArrowArray, want: usize, out: &mut Vec<String>) -> usize {
    if let Some(s) = col.as_string_opt::<i32>() {
        let n = s.len().min(want);
        for i in 0..n { out.push(s.value(i).to_string()); }
        return n;
    }
    if let Some(s) = col.as_string_opt::<i64>() {
        let n = s.len().min(want);
        for i in 0..n { out.push(s.value(i).to_string()); }
        return n;
    }
    if let Some(s) = col.as_string_view_opt() {
        let n = s.len().min(want);
        for i in 0..n { out.push(s.value(i).to_string()); }
        return n;
    }
    if let Some(b) = col.as_binary_opt::<i32>() {
        let n = b.len().min(want);
        for i in 0..n { out.push(String::from_utf8_lossy(b.value(i)).into_owned()); }
        return n;
    }
    if let Some(b) = col.as_binary_opt::<i64>() {
        let n = b.len().min(want);
        for i in 0..n { out.push(String::from_utf8_lossy(b.value(i)).into_owned()); }
        return n;
    }
    if let Some(b) = col.as_binary_view_opt() {
        let n = b.len().min(want);
        for i in 0..n { out.push(String::from_utf8_lossy(b.value(i)).into_owned()); }
        return n;
    }
    0
}

struct Splitmix64(u64);
impl Splitmix64 {
    fn new(seed: u64) -> Self { Self(seed) }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}
