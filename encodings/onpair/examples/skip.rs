// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
//! `skip` — one-tool design advisor for chunk-level skip indexes on a
//! Parquet string column.
//!
//! ```bash
//! cargo run --release --example skip -p vortex-onpair -- \
//!     --parquet /path/to/your.parquet --column URL --max-rows 1000000
//! ```
//!
//! It loads the column, generates a representative workload (any
//! `--contains` / `--starts-with` literals you supply, plus auto-
//! sampled real-substring / real-prefix / rare needles), then sweeps
//! every combination of `(variant × chunk_size × bits_per_row)` you
//! enable. The output is three blocks:
//!
//! 1. **Sweep table** — one row per measured configuration with
//!    bytes/row, `Pr[keep]`, and `vs_floor` per workload category.
//! 2. **Pareto frontier** — the configurations that are not dominated
//!    by any cheaper-or-tighter alternative.
//! 3. **Recommendation** — concrete `(variant, chunk_size,
//!    bits_per_row)` choices for three operating points: *cheap*
//!    (≤ 2 B/row), *balanced* (≤ 5 B/row), *tight* (≤ 16 B/row),
//!    aimed at the substring-pruning workload.
//!
//! Add `--csv path.csv` to also dump the full sweep table for offline
//! analysis (one row per `(variant, chunk_size, bits_per_row,
//! workload_category)`).
//!
//! Variants:
//!
//! * `A`  — `DictPresence` (OnPair-specific dict bitmap, ~0.5 B/row)
//! * `B`  — `TrigramBloom` (codes-agnostic byte trigrams)
//! * `C`  — `SeamBloom` + A (OnPair-specific seam trigrams)
//! * `D`  — `TokenPairBloom` + A (OnPair-specific code pairs)
//! * `AB` — A AND B (the practical default)

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
#[command(
    about = "Sweep skip-index configurations and recommend a design for your column"
)]
struct Args {
    /// Parquet file path. Repeat for multiple files.
    #[arg(long, num_args = 1.., required = true)]
    parquet: Vec<PathBuf>,
    /// Column to analyse. Must be Utf8 / Utf8View / Binary / Large{Utf8,Binary}.
    #[arg(long, default_value = "URL")]
    column: String,
    /// Cap on total rows loaded across all files. 0 = unlimited.
    #[arg(long, default_value_t = 1_000_000)]
    max_rows: usize,

    /// Chunk sizes to sweep (rows per chunk), comma-separated.
    #[arg(long, default_value = "1024")]
    chunks: String,
    /// Bloom bits per row to sweep, comma-separated.
    #[arg(long, default_value = "8,16,32,64,128")]
    bits: String,
    /// Variants to evaluate, comma-separated. Subset of `A,B,C,D,AB`.
    #[arg(long, default_value = "A,B,C,D,AB")]
    variants: String,

    /// `LIKE '%S%'` needles, repeatable.
    #[arg(long)]
    contains: Vec<String>,
    /// `LIKE 'S%'` needles, repeatable.
    #[arg(long)]
    starts_with: Vec<String>,
    /// Auto-generated substring needles sampled from real rows.
    #[arg(long, default_value_t = 200)]
    auto_substrings: usize,
    /// Auto-generated prefix needles sampled from real rows.
    #[arg(long, default_value_t = 50)]
    auto_prefixes: usize,
    /// Auto-generated synthetic rare-substring needles.
    #[arg(long, default_value_t = 50)]
    auto_rare: usize,
    /// PRNG seed for needle sampling.
    #[arg(long, default_value_t = 0x9e37_79b9_7f4a_7c15_u64)]
    seed: u64,

    /// Write full per-(variant, chunk_size, bits, category) CSV here.
    #[arg(long)]
    csv: Option<PathBuf>,

    /// Suppress the sweep-table block (recommendation only).
    #[arg(long)]
    quiet: bool,
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

#[derive(Default, Clone)]
struct Bucket {
    n_q: usize,
    n_c: usize,
    real: usize,
    kept: usize,
}
impl Bucket {
    fn real_pct(&self) -> f64 {
        if self.n_c == 0 { 0.0 } else { 100.0 * self.real as f64 / self.n_c as f64 }
    }
    fn kept_pct(&self) -> f64 {
        if self.n_c == 0 { 0.0 } else { 100.0 * self.kept as f64 / self.n_c as f64 }
    }
    fn vs_floor(&self) -> f64 {
        self.kept_pct() - self.real_pct()
    }
}

#[derive(Clone)]
struct Row {
    variant: &'static str,
    chunk_size: usize,
    bits: usize,
    bytes_per_row: f64,
    build_ns_per_row: f64,
    eval_ns_per_q: u128,
    by_cat: BTreeMap<&'static str, Bucket>,
    total: Bucket,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let chunk_sizes: Vec<usize> = parse_csv(&args.chunks)?;
    let bit_settings: Vec<usize> = parse_csv(&args.bits)?;
    let variants: Vec<&'static str> = args
        .variants
        .split(',')
        .map(str::trim)
        .map(|s| match s {
            "A" => Ok("A"),
            "B" => Ok("B"),
            "C" => Ok("C"),
            "D" => Ok("D"),
            "AB" => Ok("AB"),
            other => anyhow::bail!("unknown variant {other:?} (want A|B|C|D|AB)"),
        })
        .collect::<Result<_, _>>()?;

    // ----------------------------------------------------------- load + compress
    eprintln!("Loading column {:?} ...", args.column);
    let t0 = Instant::now();
    let rows = load_column(&args.parquet, &args.column, args.max_rows)?;
    eprintln!(
        "loaded {} rows in {:?} ({} raw bytes)",
        rows.len(),
        t0.elapsed(),
        rows.iter().map(String::len).sum::<usize>(),
    );

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

    // ----------------------------------------------------------- workload
    let workload = build_workload(&args, &rows);
    eprintln!("workload: {} queries", workload.len());

    let mut csv_writer = args.csv.as_ref().map(|p| -> anyhow::Result<_> {
        let mut w = BufWriter::new(File::create(p)?);
        writeln!(
            w,
            "variant,chunk_size,bits_per_row,category,bytes_per_row,real_pct,kept_pct,vs_floor_pp,build_ns_per_row,eval_ns_per_q"
        )?;
        Ok(w)
    }).transpose()?;

    // ------------------------------------------------------------ sweep
    let mut results: Vec<Row> = Vec::new();
    for &cs in &chunk_sizes {
        let nch = rows.len() / cs;
        if nch == 0 {
            continue;
        }
        let n_aligned = nch * cs;

        // A is independent of `bits`; build once per chunk_size.
        let t0 = Instant::now();
        let presence: Vec<DictPresence> = (0..nch)
            .map(|c| DictPresence::build(&dv, c * cs, (c + 1) * cs))
            .collect();
        let a_build_ns = t0.elapsed().as_nanos();
        let a_bytes: usize = presence.iter().map(DictPresence::byte_size).sum();

        // Evaluate A once. bits=0 is sentinel.
        if variants.contains(&"A") {
            results.push(eval(
                "A", cs, 0, n_aligned,
                a_bytes, a_build_ns,
                &workload, nch, &rows,
                |q, c| match q {
                    Pred::Contains(s) => presence[c].might_contain(&dv, s.as_bytes()),
                    Pred::StartsWith(s) => presence[c].might_starts_with(&dv, &index, s.as_bytes()),
                },
            ));
        }

        for &bits in &bit_settings {
            // ---- B = TrigramBloom (codes-agnostic) ----
            let need_b = variants.contains(&"B") || variants.contains(&"AB");
            let (bs, b_bytes, b_build_ns) = if need_b {
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
                (bs, b_bytes, b_build_ns)
            } else {
                (Vec::new(), 0, 0)
            };

            if variants.contains(&"B") {
                results.push(eval(
                    "B", cs, bits, n_aligned,
                    b_bytes, b_build_ns,
                    &workload, nch, &rows,
                    |q, c| bs[c].might_contain(q.bytes()),
                ));
            }
            if variants.contains(&"AB") {
                let ab_bytes = a_bytes + b_bytes;
                let ab_build_ns = a_build_ns + b_build_ns;
                results.push(eval(
                    "AB", cs, bits, n_aligned,
                    ab_bytes, ab_build_ns,
                    &workload, nch, &rows,
                    |q, c| {
                        let pa = match q {
                            Pred::Contains(s) => presence[c].might_contain(&dv, s.as_bytes()),
                            Pred::StartsWith(s) => {
                                presence[c].might_starts_with(&dv, &index, s.as_bytes())
                            }
                        };
                        pa && bs[c].might_contain(q.bytes())
                    },
                ));
            }

            // ---- C = SeamBloom + A ----
            if variants.contains(&"C") {
                let t0 = Instant::now();
                let cs_idx: Vec<SeamBloom> = (0..nch)
                    .map(|c| SeamBloom::build(&dv, c * cs, (c + 1) * cs, bits))
                    .collect();
                let c_build = t0.elapsed().as_nanos() + a_build_ns;
                let c_bytes = cs_idx.iter().map(SeamBloom::byte_size).sum::<usize>() + a_bytes;
                results.push(eval(
                    "C", cs, bits, n_aligned,
                    c_bytes, c_build,
                    &workload, nch, &rows,
                    |q, c| cs_idx[c].might_contain(&dv, &presence[c], q.bytes()),
                ));
            }

            // ---- D = TokenPairBloom + A ----
            if variants.contains(&"D") {
                let t0 = Instant::now();
                let ds: Vec<TokenPairBloom> = (0..nch)
                    .map(|c| TokenPairBloom::build(&dv, c * cs, (c + 1) * cs, bits))
                    .collect();
                let d_build = t0.elapsed().as_nanos() + a_build_ns;
                let d_bytes = ds.iter().map(TokenPairBloom::byte_size).sum::<usize>() + a_bytes;
                results.push(eval(
                    "D", cs, bits, n_aligned,
                    d_bytes, d_build,
                    &workload, nch, &rows,
                    |q, c| match q {
                        Pred::Contains(s) => {
                            ds[c].might_contain(&dv, &index, &presence[c], s.as_bytes())
                        }
                        Pred::StartsWith(s) => {
                            presence[c].might_starts_with(&dv, &index, s.as_bytes())
                        }
                    },
                ));
            }

            if !variants.contains(&"B")
                && !variants.contains(&"C")
                && !variants.contains(&"D")
                && !variants.contains(&"AB")
            {
                break; // only A enabled — `bits` doesn't matter, run once
            }
        }
    }

    // ----------------------------------------------------------- CSV
    if let Some(w) = csv_writer.as_mut() {
        for r in &results {
            for (cat, b) in &r.by_cat {
                writeln!(
                    w,
                    "{},{},{},{},{:.4},{:.4},{:.4},{:.4},{:.1},{}",
                    r.variant, r.chunk_size, r.bits, cat,
                    r.bytes_per_row, b.real_pct(), b.kept_pct(), b.vs_floor(),
                    r.build_ns_per_row, r.eval_ns_per_q,
                )?;
            }
            writeln!(
                w,
                "{},{},{},TOTAL,{:.4},{:.4},{:.4},{:.4},{:.1},{}",
                r.variant, r.chunk_size, r.bits,
                r.bytes_per_row, r.total.real_pct(), r.total.kept_pct(),
                r.total.vs_floor(), r.build_ns_per_row, r.eval_ns_per_q,
            )?;
        }
        w.flush()?;
    }

    // ----------------------------------------------------------- sweep table
    if !args.quiet {
        println!();
        println!("=== Sweep results ===");
        println!(
            "{:<5} {:>5} {:>5}  {:>6} {:>9}  {:>9}  {:>9}  {:>9}  {:>9}  {:>9}",
            "var", "chunk", "bits", "B/row",
            "subst", "prefix", "rare", "clickbench", "TOTAL", "eval_us",
        );
        println!("{}", "-".repeat(96));
        for r in &results {
            println!(
                "{:<5} {:>5} {:>5}  {:>6.2} {:>+8.2}pp {:>+8.2}pp {:>+8.2}pp {:>+9.2}pp {:>+8.2}pp {:>9.1}",
                r.variant, r.chunk_size, r.bits, r.bytes_per_row,
                cat_vs_floor(r, "auto/substring"),
                cat_vs_floor(r, "auto/prefix"),
                cat_vs_floor(r, "auto/rare"),
                cat_vs_floor(r, "clickbench"),
                r.total.vs_floor(),
                r.eval_ns_per_q as f64 / 1000.0,
            );
        }
        println!();
        println!("Numbers are `vs_floor` = (Pr[keep] − real_rate) in percentage points.");
        println!("Floor for substring on this data: {:.2}%", category_floor(&results, "auto/substring"));
        println!("Lower vs_floor = tighter prefilter. 0 pp = optimal pruning.");
    }

    // ----------------------------------------------------------- Pareto + reco
    let pareto = pareto_for_substring(&results);
    println!();
    println!("=== Pareto frontier (substring workload) ===");
    println!("{:<5} {:>5} {:>5}  {:>7}  {:>8}  {:>10}",
        "var", "chunk", "bits", "B/row", "vs_floor", "eval_us");
    println!("{}", "-".repeat(50));
    for r in &pareto {
        println!(
            "{:<5} {:>5} {:>5}  {:>7.2}  {:>+7.2}pp  {:>10.1}",
            r.variant, r.chunk_size, r.bits, r.bytes_per_row,
            cat_vs_floor(r, "auto/substring"),
            r.eval_ns_per_q as f64 / 1000.0,
        );
    }

    println!();
    println!("=== Recommendation ===");
    let rec_cheap = pareto.iter().rfind(|r| r.bytes_per_row <= 2.0);
    let rec_bal = pareto.iter().rfind(|r| r.bytes_per_row <= 5.0);
    let rec_tight = pareto.iter().rfind(|r| r.bytes_per_row <= 16.0);
    print_recommendation("cheap   (≤  2 B/row)", rec_cheap);
    print_recommendation("balanced(≤  5 B/row)", rec_bal);
    print_recommendation("tight   (≤ 16 B/row)", rec_tight);
    println!();
    println!("Eq-only workloads: A=DictPresence at 0.5 B/row is sufficient.");

    Ok(())
}

fn print_recommendation(label: &str, r: Option<&&Row>) {
    match r {
        Some(r) => println!(
            "  {label}: variant={}  chunk_size={}  bits/row={}  ⇒ {:.2} B/row, {:+.2}pp substring",
            r.variant, r.chunk_size, r.bits, r.bytes_per_row,
            cat_vs_floor(r, "auto/substring"),
        ),
        None => println!("  {label}: (no Pareto point within budget — increase --bits or --variants)"),
    }
}

fn cat_vs_floor(r: &Row, cat: &'static str) -> f64 {
    r.by_cat.get(cat).map(Bucket::vs_floor).unwrap_or(0.0)
}

fn category_floor(results: &[Row], cat: &'static str) -> f64 {
    results.first()
        .and_then(|r| r.by_cat.get(cat))
        .map(Bucket::real_pct)
        .unwrap_or(0.0)
}

fn pareto_for_substring(results: &[Row]) -> Vec<&Row> {
    // A point is on the Pareto frontier if no other point has both
    // strictly lower bytes_per_row AND strictly lower vs_floor on the
    // substring workload.
    let mut pareto: Vec<&Row> = results
        .iter()
        .filter(|r| {
            let me_bytes = r.bytes_per_row;
            let me_vs = cat_vs_floor(r, "auto/substring");
            !results.iter().any(|other| {
                let o_bytes = other.bytes_per_row;
                let o_vs = cat_vs_floor(other, "auto/substring");
                (o_bytes < me_bytes && o_vs <= me_vs)
                    || (o_bytes <= me_bytes && o_vs < me_vs)
            })
        })
        .collect();
    pareto.sort_by(|a, b| a.bytes_per_row.partial_cmp(&b.bytes_per_row).unwrap());
    pareto
}

#[allow(clippy::too_many_arguments)]
fn eval<F: FnMut(&Pred, usize) -> bool>(
    variant: &'static str,
    chunk_size: usize,
    bits: usize,
    n_aligned: usize,
    bytes: usize,
    build_ns: u128,
    workload: &[(&'static str, Pred)],
    nch: usize,
    rows: &[String],
    mut keep_fn: F,
) -> Row {
    let mut by_cat: BTreeMap<&'static str, Bucket> = BTreeMap::new();
    let mut total = Bucket::default();
    let t0 = Instant::now();
    for (tag, q) in workload {
        for c in 0..nch {
            let lo = c * chunk_size;
            let hi = lo + chunk_size;
            let real = q.truly_matches(&rows[lo..hi]);
            let k = keep_fn(q, c);
            assert!(
                !real || k,
                "{variant} bits={bits} cs={chunk_size}: FN on chunk {c}, {q:?}"
            );
            let bucket = by_cat.entry(tag).or_default();
            bucket.n_c += 1;
            bucket.real += real as usize;
            bucket.kept += k as usize;
            total.n_c += 1;
            total.real += real as usize;
            total.kept += k as usize;
        }
        by_cat.entry(tag).or_default().n_q += 1;
        total.n_q += 1;
    }
    let eval_ns = t0.elapsed().as_nanos();
    Row {
        variant,
        chunk_size,
        bits,
        bytes_per_row: bytes as f64 / n_aligned as f64,
        build_ns_per_row: build_ns as f64 / n_aligned as f64,
        eval_ns_per_q: eval_ns / workload.len().max(1) as u128,
        by_cat,
        total,
    }
}

fn build_workload(args: &Args, rows: &[String]) -> Vec<(&'static str, Pred)> {
    let mut workload: Vec<(&'static str, Pred)> = Vec::new();
    // User-supplied literals.
    for s in &args.contains {
        workload.push(("user/contains", Pred::Contains(s.clone())));
    }
    for s in &args.starts_with {
        workload.push(("user/prefix", Pred::StartsWith(s.clone())));
    }
    // ClickBench-style needles. The four LIKE queries in
    // ClickBench's `clickhouse/queries.sql` all hit `URL LIKE '%google%'`
    // or its negation `URL NOT LIKE '%.google.%'`.
    workload.push(("clickbench", Pred::Contains("google".to_string())));
    workload.push(("clickbench", Pred::Contains(".google.".to_string())));
    // Auto-sampled.
    let mut rng = Splitmix64::new(args.seed);
    for _ in 0..args.auto_substrings {
        if let Some(p) = sample_substring(rows, &mut rng) {
            workload.push(("auto/substring", p));
        }
    }
    for _ in 0..args.auto_prefixes {
        if let Some(p) = sample_prefix(rows, &mut rng) {
            workload.push(("auto/prefix", p));
        }
    }
    for _ in 0..args.auto_rare {
        workload.push(("auto/rare", sample_rare(&mut rng)));
    }
    workload
}

fn parse_csv(s: &str) -> anyhow::Result<Vec<usize>> {
    s.split(',').map(|x| x.trim().parse::<usize>().context("parse"))
        .collect()
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
