// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Run timed experiments comparing raw VarBin vs FSST string filtering.

use std::path::PathBuf;
use std::sync::LazyLock;
use std::time::Duration;
use std::time::Instant;

use anyhow::Result;
use clap::Args;
use serde::Deserialize;
use serde::Serialize;
use vortex::VortexSessionDefault;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::ConstantArray;
use vortex::array::arrays::VarBinArray;
use vortex::array::arrays::bool::BoolArrayExt;
use vortex::array::arrays::scalar_fn::ScalarFnFactoryExt;
use vortex::array::dtype::DType;
use vortex::array::dtype::Nullability;
use vortex::array::scalar_fn::fns::like::Like;
use vortex::array::scalar_fn::fns::like::LikeOptions;
use vortex::encodings::fsst::FSSTArray;
use vortex::encodings::fsst::fsst_compress;
use vortex::encodings::fsst::fsst_train_compressor;
use vortex::session::VortexSession;

use crate::data_prep::DatasetName;
use crate::query_miner::MinedQuery;
use crate::query_miner::{self};

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::default);

#[derive(Args)]
pub struct RunArgs {
    /// Which dataset to benchmark
    #[arg(value_enum)]
    pub dataset: DatasetName,

    /// Number of warmup iterations
    #[arg(long, default_value_t = 3)]
    pub warmup: usize,

    /// Number of timed iterations
    #[arg(long, default_value_t = 10)]
    pub iterations: usize,

    /// Only run queries matching this type (like_prefix, like_substr, regex_basic)
    #[arg(long)]
    pub query_type: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BenchResult {
    pub pattern: String,
    pub query_type: String,
    pub selectivity: String,
    pub fsst_difficulty: String,
    pub match_fraction: f64,
    pub raw_matches: usize,
    pub fsst_matches: usize,
    pub raw_scan_ms: f64,
    pub fsst_scan_ms: f64,
    pub decompress_like_ms: f64,
    pub speedup: f64,
    pub decompress_speedup: f64,
}

impl std::fmt::Display for BenchResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:40} {:12} {:6} raw={:8.3}ms fsst={:8.3}ms decomp+like={:8.3}ms speedup={:6.2}x decomp_speedup={:6.2}x matches={}",
            self.pattern,
            self.query_type,
            self.selectivity,
            self.raw_scan_ms,
            self.fsst_scan_ms,
            self.decompress_like_ms,
            self.speedup,
            self.decompress_speedup,
            self.raw_matches,
        )
    }
}

pub fn run(args: RunArgs) -> Result<()> {
    let strings = query_miner::load_strings(&args.dataset)?;
    let n = strings.len();
    println!("Loaded {n} strings for {}", args.dataset);

    // Build arrays
    let varbin = VarBinArray::from_iter_nonnull(
        strings.iter().map(|s| s.as_bytes()),
        DType::Utf8(Nullability::NonNullable),
    );
    let compressor = fsst_train_compressor(&varbin);
    let len = varbin.len();
    let dtype = varbin.dtype().clone();
    let fsst_array: FSSTArray = fsst_compress(varbin.clone(), len, &dtype, &compressor);

    // Load mined queries
    let queries = load_queries(&args.dataset)?;
    let queries: Vec<&MinedQuery> = queries
        .iter()
        .filter(|q| args.query_type.as_ref().is_none_or(|t| &q.query_type == t))
        .collect();

    println!(
        "Running {} queries ({} warmup, {} timed iterations)...\n",
        queries.len(),
        args.warmup,
        args.iterations
    );

    let mut results = Vec::new();

    for query in &queries {
        let pattern = &query.pattern;
        let is_like = query.query_type == "like_prefix"
            || query.query_type == "like_substr"
            || query.query_type == "like_suffix";

        if is_like {
            let result = bench_like(&varbin, &fsst_array, pattern, query, &args)?;
            results.push(result);
        } else {
            let result = bench_regex(&fsst_array, &strings, pattern, query, &args)?;
            results.push(result);
        }
    }

    // Print results
    println!("\n=== Results ===");
    for r in &results {
        println!("  {r}");
    }

    // Write results JSON
    let out_path = results_path(&args.dataset);
    std::fs::create_dir_all(out_path.parent().unwrap())?;
    let json = serde_json::to_string_pretty(&results)?;
    std::fs::write(&out_path, &json)?;
    println!("\nResults written to: {}", out_path.display());

    // Summary
    let like_results: Vec<&BenchResult> = results
        .iter()
        .filter(|r| r.query_type != "regex_basic")
        .collect();
    if !like_results.is_empty() {
        let avg_speedup =
            like_results.iter().map(|r| r.speedup).sum::<f64>() / like_results.len() as f64;
        println!("\nAverage FSST speedup for LIKE queries: {avg_speedup:.2}x");
    }

    Ok(())
}

fn bench_like(
    varbin: &VarBinArray,
    fsst_array: &FSSTArray,
    pattern: &str,
    query: &MinedQuery,
    args: &RunArgs,
) -> Result<BenchResult> {
    let len = varbin.len();
    let raw_arr = varbin.clone().into_array();
    let fsst_arr = fsst_array.clone().into_array();
    let pattern_arr = ConstantArray::new(pattern, len).into_array();

    // Warmup raw
    for _ in 0..args.warmup {
        run_like(&raw_arr, &pattern_arr, len)?;
    }

    // Time raw
    let mut raw_times = Vec::with_capacity(args.iterations);
    let mut raw_matches = 0;
    for _ in 0..args.iterations {
        let start = Instant::now();
        let result = run_like(&raw_arr, &pattern_arr, len)?;
        raw_times.push(start.elapsed());
        raw_matches = count_true_bits(&result)?;
    }

    // Warmup FSST
    for _ in 0..args.warmup {
        run_like(&fsst_arr, &pattern_arr, len)?;
    }

    // Time FSST
    let mut fsst_times = Vec::with_capacity(args.iterations);
    let mut fsst_matches = 0;
    for _ in 0..args.iterations {
        let start = Instant::now();
        let result = run_like(&fsst_arr, &pattern_arr, len)?;
        fsst_times.push(start.elapsed());
        fsst_matches = count_true_bits(&result)?;
    }

    // Warmup decompress+like
    for _ in 0..args.warmup {
        run_decompress_like(&fsst_arr, &pattern_arr, len)?;
    }

    // Time decompress+like (decompress FSST to canonical, then run LIKE on decompressed)
    let mut decompress_times = Vec::with_capacity(args.iterations);
    for _ in 0..args.iterations {
        let start = Instant::now();
        let result = run_decompress_like(&fsst_arr, &pattern_arr, len)?;
        decompress_times.push(start.elapsed());
        let decompress_matches = count_true_bits(&result)?;
        debug_assert_eq!(decompress_matches, raw_matches);
    }

    let raw_median = median_ms(&mut raw_times);
    let fsst_median = median_ms(&mut fsst_times);
    let decompress_median = median_ms(&mut decompress_times);
    let speedup = if fsst_median > 0.0 {
        raw_median / fsst_median
    } else {
        0.0
    };
    let decompress_speedup = if decompress_median > 0.0 {
        raw_median / decompress_median
    } else {
        0.0
    };

    anyhow::ensure!(
        raw_matches == fsst_matches,
        "Match count mismatch for pattern '{pattern}': raw={raw_matches}, fsst={fsst_matches}"
    );

    Ok(BenchResult {
        pattern: truncate_pattern(pattern),
        query_type: query.query_type.clone(),
        selectivity: query.selectivity.clone(),
        fsst_difficulty: query.fsst_difficulty.clone(),
        match_fraction: raw_matches as f64 / len as f64,
        raw_matches,
        fsst_matches,
        raw_scan_ms: raw_median,
        fsst_scan_ms: fsst_median,
        decompress_like_ms: decompress_median,
        speedup,
        decompress_speedup,
    })
}

fn bench_regex(
    fsst_array: &FSSTArray,
    strings: &[String],
    pattern: &str,
    query: &MinedQuery,
    args: &RunArgs,
) -> Result<BenchResult> {
    let n = strings.len();
    let re = regex::Regex::new(pattern)?;

    // Try to lower the regex to a LIKE pattern for FSST
    let like_pattern = try_lower_regex_to_like(pattern);

    // Warmup + time raw regex
    for _ in 0..args.warmup {
        let _: usize = strings.iter().filter(|s| re.is_match(s)).count();
    }
    let mut raw_times = Vec::with_capacity(args.iterations);
    let mut raw_matches = 0;
    for _ in 0..args.iterations {
        let start = Instant::now();
        raw_matches = strings.iter().filter(|s| re.is_match(s)).count();
        raw_times.push(start.elapsed());
    }

    // FSST path: use LIKE if we can lower, otherwise fall back to raw regex on strings
    let mut fsst_times = Vec::with_capacity(args.iterations);
    let mut fsst_matches = 0;

    if let Some(like_pat) = &like_pattern {
        let fsst_arr = fsst_array.clone().into_array();
        let pattern_arr = ConstantArray::new(like_pat.as_str(), n).into_array();

        for _ in 0..args.warmup {
            run_like(&fsst_arr, &pattern_arr, n)?;
        }
        for _ in 0..args.iterations {
            let start = Instant::now();
            let result = run_like(&fsst_arr, &pattern_arr, n)?;
            fsst_times.push(start.elapsed());
            fsst_matches = count_true_bits(&result)?;
        }
    } else {
        // Fallback: run regex on raw strings (same as raw path)
        for _ in 0..args.warmup {
            let _: usize = strings.iter().filter(|s| re.is_match(s)).count();
        }
        for _ in 0..args.iterations {
            let start = Instant::now();
            fsst_matches = strings.iter().filter(|s| re.is_match(s)).count();
            fsst_times.push(start.elapsed());
        }
    }

    // Decompress+like path for regex (only if we can lower to LIKE)
    let mut decompress_times = Vec::with_capacity(args.iterations);
    if let Some(like_pat) = &like_pattern {
        let fsst_arr = fsst_array.clone().into_array();
        let pattern_arr = ConstantArray::new(like_pat.as_str(), n).into_array();

        for _ in 0..args.warmup {
            run_decompress_like(&fsst_arr, &pattern_arr, n)?;
        }
        for _ in 0..args.iterations {
            let start = Instant::now();
            run_decompress_like(&fsst_arr, &pattern_arr, n)?;
            decompress_times.push(start.elapsed());
        }
    } else {
        // No LIKE lowering possible, same as raw
        decompress_times = raw_times.clone();
    }

    let raw_median = median_ms(&mut raw_times);
    let fsst_median = median_ms(&mut fsst_times);
    let decompress_median = median_ms(&mut decompress_times);
    let speedup = if fsst_median > 0.0 {
        raw_median / fsst_median
    } else {
        0.0
    };
    let decompress_speedup = if decompress_median > 0.0 {
        raw_median / decompress_median
    } else {
        0.0
    };

    Ok(BenchResult {
        pattern: truncate_pattern(pattern),
        query_type: query.query_type.clone(),
        selectivity: query.selectivity.clone(),
        fsst_difficulty: query.fsst_difficulty.clone(),
        match_fraction: raw_matches as f64 / n as f64,
        raw_matches,
        fsst_matches,
        raw_scan_ms: raw_median,
        fsst_scan_ms: fsst_median,
        decompress_like_ms: decompress_median,
        speedup,
        decompress_speedup,
    })
}

/// Try to convert a simple regex to a SQL LIKE pattern.
fn try_lower_regex_to_like(pattern: &str) -> Option<String> {
    if pattern.starts_with('^') && is_literal_regex(&pattern[1..]) {
        return Some(format!("{}%", &pattern[1..]));
    }
    if is_literal_regex(pattern) {
        return Some(format!("%{pattern}%"));
    }
    None
}

fn is_literal_regex(s: &str) -> bool {
    !s.contains(|c: char| {
        matches!(
            c,
            '.' | '*' | '+' | '?' | '[' | ']' | '(' | ')' | '{' | '}' | '|' | '\\' | '^' | '$'
        )
    })
}

fn run_like(array: &ArrayRef, pattern: &ArrayRef, len: usize) -> Result<ArrayRef> {
    let mut ctx = SESSION.create_execution_ctx();
    let result = Like
        .try_new_array(
            len,
            LikeOptions::default(),
            [array.clone(), pattern.clone()],
        )?
        .into_array()
        .execute::<Canonical>(&mut ctx)?;
    Ok(result.into_array())
}

fn run_decompress_like(fsst_arr: &ArrayRef, pattern: &ArrayRef, len: usize) -> Result<ArrayRef> {
    let mut ctx = SESSION.create_execution_ctx();
    // Decompress FSST to canonical (VarBinView)
    let decompressed = fsst_arr
        .clone()
        .execute::<Canonical>(&mut ctx)?
        .into_array();
    // Run LIKE on the decompressed array
    let result = Like
        .try_new_array(len, LikeOptions::default(), [decompressed, pattern.clone()])?
        .into_array()
        .execute::<Canonical>(&mut ctx)?;
    Ok(result.into_array())
}

fn count_true_bits(array: &ArrayRef) -> Result<usize> {
    let mut ctx = SESSION.create_execution_ctx();
    let canonical = array.clone().execute::<Canonical>(&mut ctx)?;
    match canonical {
        Canonical::Bool(b) => Ok(b.to_bit_buffer().true_count()),
        _ => anyhow::bail!("Expected BoolArray from LIKE"),
    }
}

fn median_ms(times: &mut [Duration]) -> f64 {
    times.sort();
    if times.is_empty() {
        return 0.0;
    }
    let mid = times.len() / 2;
    times[mid].as_secs_f64() * 1000.0
}

fn truncate_pattern(s: &str) -> String {
    if s.len() > 40 {
        format!("{}...", &s[..37])
    } else {
        s.to_string()
    }
}

fn load_queries(dataset: &DatasetName) -> Result<Vec<MinedQuery>> {
    let path = query_miner::queries_path(dataset);
    if !path.exists() {
        anyhow::bail!("Queries not mined yet. Run `mine {dataset}` first.",);
    }

    let content = std::fs::read_to_string(&path)?;
    let queries: Vec<MinedQuery> = serde_json::from_str(&content)?;
    Ok(queries)
}

fn results_path(dataset: &DatasetName) -> PathBuf {
    crate::data_prep::output_dir(dataset).join(format!("{}_results.json", dataset.file_stem()))
}
