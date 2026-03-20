//! Generic FSST LIKE + regex performance analysis on real data.
//!
//! Reads strings from a parquet file or stdin (one string per line), compresses
//! with FSST, then benchmarks Arrow LIKE vs FSST DFA LIKE and byte-level regex
//! vs FSST fused regex. Reports compression statistics and symbol table details.
//!
//! Runs are statistically significant: configurable warmup + sample iterations
//! with mean/stddev/min/max reported.
//!
//! ## Usage
//!
//! From a parquet file:
//!   cargo run --example fsst_like_analyze -p vortex-fsst --release -- \
//!       --parquet data/clickbench.parquet --column URL \
//!       --like 'https%' '%yandex%' \
//!       --regex 'https?://' 'yandex|google'
//!
//! From stdin (one string per line):
//!   cat strings.txt | cargo run --example fsst_like_analyze -p vortex-fsst --release -- \
//!       --like 'prefix%' '%needle%' --regex 'pat[0-9]+'
//!
//! Options:
//!   --parquet <path>    Read strings from parquet file
//!   --column <name>     Column name (required with --parquet)
//!   --like <patterns>   LIKE patterns to benchmark (prefix% or %needle%)
//!   --regex <patterns>  Regex patterns to benchmark
//!   --max-rows <N>      Max strings to read (default: 100000)
//!   --iters <N>         Benchmark iterations per sample (default: 10)
//!   --samples <N>       Number of samples for statistics (default: 5)

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::needless_range_loop,
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::min_ident_chars,
    clippy::many_single_char_names,
    clippy::too_many_arguments,
    clippy::disallowed_types,
    clippy::exit,
    clippy::use_debug,
    clippy::redundant_clone,
    dead_code
)]

use std::collections::HashSet;
use std::collections::VecDeque;
use std::env;
use std::fs;
use std::io;
use std::io::BufRead;
use std::io::Write;
use std::time::Instant;

use regex_automata::Anchored;
use regex_automata::dfa::Automaton;
use regex_automata::dfa::dense::DFA;
use regex_automata::util::primitives::StateID;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::scalar_fn::ScalarFnArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::scalar_fn::fns::like::Like;
use vortex_array::scalar_fn::fns::like::LikeOptions;
use vortex_array::session::ArraySession;
use vortex_fsst::FSSTArray;
use vortex_fsst::fsst_compress;
use vortex_fsst::fsst_train_compressor;
use vortex_session::VortexSession;

const ESCAPE_CODE: u8 = 255;

// ---------------------------------------------------------------------------
// CLI argument parsing
// ---------------------------------------------------------------------------

struct Args {
    parquet_path: Option<String>,
    column: Option<String>,
    like_patterns: Vec<String>,
    regex_patterns: Vec<String>,
    max_rows: usize,
    iters: usize,
    samples: usize,
}

fn parse_args() -> Args {
    let args: Vec<String> = env::args().collect();

    let mut parquet_path = None;
    let mut column = None;
    let mut like_patterns = Vec::new();
    let mut regex_patterns = Vec::new();
    let mut max_rows = 100_000usize;
    let mut iters = 10usize;
    let mut samples = 5usize;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--parquet" => {
                i += 1;
                parquet_path = Some(args[i].clone());
                i += 1;
            }
            "--column" => {
                i += 1;
                column = Some(args[i].clone());
                i += 1;
            }
            "--like" => {
                i += 1;
                while i < args.len() && !args[i].starts_with("--") {
                    like_patterns.push(args[i].clone());
                    i += 1;
                }
            }
            "--regex" => {
                i += 1;
                while i < args.len() && !args[i].starts_with("--") {
                    regex_patterns.push(args[i].clone());
                    i += 1;
                }
            }
            "--max-rows" => {
                i += 1;
                max_rows = args[i].parse().expect("--max-rows must be a number");
                i += 1;
            }
            "--iters" => {
                i += 1;
                iters = args[i].parse().expect("--iters must be a number");
                i += 1;
            }
            "--samples" => {
                i += 1;
                samples = args[i].parse().expect("--samples must be a number");
                i += 1;
            }
            "--help" | "-h" => {
                eprintln!("Usage: fsst_like_analyze [OPTIONS]");
                eprintln!();
                eprintln!("  --parquet <path>    Read from parquet file");
                eprintln!("  --column <name>     Column name (required with --parquet)");
                eprintln!("  --like <patterns>   LIKE patterns (prefix% or %needle%)");
                eprintln!("  --regex <patterns>  Regex patterns");
                eprintln!("  --max-rows <N>      Max rows (default: 100000)");
                eprintln!("  --iters <N>         Iterations per sample (default: 10)");
                eprintln!("  --samples <N>       Samples for statistics (default: 5)");
                eprintln!();
                eprintln!("Without --parquet, reads strings from stdin (one per line).");
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {other}. Use --help for usage.");
                std::process::exit(1);
            }
        }
    }

    if like_patterns.is_empty() && regex_patterns.is_empty() {
        eprintln!("No patterns specified. Use --like and/or --regex. Try --help.");
        std::process::exit(1);
    }

    Args {
        parquet_path,
        column,
        like_patterns,
        regex_patterns,
        max_rows,
        iters,
        samples,
    }
}

// ---------------------------------------------------------------------------
// Data loading
// ---------------------------------------------------------------------------

fn read_parquet_strings(path: &str, column: &str, max_rows: usize) -> Vec<String> {
    use arrow_array::Array as _;
    use arrow_array::StringArray;
    use arrow_array::cast::AsArray;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    let data = fs::read(path).unwrap_or_else(|e| {
        eprintln!("Failed to read {path}: {e}");
        std::process::exit(1);
    });

    let reader = ParquetRecordBatchReaderBuilder::try_new(bytes::Bytes::from(data))
        .unwrap()
        .with_batch_size(8192)
        .build()
        .unwrap();

    let mut strings = Vec::new();
    for batch in reader {
        let batch = batch.unwrap();
        let col_idx = batch
            .schema()
            .fields()
            .iter()
            .position(|f| f.name() == column)
            .unwrap_or_else(|| {
                let cols: Vec<_> = batch
                    .schema()
                    .fields()
                    .iter()
                    .map(|f| f.name().clone())
                    .collect();
                eprintln!("Column '{column}' not found. Available: {cols:?}");
                std::process::exit(1);
            });

        let col = batch.column(col_idx);
        let str_array: &StringArray = col.as_string();
        for i in 0..str_array.len() {
            if strings.len() >= max_rows {
                break;
            }
            if str_array.is_valid(i) {
                let s = str_array.value(i);
                if !s.is_empty() {
                    strings.push(s.to_string());
                }
            }
        }
        if strings.len() >= max_rows {
            break;
        }
    }
    strings
}

fn read_stdin_strings(max_rows: usize) -> Vec<String> {
    let stdin = io::stdin();
    let mut strings = Vec::new();
    for line in stdin.lock().lines() {
        if strings.len() >= max_rows {
            break;
        }
        let line = line.expect("failed to read stdin line");
        if !line.is_empty() {
            strings.push(line);
        }
    }
    strings
}

// ---------------------------------------------------------------------------
// Statistics helper
// ---------------------------------------------------------------------------

struct Stats {
    mean: f64,
    stddev: f64,
    min: f64,
    max: f64,
}

impl Stats {
    fn from_samples(samples: &[f64]) -> Self {
        let n = samples.len() as f64;
        let mean = samples.iter().sum::<f64>() / n;
        let variance =
            samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
        let stddev = variance.sqrt();
        let min = samples.iter().copied().fold(f64::INFINITY, f64::min);
        let max = samples.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Self {
            mean,
            stddev,
            min,
            max,
        }
    }
}

fn format_duration(us: f64) -> String {
    if us >= 1000.0 {
        format!("{:.2} ms", us / 1000.0)
    } else {
        format!("{:.1} µs", us)
    }
}

// ---------------------------------------------------------------------------
// Compression statistics
// ---------------------------------------------------------------------------

fn print_compression_stats(fsst: &FSSTArray, raw_bytes: usize) {
    let n_strings = fsst.len();
    let symbols = fsst.symbols();
    let sym_lengths = fsst.symbol_lengths();
    let n_symbols = symbols.len();

    let mut len_dist = [0u32; 9];
    for i in 0..n_symbols {
        len_dist[sym_lengths.as_slice()[i] as usize] += 1;
    }

    let mean_sym_len = if n_symbols > 0 {
        sym_lengths
            .as_slice()
            .iter()
            .map(|&l| l as f64)
            .sum::<f64>()
            / n_symbols as f64
    } else {
        0.0
    };

    let codes = fsst.codes();
    let bytes_buf = codes.bytes();
    let all_bytes = bytes_buf.as_ref();

    let mut total_codes: u64 = 0;
    let mut escape_count: u64 = 0;
    let mut compressed_bytes: u64 = 0;

    for i in 0..codes.len() {
        let start = codes.offset_at(i);
        let end = codes.offset_at(i + 1);
        let slice = &all_bytes[start..end];
        compressed_bytes += slice.len() as u64;
        let mut j = 0;
        while j < slice.len() {
            total_codes += 1;
            if slice[j] == ESCAPE_CODE {
                escape_count += 1;
                j += 1;
            }
            j += 1;
        }
    }

    let escape_rate = escape_count as f64 / total_codes.max(1) as f64 * 100.0;
    let compression_ratio = raw_bytes as f64 / compressed_bytes.max(1) as f64;
    let codes_per_string = total_codes as f64 / n_strings.max(1) as f64;
    let avg_len = raw_bytes as f64 / n_strings.max(1) as f64;

    // Compute entropy
    let mut code_freq = vec![0u64; 256];
    for i in 0..codes.len() {
        let start = codes.offset_at(i);
        let end = codes.offset_at(i + 1);
        for &b in &all_bytes[start..end] {
            code_freq[b as usize] += 1;
        }
    }
    let total_bytes_f = compressed_bytes as f64;
    let entropy: f64 = code_freq
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / total_bytes_f;
            -p * p.log2()
        })
        .sum();

    println!("## Symbol Table & Compression");
    println!();
    println!("| Metric | Value |");
    println!("|--------|-------|");
    println!("| Strings | {} |", n_strings);
    println!("| Avg string length | {:.1} bytes |", avg_len);
    println!(
        "| Raw / Compressed | {:.1} MB / {:.1} MB |",
        raw_bytes as f64 / 1e6,
        compressed_bytes as f64 / 1e6
    );
    println!("| **Compression ratio** | **{:.2}x** |", compression_ratio);
    println!("| Symbols in table | {} |", n_symbols);
    println!("| **Mean symbol length** | **{:.2} bytes** |", mean_sym_len);
    println!("| **Escape rate** | **{:.2}%** |", escape_rate);
    println!("| Avg codes/string | {:.1} |", codes_per_string);
    println!("| Code entropy | {:.2} bits |", entropy);
    println!();

    println!("### Symbol Length Distribution");
    println!();
    println!("| Length | Count | % |");
    println!("|--------|-------|---|");
    for l in 1..=8 {
        if len_dist[l] > 0 {
            println!(
                "| {l}-byte | {} | {:.1}% |",
                len_dist[l],
                len_dist[l] as f64 / n_symbols.max(1) as f64 * 100.0
            );
        }
    }

    // Top 10 symbols
    let mut freq_with_idx: Vec<(usize, u64)> = code_freq
        .iter()
        .enumerate()
        .filter(|&(idx, &c)| c > 0 && idx != ESCAPE_CODE as usize && idx < n_symbols)
        .map(|(idx, &c)| (idx, c))
        .collect();
    freq_with_idx.sort_by(|a, b| b.1.cmp(&a.1));

    println!();
    println!("### Top 10 Symbols");
    println!();
    println!("| Rank | Length | Freq | % | Symbol |");
    println!("|------|--------|------|---|--------|");
    for (rank, &(code, count)) in freq_with_idx.iter().take(10).enumerate() {
        let sym = symbols.as_slice()[code];
        let sym_len = sym_lengths.as_slice()[code] as usize;
        let sym_bytes = &sym.to_u64().to_le_bytes()[..sym_len];
        let sym_str = String::from_utf8_lossy(sym_bytes);
        let pct = count as f64 / total_bytes_f * 100.0;
        println!(
            "| {} | {} | {} | {:.2}% | {:?} |",
            rank + 1,
            sym_len,
            count,
            pct,
            sym_str
        );
    }
    println!();
}

// ---------------------------------------------------------------------------
// LIKE benchmark: Arrow vs FSST DFA
// ---------------------------------------------------------------------------

fn bench_like(
    arrow_arr: &ArrayRef,
    fsst_arr: &ArrayRef,
    patterns: &[String],
    session: &VortexSession,
    n_iters: usize,
    n_samples: usize,
) {
    println!("## LIKE: Arrow String vs FSST DFA");
    println!();
    println!(
        "({n_samples} samples × {n_iters} iterations per sample, {} strings)",
        arrow_arr.len()
    );
    println!();
    println!(
        "| Pattern | Arrow (mean ± σ) | FSST DFA (mean ± σ) | Speedup | Arrow min | FSST min |"
    );
    println!(
        "|---------|-----------------|--------------------:|--------:|----------:|---------:|"
    );

    let len = arrow_arr.len();
    let opts = LikeOptions::default();

    let mut csv_rows: Vec<String> = Vec::new();

    for pattern in patterns {
        let pat = ConstantArray::new(pattern.as_str(), len).into_array();

        // Warmup
        for arr in [arrow_arr, fsst_arr] {
            for _ in 0..2 {
                std::hint::black_box(
                    Like.try_new_array(len, opts, [arr.clone(), pat.clone()])
                        .unwrap()
                        .into_array()
                        .execute::<Canonical>(&mut session.create_execution_ctx())
                        .unwrap(),
                );
            }
        }

        // Arrow samples
        let mut arrow_samples = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            let t = Instant::now();
            for _ in 0..n_iters {
                std::hint::black_box(
                    Like.try_new_array(len, opts, [arrow_arr.clone(), pat.clone()])
                        .unwrap()
                        .into_array()
                        .execute::<Canonical>(&mut session.create_execution_ctx())
                        .unwrap(),
                );
            }
            arrow_samples.push(t.elapsed().as_micros() as f64 / n_iters as f64);
        }

        // FSST samples
        let mut fsst_samples = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            let t = Instant::now();
            for _ in 0..n_iters {
                std::hint::black_box(
                    Like.try_new_array(len, opts, [fsst_arr.clone(), pat.clone()])
                        .unwrap()
                        .into_array()
                        .execute::<Canonical>(&mut session.create_execution_ctx())
                        .unwrap(),
                );
            }
            fsst_samples.push(t.elapsed().as_micros() as f64 / n_iters as f64);
        }

        let arrow_stats = Stats::from_samples(&arrow_samples);
        let fsst_stats = Stats::from_samples(&fsst_samples);
        let speedup = arrow_stats.mean / fsst_stats.mean;

        println!(
            "| `{}` | {} ± {} | {} ± {} | {:.2}x | {} | {} |",
            pattern,
            format_duration(arrow_stats.mean),
            format_duration(arrow_stats.stddev),
            format_duration(fsst_stats.mean),
            format_duration(fsst_stats.stddev),
            speedup,
            format_duration(arrow_stats.min),
            format_duration(fsst_stats.min),
        );

        csv_rows.push(format!(
            "{},{:.2},{:.2},{:.2},{:.2},{:.3}",
            pattern,
            arrow_stats.mean,
            arrow_stats.stddev,
            fsst_stats.mean,
            fsst_stats.stddev,
            speedup,
        ));
    }
    println!();

    let csv_path = "encodings/fsst/data/like_analyze.csv";
    fs::create_dir_all("encodings/fsst/data").ok();
    if let Ok(mut f) = fs::File::create(csv_path) {
        writeln!(
            f,
            "pattern,arrow_mean_us,arrow_stddev_us,fsst_mean_us,fsst_stddev_us,speedup"
        )
        .unwrap();
        for row in &csv_rows {
            writeln!(f, "{row}").unwrap();
        }
        eprintln!("Wrote {csv_path}");
    }
}

// ---------------------------------------------------------------------------
// Fused regex DFA over FSST codes
// ---------------------------------------------------------------------------

struct FusedRegexDfa {
    sym_transitions: Vec<u32>,
    sym_hit_match: Vec<bool>,
    escape_transitions: Vec<u32>,
    eoi_transitions: Vec<u32>,
    is_match: Vec<bool>,
    dead_idx: u32,
    start_idx: u32,
    sentinel: u32,
    n_states: u32,
}

impl FusedRegexDfa {
    fn new(
        pattern: &str,
        symbols: &[fsst::Symbol],
        symbol_lengths: &[u8],
        n_symbols: usize,
    ) -> Result<Self, String> {
        let dfa = DFA::new(pattern).map_err(|e| format!("regex compile: {e}"))?;
        let start = dfa
            .universal_start_state(Anchored::No)
            .ok_or("no universal unanchored start state")?;

        // BFS reachable states
        let mut seen = HashSet::new();
        let mut queue = VecDeque::new();
        seen.insert(start);
        queue.push_back(start);
        while let Some(state) = queue.pop_front() {
            for byte in 0..=255u8 {
                let next = dfa.next_state(state, byte);
                if seen.insert(next) {
                    queue.push_back(next);
                }
            }
            let eoi = dfa.next_eoi_state(state);
            if seen.insert(eoi) {
                queue.push_back(eoi);
            }
        }

        let mut states: Vec<StateID> = seen.into_iter().collect();
        states.sort();
        let n_states = states.len() as u32;
        let sentinel = n_states;

        let state_to_idx: std::collections::HashMap<StateID, u32> = states
            .iter()
            .enumerate()
            .map(|(i, &s)| (s, i as u32))
            .collect();

        let dead_idx = states
            .iter()
            .position(|&s| dfa.is_dead_state(s))
            .map(|i| i as u32)
            .unwrap_or(0);
        let start_idx = state_to_idx[&start];

        let is_match: Vec<bool> = states.iter().map(|&s| dfa.is_match_state(s)).collect();

        let table_size = n_states as usize * 256;
        let mut sym_transitions = vec![dead_idx; table_size];
        let mut sym_hit_match = vec![false; table_size];
        let mut escape_transitions = vec![dead_idx; table_size];
        let mut eoi_transitions = vec![dead_idx; n_states as usize];

        for (state_idx, &state_id) in states.iter().enumerate() {
            let base = state_idx * 256;

            for code in 0..n_symbols {
                let sym = symbols[code].to_u64().to_le_bytes();
                let sym_len = symbol_lengths[code] as usize;
                let mut s = state_id;
                let mut hit = false;
                for &b in &sym[..sym_len] {
                    s = dfa.next_state(s, b);
                    if dfa.is_match_state(s) {
                        hit = true;
                    }
                }
                sym_transitions[base + code] = state_to_idx[&s];
                sym_hit_match[base + code] = hit;
            }
            sym_transitions[base + ESCAPE_CODE as usize] = sentinel;

            for byte in 0..=255u8 {
                let next = dfa.next_state(state_id, byte);
                escape_transitions[base + byte as usize] = state_to_idx[&next];
            }

            let eoi = dfa.next_eoi_state(state_id);
            eoi_transitions[state_idx] = *state_to_idx.get(&eoi).unwrap_or(&dead_idx);
        }

        Ok(Self {
            sym_transitions,
            sym_hit_match,
            escape_transitions,
            eoi_transitions,
            is_match,
            dead_idx,
            start_idx,
            sentinel,
            n_states,
        })
    }

    fn matches(&self, codes: &[u8]) -> bool {
        let mut idx = self.start_idx;
        let mut pos = 0;
        while pos < codes.len() {
            let code = codes[pos];
            pos += 1;
            let next = self.sym_transitions[idx as usize * 256 + code as usize];
            if next == self.sentinel {
                if pos >= codes.len() {
                    return false;
                }
                let b = codes[pos];
                pos += 1;
                idx = self.escape_transitions[idx as usize * 256 + b as usize];
            } else {
                if self.sym_hit_match[idx as usize * 256 + code as usize] {
                    return true;
                }
                idx = next;
            }
            if self.is_match[idx as usize] {
                return true;
            }
            if idx == self.dead_idx {
                return false;
            }
        }
        let eoi = self.eoi_transitions[idx as usize];
        self.is_match[eoi as usize]
    }
}

#[inline(never)]
fn byte_dfa_matches(dfa: &DFA<Vec<u32>>, start: StateID, haystack: &[u8]) -> bool {
    let mut state = start;
    for &byte in haystack {
        state = dfa.next_state(state, byte);
        if dfa.is_match_state(state) {
            return true;
        }
        if dfa.is_dead_state(state) {
            return false;
        }
    }
    let eoi = dfa.next_eoi_state(state);
    dfa.is_match_state(eoi)
}

// ---------------------------------------------------------------------------
// Regex benchmark
// ---------------------------------------------------------------------------

fn bench_regex(
    strings: &[String],
    compressor: &fsst::Compressor,
    patterns: &[String],
    n_iters: usize,
    n_samples: usize,
) {
    let bytes_vec: Vec<&[u8]> = strings.iter().map(|s| s.as_bytes()).collect();
    let compressed: Vec<Vec<u8>> = bytes_vec.iter().map(|b| compressor.compress(b)).collect();
    let symbols = compressor.symbol_table();
    let symbol_lengths = compressor.symbol_lengths();
    let n_symbols = symbols.len();

    println!("## Regex: Byte DFA vs FSST Fused DFA");
    println!();
    println!(
        "({n_samples} samples × {n_iters} iterations, {} strings)",
        strings.len()
    );
    println!();
    println!(
        "| Pattern | Byte DFA (mean ± σ) | Fused DFA (mean ± σ) | Speedup | Matches | States |"
    );
    println!(
        "|---------|--------------------:|---------------------:|--------:|--------:|-------:|"
    );

    let mut csv_rows: Vec<String> = Vec::new();

    for pattern in patterns {
        let fused = match FusedRegexDfa::new(
            pattern,
            &symbols[..n_symbols],
            &symbol_lengths[..n_symbols],
            n_symbols,
        ) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("  Skipping regex \"{pattern}\": {e}");
                continue;
            }
        };

        let byte_dfa = match DFA::new(pattern.as_str()) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("  Skipping byte DFA \"{pattern}\": {e}");
                continue;
            }
        };
        let byte_start = match byte_dfa.universal_start_state(Anchored::No) {
            Some(s) => s,
            None => {
                eprintln!("  Skipping \"{pattern}\": no universal start state");
                continue;
            }
        };

        // Warmup
        for codes in compressed.iter().take(1000) {
            std::hint::black_box(fused.matches(codes));
        }
        for s in strings.iter().take(1000) {
            std::hint::black_box(byte_dfa_matches(&byte_dfa, byte_start, s.as_bytes()));
        }

        // Fused samples
        let mut fused_samples = Vec::with_capacity(n_samples);
        let mut n_matches = 0;
        for _ in 0..n_samples {
            n_matches = 0;
            let t = Instant::now();
            for _ in 0..n_iters {
                for codes in &compressed {
                    if fused.matches(codes) {
                        n_matches += 1;
                    }
                }
            }
            let ns_per = t.elapsed().as_nanos() as f64 / (n_iters as f64 * compressed.len() as f64);
            fused_samples.push(ns_per);
        }
        let n_fused = n_matches / n_iters;

        // Byte samples
        let mut byte_samples = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            let mut nm = 0;
            let t = Instant::now();
            for _ in 0..n_iters {
                for s in strings {
                    if byte_dfa_matches(&byte_dfa, byte_start, s.as_bytes()) {
                        nm += 1;
                    }
                }
            }
            let _ = nm;
            let ns_per = t.elapsed().as_nanos() as f64 / (n_iters as f64 * strings.len() as f64);
            byte_samples.push(ns_per);
        }

        let fused_stats = Stats::from_samples(&fused_samples);
        let byte_stats = Stats::from_samples(&byte_samples);
        let speedup = byte_stats.mean / fused_stats.mean;

        println!(
            "| `{}` | {:.1} ± {:.1} ns | {:.1} ± {:.1} ns | {:.2}x | {} | {} |",
            pattern,
            byte_stats.mean,
            byte_stats.stddev,
            fused_stats.mean,
            fused_stats.stddev,
            speedup,
            n_fused,
            fused.n_states,
        );

        csv_rows.push(format!(
            "{},{:.2},{:.2},{:.2},{:.2},{:.3},{},{}",
            pattern,
            byte_stats.mean,
            byte_stats.stddev,
            fused_stats.mean,
            fused_stats.stddev,
            speedup,
            n_fused,
            fused.n_states,
        ));
    }
    println!();

    let csv_path = "encodings/fsst/data/regex_analyze.csv";
    if let Ok(mut f) = fs::File::create(csv_path) {
        writeln!(
            f,
            "pattern,byte_mean_ns,byte_stddev_ns,fused_mean_ns,fused_stddev_ns,speedup,matches,dfa_states"
        )
        .unwrap();
        for row in &csv_rows {
            writeln!(f, "{row}").unwrap();
        }
        eprintln!("Wrote {csv_path}");
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let args = parse_args();

    // Load strings
    let strings = if let Some(ref path) = args.parquet_path {
        let column = args.column.as_deref().unwrap_or_else(|| {
            eprintln!("--column is required with --parquet");
            std::process::exit(1);
        });
        eprintln!(
            "Reading {path} column '{column}' (max {} rows)...",
            args.max_rows
        );
        read_parquet_strings(path, column, args.max_rows)
    } else {
        eprintln!("Reading strings from stdin (max {} rows)...", args.max_rows);
        read_stdin_strings(args.max_rows)
    };

    eprintln!("  Loaded {} strings", strings.len());

    if strings.is_empty() {
        eprintln!("No strings found. Exiting.");
        return;
    }

    // Compress with FSST
    let raw_bytes: usize = strings.iter().map(|s| s.len()).sum();
    let varbin = VarBinArray::from_iter(
        strings.iter().map(|s| Some(s.as_str())),
        DType::Utf8(Nullability::NonNullable),
    );
    let compressor = fsst_train_compressor(&varbin);
    let fsst = fsst_compress(varbin.clone(), &compressor);

    let source = args.parquet_path.as_deref().unwrap_or("stdin");
    let col = args.column.as_deref().unwrap_or("");

    println!("# FSST Analysis: {source} {col}");
    println!();

    // Part 1: Compression stats
    print_compression_stats(&fsst, raw_bytes);

    let session = VortexSession::empty().with::<ArraySession>();

    // Part 2: LIKE benchmark
    if !args.like_patterns.is_empty() {
        let arrow_arr = varbin.into_array();
        let fsst_arr = fsst.clone().into_array();
        bench_like(
            &arrow_arr,
            &fsst_arr,
            &args.like_patterns,
            &session,
            args.iters,
            args.samples,
        );
    }

    // Part 3: Regex benchmark
    if !args.regex_patterns.is_empty() {
        bench_regex(
            &strings,
            &compressor,
            &args.regex_patterns,
            args.iters,
            args.samples,
        );
    }
}
