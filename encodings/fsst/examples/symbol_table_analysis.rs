// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Symbol table distribution analysis for the workshop paper.
//!
//! Prints per-dataset statistics about FSST symbol tables:
//! - Number of symbols and symbol length distribution (1-8 bytes)
//! - Escape byte frequency in compressed output
//! - Compression ratio
//! - Code frequency distribution and entropy
//! - Continuous measures: mean symbol length, effective alphabet size

#![allow(clippy::unwrap_used)]

use fsst::ESCAPE_CODE;
use vortex_array::arrays::VarBinArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_fsst::FSSTArray;
use vortex_fsst::fsst_compress;
use vortex_fsst::fsst_train_compressor;
use vortex_fsst::test_utils::*;

const N: usize = 100_000;

struct DatasetInfo {
    name: &'static str,
    fsst: FSSTArray,
    raw_bytes: usize,
}

fn make_dataset(name: &'static str, strings: Vec<String>) -> DatasetInfo {
    let raw_bytes: usize = strings.iter().map(|s| s.len()).sum();
    let varbin = VarBinArray::from_iter(
        strings.iter().map(|s| Some(s.as_str())),
        DType::Utf8(Nullability::NonNullable),
    );
    let compressor = fsst_train_compressor(&varbin);
    let fsst = fsst_compress(varbin, &compressor);
    DatasetInfo {
        name,
        fsst,
        raw_bytes,
    }
}

/// Extract all code bytes from the FSSTArray's underlying VarBin codes.
fn all_code_slices(fsst: &FSSTArray) -> Vec<Vec<u8>> {
    let codes = fsst.codes();
    let bytes = codes.bytes();
    let n = codes.len();
    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        let start = codes.offset_at(i);
        let end = codes.offset_at(i + 1);
        result.push(bytes.as_ref()[start..end].to_vec());
    }
    result
}

struct AnalysisResult {
    name: String,
    n_symbols: usize,
    mean_sym_len: f64,
    compression_ratio: f64,
    escape_rate: f64,
    codes_per_string: f64,
    entropy: f64,
    effective_alphabet: f64,
    // For the continuous measures
    len_dist: [u32; 9],
    p50_syms: usize,
    p90_syms: usize,
    p99_syms: usize,
    unused_syms: usize,
}

fn analyze(info: &DatasetInfo) -> AnalysisResult {
    let fsst = &info.fsst;
    let symbols = fsst.symbols();
    let sym_lengths = fsst.symbol_lengths();
    let n_symbols = symbols.len();

    // Symbol length distribution
    let mut len_dist = [0u32; 9];
    for i in 0..n_symbols {
        let l = sym_lengths.as_slice()[i] as usize;
        len_dist[l] += 1;
    }

    let total_sym_bytes: usize = sym_lengths.as_slice().iter().map(|&l| l as usize).sum();
    let mean_sym_len = total_sym_bytes as f64 / n_symbols as f64;

    // Scan compressed codes
    let code_slices = all_code_slices(fsst);
    let mut total_codes: u64 = 0;
    let mut escape_count: u64 = 0;
    let mut code_freq = vec![0u64; 256];
    let mut compressed_bytes: u64 = 0;

    for slice in &code_slices {
        compressed_bytes += slice.len() as u64;
        let mut j = 0;
        while j < slice.len() {
            let code = slice[j];
            code_freq[code as usize] += 1;
            total_codes += 1;
            if code == ESCAPE_CODE {
                escape_count += 1;
                j += 1; // skip literal byte
            }
            j += 1;
        }
    }

    let escape_rate = escape_count as f64 / total_codes as f64 * 100.0;
    let compression_ratio = info.raw_bytes as f64 / compressed_bytes as f64;
    let codes_per_string = total_codes as f64 / fsst.len() as f64;

    // Shannon entropy of code distribution
    let entropy: f64 = code_freq
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / total_codes as f64;
            -p * p.log2()
        })
        .sum();

    // Effective alphabet size = 2^entropy
    let effective_alphabet = 2.0f64.powf(entropy);

    // Symbol usage: sort by frequency, compute coverage thresholds
    let mut freq_with_idx: Vec<(usize, u64)> = code_freq
        .iter()
        .enumerate()
        .filter(|&(idx, &c)| c > 0 && idx != ESCAPE_CODE as usize && (idx) < n_symbols)
        .map(|(idx, &c)| (idx, c))
        .collect();
    freq_with_idx.sort_by(|a, b| b.1.cmp(&a.1));

    let total_symbol_codes: u64 = freq_with_idx.iter().map(|(_, c)| c).sum();
    let mut cumulative = 0u64;
    let mut p50 = 0;
    let mut p90 = 0;
    let mut p99 = 0;
    for (i, &(_, count)) in freq_with_idx.iter().enumerate() {
        cumulative += count;
        let pct = cumulative as f64 / total_symbol_codes as f64;
        if p50 == 0 && pct >= 0.50 {
            p50 = i + 1;
        }
        if p90 == 0 && pct >= 0.90 {
            p90 = i + 1;
        }
        if p99 == 0 && pct >= 0.99 {
            p99 = i + 1;
        }
    }

    let used_symbols = freq_with_idx.len();
    let unused = n_symbols - used_symbols;

    AnalysisResult {
        name: info.name.to_string(),
        n_symbols,
        mean_sym_len,
        compression_ratio,
        escape_rate,
        codes_per_string,
        entropy,
        effective_alphabet,
        len_dist,
        p50_syms: p50,
        p90_syms: p90,
        p99_syms: p99,
        unused_syms: unused,
    }
}

fn print_detailed(info: &DatasetInfo, result: &AnalysisResult) {
    let fsst = &info.fsst;
    let symbols = fsst.symbols();
    let sym_lengths = fsst.symbol_lengths();

    println!("## {}", result.name);
    println!();
    println!("- **Strings**: {}", fsst.len());
    println!("- **Symbols in table**: {}", result.n_symbols);
    println!("- **Symbol length distribution**:");
    for l in 1..=8 {
        if result.len_dist[l] > 0 {
            println!(
                "  - {l}-byte: {} ({:.1}%)",
                result.len_dist[l],
                result.len_dist[l] as f64 / result.n_symbols as f64 * 100.0
            );
        }
    }
    println!("- **Mean symbol length**: {:.2} bytes", result.mean_sym_len);
    println!("- **Raw bytes**: {}", info.raw_bytes);
    println!("- **Compression ratio**: {:.2}x", result.compression_ratio);
    println!("- **Escape rate**: {:.2}%", result.escape_rate);
    println!("- **Avg codes per string**: {:.1}", result.codes_per_string);
    println!("- **Code entropy**: {:.2} bits", result.entropy);
    println!(
        "- **Effective alphabet size**: {:.1} (of {})",
        result.effective_alphabet, result.n_symbols
    );
    println!(
        "- **Symbol coverage**: 50% in {} syms, 90% in {}, 99% in {} (of {})",
        result.p50_syms, result.p90_syms, result.p99_syms, result.n_symbols
    );
    println!(
        "- **Unused symbols**: {} ({:.1}%)",
        result.unused_syms,
        result.unused_syms as f64 / result.n_symbols as f64 * 100.0
    );

    // Top 10 symbols
    let code_slices = all_code_slices(fsst);
    let mut code_freq = vec![0u64; 256];
    let mut total_codes: u64 = 0;
    for slice in &code_slices {
        let mut j = 0;
        while j < slice.len() {
            code_freq[slice[j] as usize] += 1;
            total_codes += 1;
            if slice[j] == ESCAPE_CODE {
                j += 1;
            }
            j += 1;
        }
    }

    let mut freq_with_idx: Vec<(usize, u64)> = code_freq
        .iter()
        .enumerate()
        .filter(|&(idx, &c)| c > 0 && idx != ESCAPE_CODE as usize && idx < result.n_symbols)
        .map(|(idx, &c)| (idx, c))
        .collect();
    freq_with_idx.sort_by(|a, b| b.1.cmp(&a.1));

    println!("- **Top 10 symbols by frequency**:");
    for (rank, &(code, count)) in freq_with_idx.iter().take(10).enumerate() {
        let sym = symbols.as_slice()[code];
        let sym_len = sym_lengths.as_slice()[code] as usize;
        let sym_bytes = &sym.to_u64().to_le_bytes()[..sym_len];
        let sym_str = String::from_utf8_lossy(sym_bytes);
        let pct = count as f64 / total_codes as f64 * 100.0;
        println!(
            "  {}. code={code:3}, len={sym_len}, freq={count:>8} ({pct:5.2}%), sym={:?}",
            rank + 1,
            sym_str
        );
    }

    println!();
}

fn main() {
    println!("# FSST Symbol Table Distribution Analysis");
    println!();
    println!("Dataset size: {N} strings each");
    println!();

    let datasets = vec![
        make_dataset("Short URLs", generate_short_urls(N)),
        make_dataset("ClickBench URLs", generate_clickbench_urls(N)),
        make_dataset("Log Lines", generate_log_lines(N)),
        make_dataset("JSON Strings", generate_json_strings(N)),
        make_dataset("File Paths", generate_file_paths(N)),
        make_dataset("Emails", generate_emails(N)),
        make_dataset("Rare Match (high entropy)", generate_rare_match_strings(N, 0.00001)),
    ];

    let results: Vec<AnalysisResult> = datasets.iter().map(|ds| analyze(ds)).collect();

    // Detailed per-dataset output
    for (ds, res) in datasets.iter().zip(results.iter()) {
        print_detailed(ds, res);
    }

    // Summary table
    println!("## Summary Table");
    println!();
    println!(
        "| Dataset | Syms | Mean Len | Ratio | Esc % | Codes/Str | Entropy | Eff. Alpha | p50 | p90 | p99 | Unused |"
    );
    println!(
        "|---------|------|----------|-------|-------|-----------|---------|------------|-----|-----|-----|--------|"
    );
    for r in &results {
        println!(
            "| {:<25} | {:>4} | {:>7.2} | {:>5.2}x | {:>5.2} | {:>9.1} | {:>7.2} | {:>10.1} | {:>3} | {:>3} | {:>3} | {:>6} |",
            r.name,
            r.n_symbols,
            r.mean_sym_len,
            r.compression_ratio,
            r.escape_rate,
            r.codes_per_string,
            r.entropy,
            r.effective_alphabet,
            r.p50_syms,
            r.p90_syms,
            r.p99_syms,
            r.unused_syms,
        );
    }

    // Key observations for the paper
    println!();
    println!("## Key Continuous Measures for Paper");
    println!();
    println!("These measures correlate with DFA scan and decompression throughput:");
    println!();
    println!("1. **Mean symbol length** (bytes/symbol): Higher → fewer codes per string →");
    println!("   fewer DFA transitions → faster DFA scan. Also faster decompression (fewer");
    println!("   table lookups, more bytes emitted per lookup).");
    println!();
    println!("2. **Escape rate** (%): Higher → more branch mispredictions in DFA scan");
    println!("   (sentinel check), more single-byte emissions in decompression. Key");
    println!("   continuous predictor of DFA overhead.");
    println!();
    println!("3. **Effective alphabet size** (2^entropy): Measures how many symbols are");
    println!("   'active'. Lower → more cache-friendly DFA transitions (fewer hot states).");
    println!("   Good continuous measure of symbol table 'quality'.");
    println!();
    println!("4. **Compression ratio**: Composite measure. Correlates with all of the above.");
    println!("   Higher ratio → fewer bytes to scan → faster wall-clock time regardless of");
    println!("   DFA vs decompress.");
    println!();
    println!("5. **Codes per string**: Direct measure of DFA work per string. Combined with");
    println!("   escape rate, predicts per-string scan latency.");
    println!();
    println!("### Suggested experiments:");
    println!();
    println!("- **Scatter: escape_rate vs DFA_throughput_GB/s** — expect negative linear relationship");
    println!("- **Scatter: mean_sym_len vs DFA_speedup** — expect positive correlation");
    println!("- **Scatter: compression_ratio vs decompression_throughput** — expect positive (less work)");
    println!("- **Sweep synthetic data**: vary unique_chars from 2 to 60 to sweep escape rate");
    println!("   from ~0% to ~50%, measure DFA + decompress throughput continuously");
}
