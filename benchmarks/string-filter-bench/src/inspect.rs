// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Inspect the FSST symbol table and enumerate DFA accepting paths for memmem.

use std::collections::BTreeSet;
use std::time::Instant;

use anyhow::Result;
use clap::Args;
use fsst::Symbol;
use vortex::array::accessor::ArrayAccessor;
use vortex::array::arrays::VarBinArray;
use vortex::array::dtype::DType;
use vortex::array::dtype::Nullability;
use vortex::encodings::fsst::FSSTArray;
use vortex::encodings::fsst::FSSTArrayExt;
use vortex::encodings::fsst::fsst_compress;
use vortex::encodings::fsst::fsst_train_compressor;

use crate::data_prep::DatasetName;
use crate::query_miner;

#[derive(Args)]
pub struct InspectArgs {
    /// Which dataset to inspect
    #[arg(value_enum)]
    pub dataset: DatasetName,

    /// Max strings to load
    #[arg(long, default_value_t = 100_000)]
    pub max_rows: usize,
}

pub fn run(args: InspectArgs) -> Result<()> {
    let strings = query_miner::load_strings(&args.dataset)?;
    let strings: Vec<&str> = strings
        .iter()
        .take(args.max_rows)
        .map(|s| s.as_str())
        .collect();
    let n = strings.len();
    println!("Loaded {n} strings for {}", args.dataset);

    let varbin = VarBinArray::from_iter_nonnull(
        strings.iter().map(|s| s.as_bytes()),
        DType::Utf8(Nullability::NonNullable),
    );
    let compressor = fsst_train_compressor(&varbin);
    let len = varbin.len();
    let dtype = varbin.dtype().clone();
    let fsst_array: FSSTArray = fsst_compress(varbin, len, &dtype, &compressor);

    let symbols = fsst_array.symbols();
    let symbol_lengths = fsst_array.symbol_lengths();
    let n_symbols = symbols.len();

    // Build expansion table: code -> raw bytes
    let expansions: Vec<Vec<u8>> = symbols
        .iter()
        .zip(symbol_lengths.iter())
        .map(|(sym, &slen)| symbol_bytes(sym, slen))
        .collect();

    let needles: Vec<&str> = match args.dataset {
        DatasetName::ClickbenchUrl
        | DatasetName::ClickbenchReferer
        | DatasetName::GharchiveActorAvatarUrl
        | DatasetName::FinewebUrl => {
            // Just "google" — ClickBench Q20 needle. Short fragments
            // (2–3 chars) explode the depth-8 DFS path enumeration on
            // 231-symbol tables, so we keep this list tight.
            vec!["google"]
        }
        DatasetName::ClickbenchTitle
        | DatasetName::ClickbenchSearchPhrase
        | DatasetName::ClickbenchParams => {
            vec![
                "utm", "http", "www", "page", "news", "search", "query", "text", "title", "id=",
            ]
        }
        DatasetName::JsonLines => {
            vec![
                "e\":", "dept\"", "\":4", "\"sa", "dy\"", "d\":", ":\"s", "{\"id",
            ]
        }
        DatasetName::GharchiveRepoName => {
            vec![
                "rust", "data", "test", "api", "lib", "bot", "docs", "node", "react", "vortex",
            ]
        }
        DatasetName::GharchiveActorLogin => {
            vec![
                "bot",
                "user",
                "dev",
                "github",
                "renovate",
                "dependabot",
                "rust",
                "john",
                "alex",
                "team",
            ]
        }
        DatasetName::GharchivePayloadRef => {
            vec![
                "refs/", "heads", "main", "master", "feature", "develop", "release", "fix", "pull",
                "tag",
            ]
        }
        DatasetName::PolarsignalsLabelsComm | DatasetName::PolarsignalsLabelsThreadName => {
            vec![
                "comm", "thread", "main", "worker", "pool", "http", "grpc", "async", "task",
                "sched",
            ]
        }
        DatasetName::PolarsignalsMappingFile
        | DatasetName::PolarsignalsFunctionName
        | DatasetName::PolarsignalsFunctionFilename => {
            vec![
                "src", "lib", ".go", ".rs", "func", "main", "http", "grpc", "runtime", "handler",
            ]
        }
        DatasetName::FinewebText => {
            vec![
                " th", "an ", " st", "eve", ". T", "re ", " on", "en ", "n a", "The",
            ]
        }
        DatasetName::TpchLineitem => {
            vec![
                "special", "request", "regular", "careful", "furious", "pending", "quick",
                "silent", "blith", "final",
            ]
        }
    };

    println!("\n=== DFA ACCEPTING PATHS ===");
    println!("  Enumerate all minimal code sequences that reach accept in the contains DFA.");
    println!("  Then test multi-pattern memmem with ONLY these paths.\n");

    let codes_varbin = fsst_array.codes();

    for needle in &needles {
        let needle_bytes = needle.as_bytes();

        // Build the contains DFA transition table (KMP-based, same as the real FSST DFA)
        let accept = u8::try_from(needle_bytes.len())?;
        let n_states = usize::from(accept) + 1; // 0..needle_len = progress, needle_len = accept

        // Byte-level KMP transitions
        let byte_table = kmp_byte_transitions(needle_bytes)?;

        // Symbol-level transitions: for each (state, code), process all bytes of the symbol
        let mut sym_trans = vec![0u8; n_states * n_symbols];
        for state in 0..=accept {
            for code in 0..n_symbols {
                if state == accept {
                    sym_trans[state as usize * n_symbols + code] = accept;
                    continue;
                }
                let exp = &expansions[code];
                let mut s = state;
                for &b in exp {
                    if s == accept {
                        break;
                    }
                    s = byte_table[s as usize * 256 + b as usize];
                }
                sym_trans[state as usize * n_symbols + code] = s;
            }
        }

        // BFS to find all minimal accepting paths (code sequences from state 0 → accept)
        // "Minimal" = accept reached at the last code (not earlier)
        let paths = enumerate_accepting_paths(&sym_trans, n_symbols, accept, 8);

        // Test: scan compressed stream for any of these paths
        let start = Instant::now();
        let iters = 10u32;
        let mut path_hits = 0u64;
        for _ in 0..iters {
            path_hits = 0;
            codes_varbin.with_iterator(|iter| {
                for codes in iter.flatten() {
                    if paths_match(codes, &paths) {
                        path_hits += 1;
                    }
                }
            });
        }
        let path_ms = start.elapsed().as_secs_f64() * 1000.0 / f64::from(iters);

        let true_hits: u64 = strings.iter().filter(|s| s.contains(needle)).count() as u64;

        let status = if path_hits == true_hits {
            "EXACT"
        } else if path_hits < true_hits {
            "FALSE_NEG"
        } else {
            "FALSE_POS"
        };

        println!(
            "  {:16} paths={:4} hits={:7} true={:7} {:9} ms={:.3}",
            format!("\"{needle}\""),
            paths.len(),
            path_hits,
            true_hits,
            status,
            path_ms,
        );

        // Show paths (up to 20)
        for (i, path) in paths.iter().take(20).enumerate() {
            let decoded: Vec<String> = path
                .iter()
                .map(|&c| format!("{}={}", c, String::from_utf8_lossy(&expansions[c as usize])))
                .collect();
            println!("    path[{i}] [{}]", decoded.join(", "));
        }
        if paths.len() > 20 {
            println!("    ... and {} more", paths.len() - 20);
        }
    }

    Ok(())
}

/// Enumerate all minimal code sequences that go from state 0 to accept.
/// "Minimal" means accept is reached exactly at the last code (not before).
/// `max_len` caps the search depth.
fn enumerate_accepting_paths(
    sym_trans: &[u8],
    n_symbols: usize,
    accept: u8,
    max_len: usize,
) -> Vec<Vec<u8>> {
    let mut results = Vec::new();
    let mut stack: Vec<(u8, Vec<u8>)> = vec![(0, Vec::new())]; // (current_state, path)

    while let Some((state, path)) = stack.pop() {
        if path.len() >= max_len {
            continue;
        }
        for code in 0..n_symbols {
            let next = sym_trans[state as usize * n_symbols + code];
            if next == accept {
                // This code reaches accept — record the path
                let Ok(code) = u8::try_from(code) else {
                    continue;
                };
                let mut full_path = path.clone();
                full_path.push(code);
                results.push(full_path);
            } else if next != state || path.is_empty() {
                // Only follow transitions that make progress (avoid infinite loops)
                // Allow self-loops only at the start (state 0 → 0 means no match, skip)
                if next > 0 || (state == 0 && path.is_empty()) {
                    // Only follow if we're making progress toward accept
                    if next > 0 {
                        let Ok(code) = u8::try_from(code) else {
                            continue;
                        };
                        let mut new_path = path.clone();
                        new_path.push(code);
                        stack.push((next, new_path));
                    }
                }
            }
        }
    }

    // Deduplicate
    let unique: BTreeSet<Vec<u8>> = results.into_iter().collect();
    unique.into_iter().collect()
}

/// Check if any accepting path appears in the code stream (escape-aware).
fn paths_match(codes: &[u8], paths: &[Vec<u8>]) -> bool {
    // Build a set of code-boundary positions (skip escape sequences)
    // Then check if any path matches at any boundary position.
    let mut i = 0;
    while i < codes.len() {
        if codes[i] == 255 {
            i += 2; // skip escape + literal
            continue;
        }
        // Try matching each path starting at this position
        for path in paths {
            if matches_at(codes, i, path) {
                return true;
            }
        }
        i += 1;
    }
    false
}

/// Check if a path matches at position `start` in the codes, respecting escape sequences.
fn matches_at(codes: &[u8], start: usize, path: &[u8]) -> bool {
    let mut ci = start;
    for &expected_code in path {
        if ci >= codes.len() {
            return false;
        }
        if codes[ci] == 255 {
            return false; // hit an escape in the middle — doesn't match
        }
        if codes[ci] != expected_code {
            return false;
        }
        ci += 1;
    }
    true
}

fn kmp_byte_transitions(needle: &[u8]) -> Result<Vec<u8>> {
    let accept = u8::try_from(needle.len())?;
    let n_states = usize::from(accept) + 1;

    // KMP failure function
    let mut failure = vec![0u8; needle.len()];
    let mut k = 0u8;
    for i in 1..needle.len() {
        while k > 0 && needle[k as usize] != needle[i] {
            k = failure[k as usize - 1];
        }
        if needle[k as usize] == needle[i] {
            k += 1;
        }
        failure[i] = k;
    }

    // Build transition table
    let mut table = vec![0u8; n_states * 256];
    for state in 0..=accept {
        for byte in 0..256usize {
            if state == accept {
                table[state as usize * 256 + byte] = accept;
                continue;
            }
            let mut s = state;
            loop {
                if byte == needle[s as usize] as usize {
                    s += 1;
                    break;
                }
                if s == 0 {
                    break;
                }
                s = failure[s as usize - 1];
            }
            table[state as usize * 256 + byte] = s;
        }
    }
    Ok(table)
}

fn symbol_bytes(sym: &Symbol, len: u8) -> Vec<u8> {
    sym.to_u64().to_le_bytes()[..len as usize].to_vec()
}
