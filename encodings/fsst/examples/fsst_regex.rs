//! FSST Regex-over-Compressed-Codes Prototype
//!
//! Demonstrates that regex matching can be performed directly on FSST-compressed
//! data by building a "fused DFA" where each FSST symbol code triggers a single
//! transition that simulates feeding all the symbol's bytes through the regex DFA.
//!
//! Key insight: with FSST symbols averaging N bytes, the fused DFA executes ~N×
//! fewer transitions than the byte-level DFA, yielding proportional speedups.
//!
//! Run: cargo run --example fsst_regex -p vortex-fsst

#![allow(
    clippy::cast_precision_loss,
    clippy::needless_range_loop,
    clippy::cast_possible_truncation,
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::min_ident_chars,
    clippy::disallowed_types,
    clippy::too_many_arguments,
    clippy::many_single_char_names
)]

use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;
use std::fs;
use std::io::Write;
use std::time::Instant;

use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use regex_automata::Anchored;
use regex_automata::dfa::Automaton;
use regex_automata::dfa::dense::DFA;
use regex_automata::util::primitives::StateID;

const ESCAPE_CODE: u8 = 255;

// ---------------------------------------------------------------------------
// Fused regex DFA over FSST codes
// ---------------------------------------------------------------------------

/// A DFA that operates directly on FSST-compressed code sequences.
///
/// For each (state, symbol_code) pair, the transition is precomputed by
/// simulating feeding all of the symbol's bytes through the original
/// byte-level regex DFA. This reduces the number of transitions from
/// O(uncompressed_bytes) to O(compressed_codes).
struct FusedRegexDfa {
    /// Fused transition table: [idx * 256 + code] -> next idx
    sym_transitions: Vec<u32>,
    /// Whether processing this (state, code) hits a match state mid-symbol.
    /// Needed because a multi-byte symbol might cross a match boundary.
    sym_hit_match: Vec<bool>,
    /// Byte-level transitions for escaped literal bytes: [idx * 256 + byte] -> next idx
    escape_transitions: Vec<u32>,
    /// EOI transitions: [idx] -> next idx after end-of-input
    eoi_transitions: Vec<u32>,
    /// Flat bitmap: is_match[idx] = true if this state is a match state
    is_match: Vec<bool>,
    /// Dead state index
    dead_idx: u32,
    /// Start state index
    start_idx: u32,
    /// Sentinel value indicating escape code encountered
    sentinel: u32,
    /// Number of states
    n_states: u32,
}

impl FusedRegexDfa {
    /// Build a fused DFA from a regex pattern and FSST symbol table.
    fn new(
        pattern: &str,
        symbols: &[fsst::Symbol],
        symbol_lengths: &[u8],
        n_symbols: usize,
    ) -> Result<Self, String> {
        // Build byte-level DFA
        let dfa = DFA::new(pattern).map_err(|e| format!("DFA build error: {e}"))?;

        // Get universal start state (unanchored)
        let start = dfa
            .universal_start_state(Anchored::No)
            .ok_or("DFA has no universal unanchored start state")?;

        // BFS to collect all reachable states
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
        let sentinel = n_states; // one past the last valid index

        // Build state index mapping
        let state_to_idx: HashMap<StateID, u32> = states
            .iter()
            .enumerate()
            .map(|(i, &s)| (s, i as u32))
            .collect();

        // Find dead state index
        let dead_idx = states
            .iter()
            .position(|&s| dfa.is_dead_state(s))
            .map(|i| i as u32)
            .unwrap_or(0);

        let start_idx = state_to_idx[&start];

        // Build match state bitmap
        let is_match: Vec<bool> = states.iter().map(|&s| dfa.is_match_state(s)).collect();

        // Build fused symbol transition table
        let table_size = n_states as usize * 256;
        let mut sym_transitions = vec![dead_idx; table_size];
        let mut sym_hit_match = vec![false; table_size];
        let mut escape_transitions = vec![dead_idx; table_size];
        let mut eoi_transitions = vec![dead_idx; n_states as usize];

        for (state_idx, &state_id) in states.iter().enumerate() {
            let base = state_idx * 256;

            // Symbol codes: simulate all bytes of the symbol
            for code in 0..n_symbols {
                let sym = symbols[code];
                let len = symbol_lengths[code] as usize;
                let sym_bytes = sym.to_u64().to_le_bytes();
                let bytes = &sym_bytes[..len];

                let mut s = state_id;
                let mut hit_match = false;
                for &b in bytes {
                    s = dfa.next_state(s, b);
                    if dfa.is_match_state(s) {
                        hit_match = true;
                    }
                }
                sym_transitions[base + code] = state_to_idx[&s];
                sym_hit_match[base + code] = hit_match;
            }

            // Escape code -> sentinel
            sym_transitions[base + ESCAPE_CODE as usize] = sentinel;

            // Unused codes (n_symbols..255) stay as dead_idx

            // Byte-level transitions for escaped bytes
            for byte in 0..=255u8 {
                let next = dfa.next_state(state_id, byte);
                escape_transitions[base + byte as usize] = state_to_idx[&next];
            }

            // EOI transition
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

    /// Scan FSST-compressed codes and return whether the regex matches.
    #[inline(never)]
    fn matches(&self, codes: &[u8]) -> bool {
        let mut state = self.start_idx;
        let mut pos = 0;
        while pos < codes.len() {
            let code = codes[pos];
            pos += 1;
            let table_idx = state as usize * 256 + code as usize;
            let next = self.sym_transitions[table_idx];
            if next == self.sentinel {
                // Escape: next byte is literal
                if pos >= codes.len() {
                    return false;
                }
                let byte = codes[pos];
                pos += 1;
                state = self.escape_transitions[state as usize * 256 + byte as usize];
            } else {
                // Check if processing this symbol's bytes hit a match mid-symbol
                if self.sym_hit_match[table_idx] {
                    return true;
                }
                state = next;
            }
            if self.is_match[state as usize] {
                return true;
            }
            if state == self.dead_idx {
                return false;
            }
        }
        // End of input
        let eoi_state = self.eoi_transitions[state as usize];
        self.is_match[eoi_state as usize]
    }

    fn report(&self, pattern: &str) {
        println!(
            "  Pattern: \"{}\" | States: {} | Match states: {} | Dead idx: {}",
            pattern,
            self.n_states,
            self.is_match.iter().filter(|&&m| m).count(),
            self.dead_idx,
        );
        println!(
            "  Sym table size: {} entries ({:.1} KB) | Esc table size: {} entries ({:.1} KB)",
            self.sym_transitions.len(),
            self.sym_transitions.len() as f64 * 4.0 / 1024.0,
            self.escape_transitions.len(),
            self.escape_transitions.len() as f64 * 4.0 / 1024.0,
        );
    }
}

// ---------------------------------------------------------------------------
// Byte-level DFA baseline scanner
// ---------------------------------------------------------------------------

/// Scan raw bytes with a byte-level regex DFA (same engine, same DFA, no FSST).
/// This is the "fair" baseline: same DFA structure, just more transitions.
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
// Text generator (Zipf English prose)
// ---------------------------------------------------------------------------

fn generate_text(n: usize, rng: &mut StdRng) -> Vec<String> {
    let words = [
        "the",
        "be",
        "to",
        "of",
        "and",
        "a",
        "in",
        "that",
        "have",
        "I",
        "it",
        "for",
        "not",
        "on",
        "with",
        "he",
        "as",
        "you",
        "do",
        "at",
        "this",
        "but",
        "his",
        "by",
        "from",
        "they",
        "we",
        "say",
        "her",
        "she",
        "or",
        "an",
        "will",
        "my",
        "one",
        "all",
        "would",
        "there",
        "their",
        "what",
        "so",
        "up",
        "out",
        "if",
        "about",
        "who",
        "get",
        "which",
        "go",
        "me",
        "when",
        "make",
        "can",
        "like",
        "time",
        "no",
        "just",
        "him",
        "know",
        "take",
        "people",
        "into",
        "year",
        "your",
        "good",
        "some",
        "could",
        "them",
        "see",
        "other",
        "than",
        "then",
        "now",
        "look",
        "only",
        "come",
        "its",
        "over",
        "think",
        "also",
        "back",
        "after",
        "use",
        "two",
        "how",
        "our",
        "work",
        "first",
        "well",
        "way",
        "even",
        "new",
        "want",
        "because",
        "any",
        "these",
        "give",
        "day",
        "most",
        "error",
        "warning",
        "critical",
        "debug",
        "info",
        "performance",
        "algorithm",
        "database",
        "compression",
        "implementation",
    ];
    let harmonic: f64 = (1..=words.len()).map(|r| 1.0 / r as f64).sum();
    let cdf: Vec<f64> = (1..=words.len())
        .scan(0.0, |acc, r| {
            *acc += 1.0 / (r as f64 * harmonic);
            Some(*acc)
        })
        .collect();

    (0..n)
        .map(|_| {
            let wc = rng.random_range(5..30usize);
            let mut s = String::with_capacity(wc * 6);
            for i in 0..wc {
                if i > 0 {
                    s.push(' ');
                }
                let u: f64 = rng.random();
                let idx = cdf.partition_point(|&p| p < u).min(cdf.len() - 1);
                s.push_str(words[idx]);
            }
            s
        })
        .collect()
}

fn generate_urls_for_regex(n: usize, rng: &mut StdRng) -> Vec<String> {
    let domains = [
        "google.com",
        "facebook.com",
        "youtube.com",
        "amazon.com",
        "github.com",
        "reddit.com",
        "example.com",
        "api.stripe.com",
    ];
    let paths = [
        "/",
        "/index.html",
        "/api/v1/users",
        "/search",
        "/login",
        "/dashboard",
        "/blog/2024/01/hello",
    ];
    let harmonic: f64 = (1..=domains.len()).map(|r| 1.0 / r as f64).sum();
    let cdf: Vec<f64> = (1..=domains.len())
        .scan(0.0, |acc, r| {
            *acc += 1.0 / (r as f64 * harmonic);
            Some(*acc)
        })
        .collect();
    (0..n)
        .map(|_| {
            let proto = if rng.random_range(0..10u32) < 8 {
                "https://"
            } else {
                "http://"
            };
            let u: f64 = rng.random();
            let d = cdf.partition_point(|&p| p < u).min(cdf.len() - 1);
            let p = rng.random_range(0..paths.len());
            format!("{}{}{}", proto, domains[d], paths[p])
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Benchmark harness
// ---------------------------------------------------------------------------

struct BenchResult {
    dataset: String,
    pattern: String,
    pattern_type: String,
    fused_ns_per_string: f64,
    byte_ns_per_string: f64,
    speedup: f64,
    n_fused_matches: usize,
    _n_byte_matches: usize,
    mean_sym_len: f64,
    escape_rate: f64,
    n_dfa_states: u32,
}

fn benchmark_pattern(
    dataset_name: &str,
    pattern: &str,
    pattern_type: &str,
    compressed: &[Vec<u8>],
    raw_strings: &[String],
    symbols: &[fsst::Symbol],
    symbol_lengths: &[u8],
    n_symbols: usize,
    mean_sym_len: f64,
    escape_rate: f64,
) -> Option<BenchResult> {
    // Build fused DFA
    let fused = match FusedRegexDfa::new(pattern, symbols, symbol_lengths, n_symbols) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("  Skipping pattern \"{pattern}\": {e}");
            return None;
        }
    };
    fused.report(pattern);

    // Build byte-level DFA for baseline
    let byte_dfa = match DFA::new(pattern) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("  Skipping byte DFA for \"{pattern}\": {e}");
            return None;
        }
    };
    let byte_start = byte_dfa
        .universal_start_state(Anchored::No)
        .expect("byte DFA has no universal start");

    // Warmup
    let n_warmup = 1000.min(raw_strings.len());
    for codes in &compressed[..n_warmup] {
        std::hint::black_box(fused.matches(codes));
    }
    for s in &raw_strings[..n_warmup] {
        std::hint::black_box(byte_dfa_matches(&byte_dfa, byte_start, s.as_bytes()));
    }

    // Benchmark fused DFA on compressed codes
    let iters = 5;
    let mut fused_total = std::time::Duration::ZERO;
    let mut n_fused_matches = 0;
    for _ in 0..iters {
        n_fused_matches = 0;
        let t = Instant::now();
        for codes in compressed {
            if fused.matches(codes) {
                n_fused_matches += 1;
            }
        }
        fused_total += t.elapsed();
    }
    let fused_ns = fused_total.as_nanos() as f64 / (iters as f64 * compressed.len() as f64);

    // Benchmark byte-level DFA on raw strings
    let mut byte_total = std::time::Duration::ZERO;
    let mut n_byte_matches = 0;
    for _ in 0..iters {
        n_byte_matches = 0;
        let t = Instant::now();
        for s in raw_strings {
            if byte_dfa_matches(&byte_dfa, byte_start, s.as_bytes()) {
                n_byte_matches += 1;
            }
        }
        byte_total += t.elapsed();
    }
    let byte_ns = byte_total.as_nanos() as f64 / (iters as f64 * raw_strings.len() as f64);

    assert_eq!(
        n_fused_matches, n_byte_matches,
        "Match count mismatch for pattern \"{pattern}\": fused={n_fused_matches}, byte={n_byte_matches}"
    );

    let speedup = byte_ns / fused_ns;
    println!(
        "  -> Fused: {:.1} ns/string | Byte: {:.1} ns/string | Speedup: {:.2}x | Matches: {}/{}",
        fused_ns,
        byte_ns,
        speedup,
        n_fused_matches,
        raw_strings.len()
    );

    Some(BenchResult {
        dataset: dataset_name.to_string(),
        pattern: pattern.to_string(),
        pattern_type: pattern_type.to_string(),
        fused_ns_per_string: fused_ns,
        byte_ns_per_string: byte_ns,
        speedup,
        n_fused_matches,
        _n_byte_matches: n_byte_matches,
        mean_sym_len,
        escape_rate,
        n_dfa_states: fused.n_states,
    })
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let n = 100_000;
    let seed = 42;
    let mut rng = StdRng::seed_from_u64(seed);

    println!("# FSST Regex-over-Compressed-Codes Prototype\n");
    println!("Generating {} strings per dataset...\n", n);

    let out_dir = "encodings/fsst/data";
    fs::create_dir_all(out_dir).ok();

    // Generate datasets
    let datasets: Vec<(&str, Vec<String>)> = vec![
        ("english_prose", generate_text(n, &mut rng)),
        ("urls", generate_urls_for_regex(n, &mut rng)),
    ];

    // Patterns to test per dataset
    let text_patterns: Vec<(&str, &str)> = vec![
        ("the", "contains_short"),
        ("would", "contains_medium"),
        ("implementation", "contains_long"),
        ("error|warning|critical", "alternation"),
        ("[a-z]{10,}", "char_class"),
        ("th[aeiou]", "char_class_short"),
        ("the .* of", "dot_star"),
    ];

    let url_patterns: Vec<(&str, &str)> = vec![
        ("https://", "contains_proto"),
        ("google", "contains_domain"),
        ("api/v[0-9]+/", "version_pattern"),
        ("/users|/login|/search", "path_alternation"),
        ("\\?.*utm_source", "query_param"),
    ];

    let mut all_results: Vec<BenchResult> = Vec::new();

    for (name, strings) in &datasets {
        println!("\n## Dataset: {name}");

        // Compress
        let bytes_vec: Vec<&[u8]> = strings.iter().map(|s| s.as_bytes()).collect();
        let compressor = fsst::Compressor::train(&bytes_vec);
        let compressed: Vec<Vec<u8>> = bytes_vec.iter().map(|b| compressor.compress(b)).collect();

        let total_raw: usize = strings.iter().map(|s| s.len()).sum();
        let total_compressed: usize = compressed.iter().map(|c| c.len()).sum();
        let compression_ratio = total_raw as f64 / total_compressed as f64;

        let symbols = compressor.symbol_table();
        let symbol_lengths = compressor.symbol_lengths();
        let n_symbols = symbols.len();

        let mean_sym_len = if n_symbols > 0 {
            symbol_lengths[..n_symbols]
                .iter()
                .map(|&l| l as f64)
                .sum::<f64>()
                / n_symbols as f64
        } else {
            1.0
        };

        // Count escapes
        let mut escape_count = 0u64;
        let mut total_codes = 0u64;
        for codes in &compressed {
            let mut i = 0;
            while i < codes.len() {
                if codes[i] == ESCAPE_CODE {
                    escape_count += 1;
                    i += 2;
                } else {
                    i += 1;
                }
                total_codes += 1;
            }
        }
        let escape_rate = escape_count as f64 / total_codes as f64;

        println!(
            "  Strings: {} | Compression: {:.2}x | Mean sym len: {:.2} | Escape rate: {:.1}%",
            strings.len(),
            compression_ratio,
            mean_sym_len,
            escape_rate * 100.0,
        );

        let patterns = match *name {
            "english_prose" => &text_patterns[..],
            "urls" => &url_patterns[..],
            _ => &text_patterns[..],
        };

        for &(pattern, pattern_type) in patterns {
            if let Some(result) = benchmark_pattern(
                name,
                pattern,
                pattern_type,
                &compressed,
                strings,
                &symbols[..n_symbols],
                &symbol_lengths[..n_symbols],
                n_symbols,
                mean_sym_len,
                escape_rate,
            ) {
                all_results.push(result);
            }
        }
    }

    // Write CSV
    let csv_path = format!("{out_dir}/regex_bench.csv");
    let mut f = fs::File::create(&csv_path).expect("create regex CSV");
    writeln!(
        f,
        "dataset,pattern,pattern_type,fused_ns,byte_ns,speedup,matches,mean_sym_len,escape_rate,dfa_states"
    )
    .unwrap();
    for r in &all_results {
        writeln!(
            f,
            "{},{},{},{:.2},{:.2},{:.3},{},{:.3},{:.4},{}",
            r.dataset,
            r.pattern,
            r.pattern_type,
            r.fused_ns_per_string,
            r.byte_ns_per_string,
            r.speedup,
            r.n_fused_matches,
            r.mean_sym_len,
            r.escape_rate,
            r.n_dfa_states,
        )
        .unwrap();
    }
    eprintln!("\nWrote {csv_path}");

    // Summary table
    println!("\n\n# Summary");
    println!("| Dataset | Pattern | Type | Fused ns | Byte ns | Speedup | Matches | DFA States |");
    println!("|---------|---------|------|----------|---------|---------|---------|------------|");
    for r in &all_results {
        println!(
            "| {} | {} | {} | {:.1} | {:.1} | {:.2}x | {} | {} |",
            r.dataset,
            r.pattern,
            r.pattern_type,
            r.fused_ns_per_string,
            r.byte_ns_per_string,
            r.speedup,
            r.n_fused_matches,
            r.n_dfa_states,
        );
    }

    println!("\n\nKey insight: fused DFA speedup correlates with mean symbol length.");
    println!(
        "FSST symbols average {:.1} bytes → each code transition replaces ~{:.1} byte transitions.",
        all_results.first().map(|r| r.mean_sym_len).unwrap_or(1.0),
        all_results.first().map(|r| r.mean_sym_len).unwrap_or(1.0),
    );
}
