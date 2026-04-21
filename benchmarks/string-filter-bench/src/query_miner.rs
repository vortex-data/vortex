// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Sample a dataset's strings and mine benchmark queries at various selectivities.
//! Produces a JSON file of queries labeled by type, selectivity, and FSST difficulty.

use std::io::BufRead;
use std::io::BufReader;
use std::path::PathBuf;

use anyhow::Result;
use clap::Args;
use itertools::Itertools;
use serde::Deserialize;
use serde::Serialize;
use vortex::utils::aliases::hash_map::HashMap;

use crate::data_prep::DatasetName;
use crate::data_prep::{self};

#[derive(Args)]
pub struct MineArgs {
    /// Which dataset to mine queries from
    #[arg(value_enum)]
    pub dataset: DatasetName,

    /// Number of strings to sample for mining
    #[arg(long, default_value_t = 50_000)]
    pub sample_size: usize,

    /// Number of queries to generate per category
    #[arg(long, default_value_t = 5)]
    pub queries_per_category: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinedQuery {
    /// SQL LIKE pattern or regex
    pub pattern: String,
    /// One of: like_prefix, like_substr, regex_basic
    pub query_type: String,
    /// Estimated selectivity: high, medium, low
    pub selectivity: String,
    /// Whether FSST can handle this natively (easy/hard/unsupported)
    pub fsst_difficulty: String,
    /// Approximate match fraction in the sample
    pub match_fraction: f64,
}

pub fn run(args: MineArgs) -> Result<()> {
    let strings = load_strings(&args.dataset)?;
    let sample: Vec<&str> = strings
        .iter()
        .take(args.sample_size)
        .map(|s| s.as_str())
        .collect();

    let n = sample.len();
    if n == 0 {
        anyhow::bail!("No strings found in dataset");
    }

    println!("Mining queries from {n} strings...");

    let mut queries = Vec::new();

    // Mine common prefixes
    let prefixes = mine_prefixes(&sample, args.queries_per_category);
    for (prefix, frac) in &prefixes {
        let selectivity = selectivity_label(*frac);
        queries.push(MinedQuery {
            pattern: format!("{prefix}%"),
            query_type: "like_prefix".into(),
            selectivity,
            fsst_difficulty: "easy".into(),
            match_fraction: *frac,
        });
    }

    // Mine common substrings
    let substrings = mine_substrings(&sample, args.queries_per_category);
    for (substr, frac) in &substrings {
        let selectivity = selectivity_label(*frac);
        queries.push(MinedQuery {
            pattern: format!("%{substr}%"),
            query_type: "like_substr".into(),
            selectivity,
            fsst_difficulty: "easy".into(),
            match_fraction: *frac,
        });
    }

    // Mine n-grams for regex patterns
    let ngrams = mine_ngrams(&sample, 3, args.queries_per_category);
    for (ngram, frac) in &ngrams {
        queries.push(MinedQuery {
            pattern: regex::escape(ngram),
            query_type: "regex_basic".into(),
            selectivity: selectivity_label(*frac),
            fsst_difficulty: "easy".into(),
            match_fraction: *frac,
        });
    }

    // Add a few "hard" regex patterns that FSST can't handle natively
    let hard_patterns = vec![
        (r"^https?://[a-z]+\.", "regex_basic", "unsupported"),
        (r"\d{3,5}", "regex_basic", "unsupported"),
        (r"[A-Z][a-z]+_[a-z]+", "regex_basic", "unsupported"),
    ];
    for (pat, qtype, difficulty) in hard_patterns {
        let re = regex::Regex::new(pat).unwrap();
        let matches = sample.iter().filter(|s| re.is_match(s)).count();
        let frac = matches as f64 / n as f64;
        queries.push(MinedQuery {
            pattern: pat.to_string(),
            query_type: qtype.into(),
            selectivity: selectivity_label(frac),
            fsst_difficulty: difficulty.into(),
            match_fraction: frac,
        });
    }

    // Add rare/control patterns
    queries.push(MinedQuery {
        pattern: "XYZZY_IMPOSSIBLE_99%".into(),
        query_type: "like_prefix".into(),
        selectivity: "low".into(),
        fsst_difficulty: "easy".into(),
        match_fraction: 0.0,
    });
    queries.push(MinedQuery {
        pattern: "%XYZZY_IMPOSSIBLE_99%".into(),
        query_type: "like_substr".into(),
        selectivity: "low".into(),
        fsst_difficulty: "easy".into(),
        match_fraction: 0.0,
    });

    // Write output
    let out_path = queries_path(&args.dataset);
    std::fs::create_dir_all(out_path.parent().unwrap())?;
    let json = serde_json::to_string_pretty(&queries)?;
    std::fs::write(&out_path, &json)?;

    println!("\nMined {} queries:", queries.len());
    for q in &queries {
        println!(
            "  [{:12}] {:10} fsst={:12} sel={:.4}  {}",
            q.query_type, q.selectivity, q.fsst_difficulty, q.match_fraction, q.pattern
        );
    }
    println!("\nWritten to: {}", out_path.display());

    Ok(())
}

pub fn queries_path(dataset: &DatasetName) -> PathBuf {
    data_prep::output_dir(dataset).join(format!("{}_queries.json", dataset.file_stem()))
}

pub fn strings_path(dataset: &DatasetName) -> PathBuf {
    data_prep::output_dir(dataset).join(format!("{}_strings.txt", dataset.file_stem()))
}

fn selectivity_label(frac: f64) -> String {
    if frac > 0.1 {
        "high".into()
    } else if frac > 0.01 {
        "medium".into()
    } else {
        "low".into()
    }
}

/// Load raw strings from the text sidecar file.
pub fn load_strings(dataset: &DatasetName) -> Result<Vec<String>> {
    let path = strings_path(dataset);
    if !path.exists() {
        anyhow::bail!(
            "Dataset not prepared yet. Run `prep {dataset}` first.\nExpected: {}",
            path.display()
        );
    }

    let file = std::fs::File::open(&path)?;
    let reader = BufReader::new(file);
    let strings: Vec<String> = reader.lines().collect::<Result<_, _>>()?;
    Ok(strings)
}

/// Find common prefixes at various lengths and return (prefix, match_fraction) pairs.
fn mine_prefixes(strings: &[&str], count: usize) -> Vec<(String, f64)> {
    let n = strings.len() as f64;
    let mut prefix_counts: HashMap<String, usize> = HashMap::default();

    for &s in strings {
        if !s.is_ascii() {
            continue;
        }
        for plen in [3, 5, 8, 12, 16, 20] {
            if s.len() >= plen {
                *prefix_counts.entry(s[..plen].to_string()).or_default() += 1;
            }
        }
    }

    let mut by_frac: Vec<(String, f64)> = prefix_counts
        .into_iter()
        .map(|(p, c)| (p, c as f64 / n))
        .collect();
    by_frac.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    pick_diverse(by_frac, count)
}

/// Find common substrings (3-7 chars) and return (substring, match_fraction) pairs.
fn mine_substrings(strings: &[&str], count: usize) -> Vec<(String, f64)> {
    let n = strings.len() as f64;
    let mut substr_counts: HashMap<String, usize> = HashMap::default();

    for &s in strings.iter().take(10_000) {
        if !s.is_ascii() {
            continue;
        }
        for slen in [3, 5, 7] {
            if s.len() >= slen {
                for start in (0..s.len().saturating_sub(slen)).step_by(slen) {
                    *substr_counts
                        .entry(s[start..start + slen].to_string())
                        .or_default() += 1;
                }
            }
        }
    }

    let candidates: Vec<String> = substr_counts
        .into_iter()
        .filter(|(_, c)| *c >= 5)
        .sorted_by_key(|(_, c)| std::cmp::Reverse(*c))
        .take(200)
        .map(|(s, _)| s)
        .collect();

    let mut measured: Vec<(String, f64)> = candidates
        .into_iter()
        .map(|sub| {
            let matches = strings.iter().filter(|s| s.contains(&sub)).count();
            (sub, matches as f64 / n)
        })
        .collect();
    measured.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    pick_diverse(measured, count)
}

/// Find common n-grams and return (ngram, match_fraction) pairs.
fn mine_ngrams(strings: &[&str], ngram_len: usize, count: usize) -> Vec<(String, f64)> {
    let n = strings.len() as f64;
    let mut ngram_counts: HashMap<String, usize> = HashMap::default();

    for &s in strings.iter().take(10_000) {
        if !s.is_ascii() {
            continue;
        }
        if s.len() >= ngram_len {
            for i in 0..s.len() - ngram_len + 1 {
                *ngram_counts
                    .entry(s[i..i + ngram_len].to_string())
                    .or_default() += 1;
            }
        }
    }

    let candidates: Vec<String> = ngram_counts
        .into_iter()
        .filter(|(_, c)| *c >= 10)
        .sorted_by_key(|(_, c)| std::cmp::Reverse(*c))
        .take(100)
        .map(|(s, _)| s)
        .collect();

    let mut measured: Vec<(String, f64)> = candidates
        .into_iter()
        .map(|ng| {
            let matches = strings.iter().filter(|s| s.contains(&ng)).count();
            (ng, matches as f64 / n)
        })
        .collect();
    measured.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    pick_diverse(measured, count)
}

/// Pick diverse entries spanning high/medium/low selectivity.
fn pick_diverse(sorted_desc: Vec<(String, f64)>, count: usize) -> Vec<(String, f64)> {
    if sorted_desc.len() <= count {
        return sorted_desc;
    }

    let mut result = Vec::with_capacity(count);
    let step = sorted_desc.len() / count;
    for i in 0..count {
        result.push(sorted_desc[i * step].clone());
    }
    result
}
