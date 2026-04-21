// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Direct DFA micro-benchmark comparing the default (skip + 2× unroll +
//! early-exit) and zero-branch (no skip, no mid-loop branches) scan
//! variants of the contains DFA. Bypasses the full Vortex `LIKE`
//! execution pipeline so the per-string overhead outside the DFA doesn't
//! dominate the measurement.

#![expect(clippy::unwrap_used)]

use std::fmt;
use std::sync::LazyLock;

use divan::Bencher;
#[expect(deprecated)]
use vortex_array::ToCanonical;
use vortex_array::arrays::varbin::VarBinArrayExt;
use vortex_fsst::FSSTArray;
use vortex_fsst::FSSTArrayExt;
use vortex_fsst::dfa_bench_api::ContainsBench;
use vortex_fsst::test_utils::NUM_STRINGS;
use vortex_fsst::test_utils::make_fsst_clickbench_urls;
use vortex_fsst::test_utils::make_fsst_emails;
use vortex_fsst::test_utils::make_fsst_file_paths;
use vortex_fsst::test_utils::make_fsst_json_strings;
use vortex_fsst::test_utils::make_fsst_log_lines;
use vortex_fsst::test_utils::make_fsst_rare_match;
use vortex_fsst::test_utils::make_fsst_short_urls;

fn main() {
    divan::main();
}

const N: usize = NUM_STRINGS;

static FSST_URLS: LazyLock<FSSTArray> = LazyLock::new(|| make_fsst_short_urls(N));
static FSST_CB_URLS: LazyLock<FSSTArray> = LazyLock::new(|| make_fsst_clickbench_urls(N));
static FSST_LOG_LINES: LazyLock<FSSTArray> = LazyLock::new(|| make_fsst_log_lines(N));
static FSST_JSON_STRINGS: LazyLock<FSSTArray> = LazyLock::new(|| make_fsst_json_strings(N));
static FSST_FILE_PATHS: LazyLock<FSSTArray> = LazyLock::new(|| make_fsst_file_paths(N));
static FSST_EMAILS: LazyLock<FSSTArray> = LazyLock::new(|| make_fsst_emails(N));
static FSST_RARE_MATCH: LazyLock<FSSTArray> = LazyLock::new(|| make_fsst_rare_match(N));

enum Dataset {
    Urls,
    Cb,
    Log,
    Json,
    Path,
    Email,
    Rare,
}

impl fmt::Display for Dataset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Urls => f.write_str("urls"),
            Self::Cb => f.write_str("cb"),
            Self::Log => f.write_str("log"),
            Self::Json => f.write_str("json"),
            Self::Path => f.write_str("path"),
            Self::Email => f.write_str("email"),
            Self::Rare => f.write_str("rare"),
        }
    }
}

impl Dataset {
    fn fsst_array(&self) -> &'static FSSTArray {
        match self {
            Self::Urls => &FSST_URLS,
            Self::Cb => &FSST_CB_URLS,
            Self::Log => &FSST_LOG_LINES,
            Self::Json => &FSST_JSON_STRINGS,
            Self::Path => &FSST_FILE_PATHS,
            Self::Email => &FSST_EMAILS,
            Self::Rare => &FSST_RARE_MATCH,
        }
    }

    fn contains_needle(&self) -> &'static [u8] {
        match self {
            Self::Urls => b"google",
            Self::Cb => b"yandex",
            Self::Log => b"Googlebot",
            Self::Json => b"enterprise",
            Self::Path => b"target/release",
            Self::Email => b"gmail",
            Self::Rare => b"xyzzy",
        }
    }
}

/// Pre-computed per-dataset inputs for the micro-bench: the concatenated
/// code bytes and the corresponding per-string offsets.
struct PreparedCodes {
    bytes: Vec<u8>,
    offsets: Vec<u32>,
}

fn prepare(arr: &FSSTArray) -> PreparedCodes {
    let codes = arr.codes();
    let bytes = codes.bytes().as_slice().to_vec();
    #[expect(deprecated)]
    let offs = codes.offsets().to_primitive();
    let mut out = Vec::with_capacity(offs.len());
    vortex_array::match_each_integer_ptype!(offs.ptype(), |T| {
        for &v in offs.as_slice::<T>() {
            out.push(v as u32);
        }
    });
    PreparedCodes { bytes, offsets: out }
}

fn run(bencher: Bencher, prep: &PreparedCodes, dfa: ContainsBench, branchless: bool) {
    bencher.bench_local(|| {
        let mut count = 0usize;
        let mut start = prep.offsets[0] as usize;
        for i in 0..(prep.offsets.len() - 1) {
            let end = prep.offsets[i + 1] as usize;
            let codes = &prep.bytes[start..end];
            let hit = if branchless {
                dfa.matches_branchless(codes)
            } else {
                dfa.matches(codes)
            };
            if hit {
                count += 1;
            }
            start = end;
        }
        divan::black_box(count)
    });
}

fn make_dfa(ds: &Dataset) -> (ContainsBench, PreparedCodes) {
    let arr = ds.fsst_array();
    let dfa = ContainsBench::new(
        arr.symbols().as_slice(),
        arr.symbol_lengths().as_slice(),
        ds.contains_needle(),
    )
    .unwrap();
    let prep = prepare(arr);
    (dfa, prep)
}

#[divan::bench(args = [
    Dataset::Urls, Dataset::Cb, Dataset::Log, Dataset::Json,
    Dataset::Path, Dataset::Email, Dataset::Rare,
])]
fn default_scan(bencher: Bencher, dataset: &Dataset) {
    let (dfa, prep) = make_dfa(dataset);
    run(bencher, &prep, dfa, false);
}

#[divan::bench(args = [
    Dataset::Urls, Dataset::Cb, Dataset::Log, Dataset::Json,
    Dataset::Path, Dataset::Email, Dataset::Rare,
])]
fn branchless_scan(bencher: Bencher, dataset: &Dataset) {
    let (dfa, prep) = make_dfa(dataset);
    run(bencher, &prep, dfa, true);
}
