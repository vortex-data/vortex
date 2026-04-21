// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Reproduces the `string-filter-bench run clickbench-url` pattern panel
//! on *synthetic* ClickBench-shaped URL data. The sandbox this runs in
//! cannot reach the real ClickBench HTTP endpoint, so the data is the
//! `make_fsst_clickbench_urls` generator from `test_utils`. Absolute
//! numbers will differ from real-data ClickBench runs, but the relative
//! Raw / FSST / Decomp+LIKE shape is what we want to track.

#![expect(clippy::unwrap_used)]
#![expect(clippy::expect_used)]

use std::sync::LazyLock;
use std::time::Duration;
use std::time::Instant;

use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::scalar_fn::ScalarFnFactoryExt;
use vortex_array::scalar_fn::fns::like::Like;
use vortex_array::scalar_fn::fns::like::LikeOptions;
use vortex_array::session::ArraySession;
use vortex_fsst::FSSTArray;
use vortex_fsst::test_utils::generate_clickbench_urls;
use vortex_fsst::test_utils::make_fsst_clickbench_urls;
use vortex_session::VortexSession;

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

const N: usize = 100_000;
const ITERATIONS: usize = 10;
const WARMUP: usize = 3;

static FSST: LazyLock<FSSTArray> = LazyLock::new(|| make_fsst_clickbench_urls(N));
static STRINGS: LazyLock<Vec<String>> = LazyLock::new(|| generate_clickbench_urls(N));

fn median_ms(times: &mut [Duration]) -> f64 {
    times.sort();
    times[times.len() / 2].as_secs_f64() * 1000.0
}

fn time_raw(pattern_regex: &regex_like::Pattern) -> f64 {
    let strings: &[String] = &STRINGS;
    for _ in 0..WARMUP {
        let _ = strings.iter().filter(|s| pattern_regex.is_match(s)).count();
    }
    let mut times = Vec::with_capacity(ITERATIONS);
    for _ in 0..ITERATIONS {
        let t = Instant::now();
        let n = strings.iter().filter(|s| pattern_regex.is_match(s)).count();
        times.push(t.elapsed());
        divan::black_box(n);
    }
    median_ms(&mut times)
}

fn time_fsst_dfa(like_pat: &str) -> f64 {
    let fsst: &FSSTArray = &FSST;
    let arr = fsst.clone().into_array();
    let pat = ConstantArray::new(like_pat, fsst.len()).into_array();
    for _ in 0..WARMUP {
        let mut ctx = SESSION.create_execution_ctx();
        let r = Like
            .try_new_array(fsst.len(), LikeOptions::default(), [arr.clone(), pat.clone()])
            .unwrap()
            .into_array()
            .execute::<Canonical>(&mut ctx)
            .unwrap();
        divan::black_box(r);
    }
    let mut times = Vec::with_capacity(ITERATIONS);
    for _ in 0..ITERATIONS {
        let mut ctx = SESSION.create_execution_ctx();
        let t = Instant::now();
        let r = Like
            .try_new_array(fsst.len(), LikeOptions::default(), [arr.clone(), pat.clone()])
            .unwrap()
            .into_array()
            .execute::<Canonical>(&mut ctx)
            .unwrap();
        times.push(t.elapsed());
        divan::black_box(r);
    }
    median_ms(&mut times)
}

fn time_decompress_like(like_pat: &str) -> f64 {
    let fsst: &FSSTArray = &FSST;
    let arr = fsst.clone().into_array();
    let pat = ConstantArray::new(like_pat, fsst.len()).into_array();
    for _ in 0..WARMUP {
        let mut ctx = SESSION.create_execution_ctx();
        let decomp = arr
            .clone()
            .execute::<Canonical>(&mut ctx)
            .unwrap()
            .into_array();
        let r = Like
            .try_new_array(fsst.len(), LikeOptions::default(), [decomp, pat.clone()])
            .unwrap()
            .into_array()
            .execute::<Canonical>(&mut ctx)
            .unwrap();
        divan::black_box(r);
    }
    let mut times = Vec::with_capacity(ITERATIONS);
    for _ in 0..ITERATIONS {
        let mut ctx = SESSION.create_execution_ctx();
        let t = Instant::now();
        let decomp = arr
            .clone()
            .execute::<Canonical>(&mut ctx)
            .unwrap()
            .into_array();
        let r = Like
            .try_new_array(fsst.len(), LikeOptions::default(), [decomp, pat.clone()])
            .unwrap()
            .into_array()
            .execute::<Canonical>(&mut ctx)
            .unwrap();
        times.push(t.elapsed());
        divan::black_box(r);
    }
    median_ms(&mut times)
}

mod regex_like {
    pub struct Pattern {
        re: regex::Regex,
    }
    impl Pattern {
        pub fn new(s: &str) -> Self {
            // Treat the input as a literal substring.
            Self {
                re: regex::Regex::new(&regex::escape(s)).unwrap(),
            }
        }
        pub fn is_match(&self, s: &str) -> bool {
            self.re.is_match(s)
        }
    }
}

fn main() {
    // Warm the static datasets.
    LazyLock::force(&STRINGS);
    LazyLock::force(&FSST);

    // Patterns from the user's requested table. Each row is:
    //   (label, literal-needle, like-pattern)
    // Where `literal-needle` is the substring the raw regex searches for,
    // and `like-pattern` is the LIKE string handed to FSST / Decomp+LIKE.
    let cases: &[(&str, &str, &str)] = &[
        ("ttp", "ttp", "%ttp%"),
        ("%htt%", "htt", "%htt%"),
        ("%http://%", "http://", "%http://%"),
        ("%rlane%", "rlane", "%rlane%"),
        ("%tor-sin%", "tor-sin", "%tor-sin%"),
    ];

    println!(
        "\nN={N} strings — synthetic `make_fsst_clickbench_urls` (sandbox blocks the real ClickBench download)"
    );
    println!(
        "\n┌───────────┬──────┬──────────┬─────────────┬──────────────┐\n\
          │  Pattern  │ Raw  │ FSST DFA │ Decomp+LIKE │ FSST speedup │\n\
          ├───────────┼──────┼──────────┼─────────────┼──────────────┤"
    );
    for (label, needle, like_pat) in cases {
        let re = regex_like::Pattern::new(needle);
        let raw_ms = time_raw(&re);
        let fsst_ms = time_fsst_dfa(like_pat);
        let decomp_ms = time_decompress_like(like_pat);
        let speedup = raw_ms / fsst_ms;
        println!(
            "│ {label:9} │ {raw_ms:>4.1} │ {fsst_ms:>8.1} │ {decomp_ms:>11.1} │ {speedup:>11.2}x │"
        );
        println!(
            "├───────────┼──────┼──────────┼─────────────┼──────────────┤"
        );
    }
}
