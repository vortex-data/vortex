// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Apples-to-apples comparison of the three FSST `%needle%` prefilter
//! strategies on the same `all_bytes` corpus. Drives them through
//! `FoldedContainsDfa` accessors exposed under the `_test-harness`
//! feature.
//!
//! ## Variants
//!
//! - `prefilter_one_byte`: 1-byte progressing-code bitset. Multi-pass
//!   PSHUFB-Mula OR-merge when the progressing set exceeds
//!   `MAX_SET_BYTES`. Always applicable.
//! - `prefilter_cartesian`: legacy 2-byte Cartesian pair bitset
//!   (`c1_union × c2_union`). Single pass; falls back to `None` when
//!   either union exceeds `MAX_SET_BYTES`.
//! - `prefilter_bucketed`: bucketed Cartesian Teddy — one (c1, c2_set)
//!   bucket per distinct c1, OR'd together. Single pass when
//!   `distinct_c1 ≤ MAX_SET_BYTES`, multi-pass otherwise. Eliminates
//!   cross-bucket false positives that the legacy Cartesian path
//!   admits.
//!
//! Each variant reports:
//! - Wall-clock time to build the bitset over the entire `all_bytes`
//!   buffer (~one-shot cost per chunk per LIKE call).
//! - The end-to-end `FsstMatcher::scan_to_bitbuf` call, which routes
//!   through the bucketed path in production; the per-variant
//!   "scan_with_..." benches drive the scan with the chosen bitset
//!   directly.
//!
//! ## How this bench was generated
//!
//! Generated 2026-05-11 alongside the bucketed Cartesian Teddy patch
//! (`encodings/fsst/src/dfa/anchor_scan.rs::build_bucketed_pair_bitset`).
//! There were no pre-existing prefilter-only benches to retire — the
//! adjacent `clickbench_url_google`, `clickbench_real_q20`,
//! `fsst_url_compare` benches measure full end-to-end LIKE evaluation
//! and pick up the bucketed path automatically.
//!
//! ## Reproduce
//!
//! ```sh
//! # Run with the new (bucketed) production code path.
//! cargo bench -p vortex-fsst --features _test-harness \
//!     --bench fsst_prefilter_compare
//!
//! # Same machine, same dataset (synthetic ClickBench URLs, 1M rows).
//! # Numbers reported in `prefilter_one_byte` vs `prefilter_bucketed`
//! # capture the bitset construction delta; `prefilter_*_scan`
//! # captures the end-to-end win including DFA verifier dispatches.
//! ```

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use vortex_array::arrays::varbin::VarBinArrayExt;
use vortex_fsst::FSSTArray;
use vortex_fsst::FSSTArrayExt;
use vortex_fsst::dfa::FsstMatcher;
use vortex_fsst::test_utils::make_fsst_clickbench_urls;

fn main() {
    divan::main();
}

const N: usize = 1_000_000;

/// `%google%` — the canonical ClickBench Q22 needle. On ClickBench-shape
/// URL data the FSST trainer typically keeps 'g' and 'o' as
/// single-byte symbols, so the bucketed/Cartesian paths see a single
/// `(c1='g', c2='o')` bucket and degenerate to plain Teddy.
const PATTERN_GOOGLE: &[u8] = b"%google%";

/// `%lazy dog%` — multi-segment-ish: more c1 choices when 'l' and the
/// space byte are independent symbols. Tends to exercise multi-bucket
/// behavior on real corpora.
const PATTERN_LAZY_DOG: &[u8] = b"%lazy dog%";

/// `%.ru%` — short, includes a punctuation byte and a vowel. Useful for
/// pattern shapes where the progressing set is small but the c2
/// advancing set spans several symbol codes.
const PATTERN_DOT_RU: &[u8] = b"%.ru%";

struct Corpus {
    all_bytes: Vec<u8>,
    offsets: Vec<u32>,
    fsst: FSSTArray,
}

static CORPUS: LazyLock<Corpus> = LazyLock::new(|| {
    let fsst = make_fsst_clickbench_urls(N);
    let view = fsst.as_view();
    let codes = view.codes();
    #[expect(deprecated)]
    use vortex_array::ToCanonical;
    let offsets = codes.offsets().to_primitive();
    let offsets: Vec<u32> = offsets.as_slice::<i32>().iter().map(|&v| v as u32).collect();
    let all_bytes = codes.bytes().as_slice().to_vec();
    Corpus { all_bytes, offsets, fsst }
});

/// Build a matcher for `pattern`. Panics if pushdown is unsupported.
fn build_matcher(pattern: &[u8]) -> FsstMatcher {
    let view = CORPUS.fsst.as_view();
    FsstMatcher::try_new(view.symbols().as_slice(), view.symbol_lengths().as_slice(), pattern)
        .unwrap()
        .unwrap()
}

// ---------------------------------------------------------------------------
// Per-variant bitset construction. Each closure rebuilds the bitset from
// scratch (matching the per-chunk cost paid in production).
// ---------------------------------------------------------------------------

fn bench_build_one_byte(bencher: Bencher, pattern: &[u8]) {
    let matcher = build_matcher(pattern);
    let dfa = matcher.as_folded().expect("folded contains DFA expected");
    let all_bytes = &CORPUS.all_bytes;
    bencher.bench_local(|| dfa.build_one_byte_bitset_for_bench(all_bytes).map(|b| b.len()));
}

fn bench_build_cartesian(bencher: Bencher, pattern: &[u8]) {
    let matcher = build_matcher(pattern);
    let dfa = matcher.as_folded().expect("folded contains DFA expected");
    let all_bytes = &CORPUS.all_bytes;
    bencher.bench_local(|| dfa.build_cartesian_bitset_for_bench(all_bytes).map(|b| b.len()));
}

fn bench_build_bucketed(bencher: Bencher, pattern: &[u8]) {
    let matcher = build_matcher(pattern);
    let dfa = matcher.as_folded().expect("folded contains DFA expected");
    let all_bytes = &CORPUS.all_bytes;
    bencher.bench_local(|| dfa.build_bucketed_bitset_for_bench(all_bytes).map(|b| b.len()));
}

// ---------------------------------------------------------------------------
// End-to-end scan_to_bitbuf using each variant's bitset. The production
// `scan_to_bitbuf` is hardcoded to bucketed; the cartesian/one-byte
// variants are reconstructed via the test-harness accessors.
// ---------------------------------------------------------------------------

#[divan::bench]
fn build_one_byte_google(b: Bencher) {
    bench_build_one_byte(b, PATTERN_GOOGLE);
}

#[divan::bench]
fn build_cartesian_google(b: Bencher) {
    bench_build_cartesian(b, PATTERN_GOOGLE);
}

#[divan::bench]
fn build_bucketed_google(b: Bencher) {
    bench_build_bucketed(b, PATTERN_GOOGLE);
}

#[divan::bench]
fn build_one_byte_lazy_dog(b: Bencher) {
    bench_build_one_byte(b, PATTERN_LAZY_DOG);
}

#[divan::bench]
fn build_cartesian_lazy_dog(b: Bencher) {
    bench_build_cartesian(b, PATTERN_LAZY_DOG);
}

#[divan::bench]
fn build_bucketed_lazy_dog(b: Bencher) {
    bench_build_bucketed(b, PATTERN_LAZY_DOG);
}

#[divan::bench]
fn build_one_byte_dot_ru(b: Bencher) {
    bench_build_one_byte(b, PATTERN_DOT_RU);
}

#[divan::bench]
fn build_cartesian_dot_ru(b: Bencher) {
    bench_build_cartesian(b, PATTERN_DOT_RU);
}

#[divan::bench]
fn build_bucketed_dot_ru(b: Bencher) {
    bench_build_bucketed(b, PATTERN_DOT_RU);
}

// ---------------------------------------------------------------------------
// End-to-end `scan_to_bitbuf` per pattern. Uses the production code
// path (bucketed). For a before/after comparison vs the legacy
// Cartesian path, run this bench at HEAD and again on the parent
// commit (`git stash`/`git checkout HEAD~`) — the parent uses
// `build_pair_bitset`.
// ---------------------------------------------------------------------------

/// End-to-end scan via the production code path (bucketed Teddy when
/// applicable, else 1-byte). The "after" measurement.
fn scan_after(b: Bencher, pattern: &[u8]) {
    let matcher = build_matcher(pattern);
    let corpus = &*CORPUS;
    b.bench_local(|| matcher.scan_to_bitbuf(N, &corpus.offsets, &corpus.all_bytes, false));
}

/// End-to-end scan forced through the 1-byte path. The "before"
/// measurement on patterns where the legacy Cartesian was
/// inapplicable (the historical case for `%google%` on ClickBench URLs).
fn scan_before_one_byte(b: Bencher, pattern: &[u8]) {
    let matcher = build_matcher(pattern);
    let dfa = matcher.as_folded().expect("folded contains DFA expected");
    let corpus = &*CORPUS;
    b.bench_local(|| dfa.scan_to_bitbuf_one_byte_only(N, &corpus.offsets, &corpus.all_bytes, false));
}

#[divan::bench]
fn scan_after_google(b: Bencher) { scan_after(b, PATTERN_GOOGLE); }
#[divan::bench]
fn scan_before_one_byte_google(b: Bencher) { scan_before_one_byte(b, PATTERN_GOOGLE); }

#[divan::bench]
fn scan_after_lazy_dog(b: Bencher) { scan_after(b, PATTERN_LAZY_DOG); }
#[divan::bench]
fn scan_before_one_byte_lazy_dog(b: Bencher) { scan_before_one_byte(b, PATTERN_LAZY_DOG); }

#[divan::bench]
fn scan_after_dot_ru(b: Bencher) { scan_after(b, PATTERN_DOT_RU); }
#[divan::bench]
fn scan_before_one_byte_dot_ru(b: Bencher) { scan_before_one_byte(b, PATTERN_DOT_RU); }

// ---------------------------------------------------------------------------
// One-shot selectivity report (printed at startup via divan::bench attr
// on a near-zero-cost closure that we just observe once). Useful for
// interpreting the bitset-construction timings: a 10× sparser bitset
// shows up as a ~10× drop in DFA verifier dispatches downstream.
// ---------------------------------------------------------------------------

fn report_selectivity(pattern: &[u8]) -> (Option<u64>, Option<u64>, Option<u64>) {
    let matcher = build_matcher(pattern);
    let dfa = matcher.as_folded().expect("folded contains DFA expected");
    let all_bytes = &CORPUS.all_bytes;
    let one_byte = dfa
        .build_one_byte_bitset_for_bench(all_bytes)
        .map(|b| b.iter().map(|w| w.count_ones() as u64).sum::<u64>());
    let cartesian = dfa
        .build_cartesian_bitset_for_bench(all_bytes)
        .map(|b| b.iter().map(|w| w.count_ones() as u64).sum::<u64>());
    let bucketed = dfa
        .build_bucketed_bitset_for_bench(all_bytes)
        .map(|b| b.iter().map(|w| w.count_ones() as u64).sum::<u64>());
    (one_byte, cartesian, bucketed)
}

/// One-shot selectivity report (printed once at bench startup). The
/// divan timing on this entry is meaningless — the value is the
/// `eprintln!` line, which captures the popcount of each prefilter
/// variant's bitset on the corpus.
#[divan::bench(sample_count = 1, sample_size = 1)]
fn aaa_selectivity_report(b: Bencher) {
    b.bench_local(|| {
        for (pat, name) in [
            (PATTERN_GOOGLE, "%google%"),
            (PATTERN_LAZY_DOG, "%lazy dog%"),
            (PATTERN_DOT_RU, "%.ru%"),
        ] {
            let (one, cart, buck) = report_selectivity(pat);
            eprintln!(
                "selectivity[{name}] all_bytes={} one_byte={:?} cartesian={:?} bucketed={:?}",
                CORPUS.all_bytes.len(),
                one,
                cart,
                buck,
            );
        }
    });
}
