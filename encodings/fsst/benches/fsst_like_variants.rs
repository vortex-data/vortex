// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Side-by-side divan benchmarks: `FlatContainsDfaBaseline` vs shufti `FlatContainsDfa`.
//!
//! Each benchmark calls the matcher directly via `bench_utils`, bypassing the
//! LikeKernel framework to measure only DFA scanning cost.
//!
//! Run with:
//! ```
//! cargo bench -p vortex-fsst --bench fsst_like_variants --features _test-harness \
//!     -- --sample-count 4 --sample-size 200
//! ```

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use vortex_fsst::FSSTArray;
use vortex_fsst::bench_utils::scan_baseline_contains;
use vortex_fsst::bench_utils::scan_classes_contains;
use vortex_fsst::bench_utils::scan_pre_classified_contains;
use vortex_fsst::bench_utils::scan_shufti_contains;
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

// Needles match the contains patterns from fsst_like.rs for a fair comparison.
const NEEDLE_URLS: &[u8] = b"google";
const NEEDLE_CB_URLS: &[u8] = b"yandex";
const NEEDLE_LOG: &[u8] = b"Googlebot";
const NEEDLE_JSON: &[u8] = b"enterprise";
const NEEDLE_PATH: &[u8] = b"target/release";
const NEEDLE_EMAIL: &[u8] = b"gmail";
const NEEDLE_RARE: &[u8] = b"xyzzy";

// ─── baseline (state-0 skip only) ─────────────────────────────────────────

#[divan::bench]
fn baseline_contains_urls(bencher: Bencher) {
    bencher.bench(|| scan_baseline_contains(&FSST_URLS, NEEDLE_URLS));
}

#[divan::bench]
fn baseline_contains_cb(bencher: Bencher) {
    bencher.bench(|| scan_baseline_contains(&FSST_CB_URLS, NEEDLE_CB_URLS));
}

#[divan::bench]
fn baseline_contains_log(bencher: Bencher) {
    bencher.bench(|| scan_baseline_contains(&FSST_LOG_LINES, NEEDLE_LOG));
}

#[divan::bench]
fn baseline_contains_json(bencher: Bencher) {
    bencher.bench(|| scan_baseline_contains(&FSST_JSON_STRINGS, NEEDLE_JSON));
}

#[divan::bench]
fn baseline_contains_path(bencher: Bencher) {
    bencher.bench(|| scan_baseline_contains(&FSST_FILE_PATHS, NEEDLE_PATH));
}

#[divan::bench]
fn baseline_contains_email(bencher: Bencher) {
    bencher.bench(|| scan_baseline_contains(&FSST_EMAILS, NEEDLE_EMAIL));
}

#[divan::bench]
fn baseline_contains_rare(bencher: Bencher) {
    bencher.bench(|| scan_baseline_contains(&FSST_RARE_MATCH, NEEDLE_RARE));
}

// ─── variant B: byte-class minimization ────────────────────────────────────

#[divan::bench]
fn classes_contains_urls(bencher: Bencher) {
    bencher.bench(|| scan_classes_contains(&FSST_URLS, NEEDLE_URLS));
}

#[divan::bench]
fn classes_contains_cb(bencher: Bencher) {
    bencher.bench(|| scan_classes_contains(&FSST_CB_URLS, NEEDLE_CB_URLS));
}

#[divan::bench]
fn classes_contains_log(bencher: Bencher) {
    bencher.bench(|| scan_classes_contains(&FSST_LOG_LINES, NEEDLE_LOG));
}

#[divan::bench]
fn classes_contains_json(bencher: Bencher) {
    bencher.bench(|| scan_classes_contains(&FSST_JSON_STRINGS, NEEDLE_JSON));
}

#[divan::bench]
fn classes_contains_path(bencher: Bencher) {
    bencher.bench(|| scan_classes_contains(&FSST_FILE_PATHS, NEEDLE_PATH));
}

#[divan::bench]
fn classes_contains_email(bencher: Bencher) {
    bencher.bench(|| scan_classes_contains(&FSST_EMAILS, NEEDLE_EMAIL));
}

#[divan::bench]
fn classes_contains_rare(bencher: Bencher) {
    bencher.bench(|| scan_classes_contains(&FSST_RARE_MATCH, NEEDLE_RARE));
}

// ─── variant C: byte-class minimization + bulk pre-classify ────────────────

#[divan::bench]
fn pre_contains_urls(bencher: Bencher) {
    bencher.bench(|| scan_pre_classified_contains(&FSST_URLS, NEEDLE_URLS));
}

#[divan::bench]
fn pre_contains_cb(bencher: Bencher) {
    bencher.bench(|| scan_pre_classified_contains(&FSST_CB_URLS, NEEDLE_CB_URLS));
}

#[divan::bench]
fn pre_contains_log(bencher: Bencher) {
    bencher.bench(|| scan_pre_classified_contains(&FSST_LOG_LINES, NEEDLE_LOG));
}

#[divan::bench]
fn pre_contains_json(bencher: Bencher) {
    bencher.bench(|| scan_pre_classified_contains(&FSST_JSON_STRINGS, NEEDLE_JSON));
}

#[divan::bench]
fn pre_contains_path(bencher: Bencher) {
    bencher.bench(|| scan_pre_classified_contains(&FSST_FILE_PATHS, NEEDLE_PATH));
}

#[divan::bench]
fn pre_contains_email(bencher: Bencher) {
    bencher.bench(|| scan_pre_classified_contains(&FSST_EMAILS, NEEDLE_EMAIL));
}

#[divan::bench]
fn pre_contains_rare(bencher: Bencher) {
    bencher.bench(|| scan_pre_classified_contains(&FSST_RARE_MATCH, NEEDLE_RARE));
}

// ─── shufti (per-state skip) ───────────────────────────────────────────────

#[divan::bench]
fn shufti_contains_urls(bencher: Bencher) {
    bencher.bench(|| scan_shufti_contains(&FSST_URLS, NEEDLE_URLS));
}

#[divan::bench]
fn shufti_contains_cb(bencher: Bencher) {
    bencher.bench(|| scan_shufti_contains(&FSST_CB_URLS, NEEDLE_CB_URLS));
}

#[divan::bench]
fn shufti_contains_log(bencher: Bencher) {
    bencher.bench(|| scan_shufti_contains(&FSST_LOG_LINES, NEEDLE_LOG));
}

#[divan::bench]
fn shufti_contains_json(bencher: Bencher) {
    bencher.bench(|| scan_shufti_contains(&FSST_JSON_STRINGS, NEEDLE_JSON));
}

#[divan::bench]
fn shufti_contains_path(bencher: Bencher) {
    bencher.bench(|| scan_shufti_contains(&FSST_FILE_PATHS, NEEDLE_PATH));
}

#[divan::bench]
fn shufti_contains_email(bencher: Bencher) {
    bencher.bench(|| scan_shufti_contains(&FSST_EMAILS, NEEDLE_EMAIL));
}

#[divan::bench]
fn shufti_contains_rare(bencher: Bencher) {
    bencher.bench(|| scan_shufti_contains(&FSST_RARE_MATCH, NEEDLE_RARE));
}
