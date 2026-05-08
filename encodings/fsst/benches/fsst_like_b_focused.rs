// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Focused bench: baseline vs. variant B (byte-class minimization) only.
//!
//! Avoids binary-layout interference from variants A/C/D so we can attribute
//! deltas to the algorithmic change rather than rustc inlining decisions.

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use vortex_fsst::FSSTArray;
use vortex_fsst::bench_utils::scan_baseline_contains;
use vortex_fsst::bench_utils::scan_classes_contains;
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

const NEEDLE_URLS: &[u8] = b"google";
const NEEDLE_CB_URLS: &[u8] = b"yandex";
const NEEDLE_LOG: &[u8] = b"Googlebot";
const NEEDLE_JSON: &[u8] = b"enterprise";
const NEEDLE_PATH: &[u8] = b"target/release";
const NEEDLE_EMAIL: &[u8] = b"gmail";
const NEEDLE_RARE: &[u8] = b"xyzzy";

#[divan::bench]
fn baseline_urls(bencher: Bencher) {
    bencher.bench(|| scan_baseline_contains(&FSST_URLS, NEEDLE_URLS));
}
#[divan::bench]
fn baseline_cb(bencher: Bencher) {
    bencher.bench(|| scan_baseline_contains(&FSST_CB_URLS, NEEDLE_CB_URLS));
}
#[divan::bench]
fn baseline_log(bencher: Bencher) {
    bencher.bench(|| scan_baseline_contains(&FSST_LOG_LINES, NEEDLE_LOG));
}
#[divan::bench]
fn baseline_json(bencher: Bencher) {
    bencher.bench(|| scan_baseline_contains(&FSST_JSON_STRINGS, NEEDLE_JSON));
}
#[divan::bench]
fn baseline_path(bencher: Bencher) {
    bencher.bench(|| scan_baseline_contains(&FSST_FILE_PATHS, NEEDLE_PATH));
}
#[divan::bench]
fn baseline_email(bencher: Bencher) {
    bencher.bench(|| scan_baseline_contains(&FSST_EMAILS, NEEDLE_EMAIL));
}
#[divan::bench]
fn baseline_rare(bencher: Bencher) {
    bencher.bench(|| scan_baseline_contains(&FSST_RARE_MATCH, NEEDLE_RARE));
}

#[divan::bench]
fn classes_urls(bencher: Bencher) {
    bencher.bench(|| scan_classes_contains(&FSST_URLS, NEEDLE_URLS));
}
#[divan::bench]
fn classes_cb(bencher: Bencher) {
    bencher.bench(|| scan_classes_contains(&FSST_CB_URLS, NEEDLE_CB_URLS));
}
#[divan::bench]
fn classes_log(bencher: Bencher) {
    bencher.bench(|| scan_classes_contains(&FSST_LOG_LINES, NEEDLE_LOG));
}
#[divan::bench]
fn classes_json(bencher: Bencher) {
    bencher.bench(|| scan_classes_contains(&FSST_JSON_STRINGS, NEEDLE_JSON));
}
#[divan::bench]
fn classes_path(bencher: Bencher) {
    bencher.bench(|| scan_classes_contains(&FSST_FILE_PATHS, NEEDLE_PATH));
}
#[divan::bench]
fn classes_email(bencher: Bencher) {
    bencher.bench(|| scan_classes_contains(&FSST_EMAILS, NEEDLE_EMAIL));
}
#[divan::bench]
fn classes_rare(bencher: Bencher) {
    bencher.bench(|| scan_classes_contains(&FSST_RARE_MATCH, NEEDLE_RARE));
}

// ─── long-needle stress (table size matters for L1 fit) ────────────────────
//
// For a 200-byte needle, baseline table = 201 * 256 ≈ 50 KiB (overflows the
// typical 32 KiB L1d), B's table = 201 * ~20 classes ≈ 4 KiB (fits).

const LONG_NEEDLE_64: &[u8] = b"the quick brown fox jumps over the lazy dog and runs through fields";
const LONG_NEEDLE_200: &[u8] =
    b"abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnop\
      qrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwx0123456789";

#[divan::bench]
fn baseline_long64_log(bencher: Bencher) {
    bencher.bench(|| scan_baseline_contains(&FSST_LOG_LINES, LONG_NEEDLE_64));
}
#[divan::bench]
fn classes_long64_log(bencher: Bencher) {
    bencher.bench(|| scan_classes_contains(&FSST_LOG_LINES, LONG_NEEDLE_64));
}
#[divan::bench]
fn baseline_long200_log(bencher: Bencher) {
    bencher.bench(|| scan_baseline_contains(&FSST_LOG_LINES, LONG_NEEDLE_200));
}
#[divan::bench]
fn classes_long200_log(bencher: Bencher) {
    bencher.bench(|| scan_classes_contains(&FSST_LOG_LINES, LONG_NEEDLE_200));
}
#[divan::bench]
fn baseline_long64_json(bencher: Bencher) {
    bencher.bench(|| scan_baseline_contains(&FSST_JSON_STRINGS, LONG_NEEDLE_64));
}
#[divan::bench]
fn classes_long64_json(bencher: Bencher) {
    bencher.bench(|| scan_classes_contains(&FSST_JSON_STRINGS, LONG_NEEDLE_64));
}
