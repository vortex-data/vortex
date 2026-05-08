// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// Binary to measure shufti skip fire rates across all benchmark datasets.
// Build and run with:
//   cargo run -p vortex-fsst --bin shufti_skip_report \
//     --features "_test-harness,shufti-counters" --release

#![expect(clippy::unwrap_used)]

use vortex_fsst::bench_utils::{
    read_shufti_counters, reset_shufti_counters, scan_shufti_contains,
};
use vortex_fsst::test_utils::NUM_STRINGS as N;
use vortex_fsst::test_utils::{
    make_fsst_clickbench_urls, make_fsst_emails, make_fsst_file_paths, make_fsst_json_strings,
    make_fsst_log_lines, make_fsst_rare_match, make_fsst_short_urls,
};

fn report(label: &str, needle: &[u8], fsst: &vortex_fsst::FSSTArray) {
    reset_shufti_counters();
    let matches = scan_shufti_contains(fsst, needle);
    let (calls, fired, skipped) = read_shufti_counters();
    let fire_rate = if calls > 0 {
        fired as f64 / calls as f64 * 100.0
    } else {
        0.0
    };
    let avg_skip = if fired > 0 { skipped as f64 / fired as f64 } else { 0.0 };
    println!(
        "{label:<25}  needle={:<15}  matches={matches:>6}  \
         skip_calls={calls:>8}  fired={fired:>8} ({fire_rate:5.1}%)  avg_skip={avg_skip:.2}",
        String::from_utf8_lossy(needle),
    );
}

fn main() {
    println!("Shufti skip-fire report ({N} strings each)");
    println!("{:-<110}", "");
    println!(
        "{:<25}  {:<20}  {:<12}  {:<30}  {}",
        "dataset", "needle", "matches", "skip_calls / fired (%)", "avg_skip"
    );
    println!("{:-<110}", "");

    let urls = make_fsst_short_urls(N);
    report("short_urls", b"google", &urls);

    let cb = make_fsst_clickbench_urls(N);
    report("clickbench_urls", b"yandex", &cb);

    let log = make_fsst_log_lines(N);
    report("log_lines", b"Googlebot", &log);

    let json = make_fsst_json_strings(N);
    report("json_strings", b"enterprise", &json);

    let paths = make_fsst_file_paths(N);
    report("file_paths", b"target/release", &paths);

    let emails = make_fsst_emails(N);
    report("emails", b"gmail", &emails);

    let rare = make_fsst_rare_match(N);
    report("rare_match", b"xyzzy", &rare);
}
