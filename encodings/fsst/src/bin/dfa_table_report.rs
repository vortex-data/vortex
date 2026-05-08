// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// Binary to characterize DFA transition table sizes across the bench datasets.
// For each dataset, prints baseline state count, byte-class count after
// minimization, and the resulting table-size compression ratio.
//
//   cargo run -p vortex-fsst --bin dfa_table_report --features _test-harness --release

#![expect(clippy::unwrap_used)]

use vortex_fsst::FSSTArray;
use vortex_fsst::bench_utils::classes_n_classes;
use vortex_fsst::test_utils::NUM_STRINGS as N;
use vortex_fsst::test_utils::{
    make_fsst_clickbench_urls, make_fsst_emails, make_fsst_file_paths, make_fsst_json_strings,
    make_fsst_log_lines, make_fsst_rare_match, make_fsst_short_urls,
};

fn report(label: &str, needle: &[u8], fsst: &FSSTArray) {
    let n_states = needle.len() + 1; // contains DFA: progress states + accept
    let n_classes = classes_n_classes(fsst, needle);
    let baseline_bytes = n_states * 256;
    let class_bytes = n_states * usize::from(n_classes);
    let ratio = baseline_bytes as f64 / class_bytes as f64;
    println!(
        "{label:<20}  needle={:<15}  states={n_states:>3}  \
         classes={n_classes:>3}  baseline={baseline_bytes:>5}B  \
         compressed={class_bytes:>5}B  ratio={ratio:.1}x",
        String::from_utf8_lossy(needle),
    );
}

fn main() {
    println!("DFA byte-class minimization table report (N={N} strings each)");
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
