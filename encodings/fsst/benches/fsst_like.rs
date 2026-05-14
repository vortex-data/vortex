// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]

use std::fmt;
use std::sync::LazyLock;

use divan::Bencher;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::scalar_fn::ScalarFnFactoryExt;
use vortex_array::scalar_fn::fns::like::Like;
use vortex_array::scalar_fn::fns::like::LikeOptions;
use vortex_array::session::ArraySession;
use vortex_fsst::FSSTArray;
use vortex_fsst::test_utils::NUM_STRINGS;
use vortex_fsst::test_utils::make_fsst_clickbench_urls;
use vortex_fsst::test_utils::make_fsst_emails;
use vortex_fsst::test_utils::make_fsst_file_paths;
use vortex_fsst::test_utils::make_fsst_json_strings;
use vortex_fsst::test_utils::make_fsst_log_lines;
use vortex_fsst::test_utils::make_fsst_rare_match;
use vortex_fsst::test_utils::make_fsst_short_urls;
use vortex_session::VortexSession;

fn main() {
    divan::main();
}

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

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

    fn prefix_pattern(&self) -> &'static str {
        match self {
            Self::Urls => "https%",
            Self::Cb => "https://www.%",
            Self::Log => "192.168%",
            Self::Json => r#"{"id%"#,
            Self::Path => "/home%",
            Self::Email => "john%",
            Self::Rare => "xyz%",
        }
    }

    fn contains_pattern(&self) -> &'static str {
        match self {
            Self::Urls => "%google%",
            Self::Cb => "%yandex%",
            Self::Log => "%Googlebot%",
            Self::Json => "%enterprise%",
            Self::Path => "%target/release%",
            Self::Email => "%gmail%",
            Self::Rare => "%xyzzy%",
        }
    }
}

fn bench_like(bencher: Bencher, fsst: &FSSTArray, pattern: &str) {
    let len = fsst.len();
    let arr = fsst.clone().into_array();
    let pattern = ConstantArray::new(pattern, len).into_array();
    bencher
        .with_inputs(|| SESSION.create_execution_ctx())
        .bench_refs(|ctx| {
            Like.try_new_array(len, LikeOptions::default(), [arr.clone(), pattern.clone()])
                .unwrap()
                .into_array()
                .execute::<Canonical>(ctx)
                .unwrap()
        });
}

fn bench_not_like(bencher: Bencher, fsst: &FSSTArray, pattern: &str) {
    let len = fsst.len();
    let arr = fsst.clone().into_array();
    let pattern = ConstantArray::new(pattern, len).into_array();
    let opts = LikeOptions {
        negated: true,
        case_insensitive: false,
    };
    bencher
        .with_inputs(|| SESSION.create_execution_ctx())
        .bench_refs(|ctx| {
            Like.try_new_array(len, opts, [arr.clone(), pattern.clone()])
                .unwrap()
                .into_array()
                .execute::<Canonical>(ctx)
                .unwrap()
        });
}

#[divan::bench(args = [
    Dataset::Urls, Dataset::Cb, Dataset::Log, Dataset::Json,
    Dataset::Path, Dataset::Email, Dataset::Rare,
])]
fn fsst_prefix(bencher: Bencher, dataset: &Dataset) {
    bench_like(bencher, dataset.fsst_array(), dataset.prefix_pattern());
}

#[divan::bench(args = [
    Dataset::Urls, Dataset::Cb, Dataset::Log, Dataset::Json,
    Dataset::Path, Dataset::Email, Dataset::Rare,
])]
fn fsst_contains(bencher: Bencher, dataset: &Dataset) {
    bench_like(bencher, dataset.fsst_array(), dataset.contains_pattern());
}

#[divan::bench]
fn fsst_contains_htt_urls(bencher: Bencher) {
    bench_like(bencher, &FSST_URLS, "%htt%");
}

#[divan::bench]
fn fsst_contains_htt_cb(bencher: Bencher) {
    bench_like(bencher, &FSST_CB_URLS, "%htt%");
}

#[divan::bench]
fn fsst_contains_ear_urls(bencher: Bencher) {
    bench_like(bencher, &FSST_URLS, "%ear%");
}

#[divan::bench]
fn fsst_contains_ear_cb(bencher: Bencher) {
    bench_like(bencher, &FSST_CB_URLS, "%ear%");
}

#[divan::bench]
fn fsst_contains_https_urls(bencher: Bencher) {
    bench_like(bencher, &FSST_URLS, "%https%");
}

#[divan::bench]
fn fsst_not_contains_google_urls(bencher: Bencher) {
    bench_not_like(bencher, &FSST_URLS, "%google%");
}

#[divan::bench]
fn fsst_not_contains_xyzzy_rare(bencher: Bencher) {
    bench_not_like(bencher, &FSST_RARE_MATCH, "%xyzzy%");
}

// Short-needle (≤ 4 byte) benches that exercise the Shift-Or matcher path
// added in dfa/shift_or.rs. On selective short needles, the bit-parallel
// `(state << shift) | or_mask` inner loop replaces the Teddy + verifier
// dispatch and wins ≥ 1.5×.

#[divan::bench]
fn fsst_contains_short_xy_urls(bencher: Bencher) {
    bench_like(bencher, &FSST_URLS, "%xy%");
}

#[divan::bench]
fn fsst_contains_short_zz_urls(bencher: Bencher) {
    bench_like(bencher, &FSST_URLS, "%zz%");
}

#[divan::bench]
fn fsst_contains_short_qq_urls(bencher: Bencher) {
    bench_like(bencher, &FSST_URLS, "%qq%");
}

#[divan::bench]
fn fsst_contains_short_zzz_urls(bencher: Bencher) {
    bench_like(bencher, &FSST_URLS, "%zzz%");
}

#[divan::bench]
fn fsst_contains_short_qq_cb(bencher: Bencher) {
    bench_like(bencher, &FSST_CB_URLS, "%qq%");
}

#[divan::bench]
fn fsst_contains_short_xyzz_rare(bencher: Bencher) {
    bench_like(bencher, &FSST_RARE_MATCH, "%xyzz%");
}

// ---------------------------------------------------------------------------
// Fat Teddy / multi-needle OR benches
//
// Each `fsst_contains_or_<n>_<dataset>` bench runs `MultiNeedleMatcher`
// (Fat Teddy single pass) on a small needle list, while the
// `fsst_contains_or_<n>_<dataset>_npass` baseline runs the same needles
// as N separate single-pattern `FsstMatcher` scans and bitwise-ORs the
// results. The Fat Teddy variant should be ≥ 1.5× faster than the
// N-pass baseline for n ≥ 4.
// ---------------------------------------------------------------------------

/// Run `MultiNeedleMatcher::scan_or_to_bitbuf` on `fsst` for the given
/// needle list, returning the OR bit-buffer.
fn fat_teddy_or(fsst: &FSSTArray, patterns: &[&str]) -> vortex_buffer::BitBuffer {
    #[expect(deprecated)]
    use vortex_array::ToCanonical;
    use vortex_array::arrays::varbin::VarBinArrayExt;
    use vortex_array::match_each_integer_ptype;
    use vortex_fsst::FSSTArrayExt;
    use vortex_fsst::dfa::MultiNeedleMatcher;

    let symbols = fsst.symbols();
    let symbol_lengths = fsst.symbol_lengths();
    let codes = fsst.codes();
    #[expect(deprecated)]
    let offsets = codes.offsets().to_primitive();
    let all_bytes = codes.bytes();
    let all_bytes = all_bytes.as_slice();
    let n = codes.len();
    let pattern_bytes: Vec<&[u8]> = patterns.iter().map(|s| s.as_bytes()).collect();
    let matcher = MultiNeedleMatcher::try_new_multi(
        symbols.as_slice(),
        symbol_lengths.as_slice(),
        &pattern_bytes,
        false,
    )
    .unwrap()
    .unwrap();
    match_each_integer_ptype!(offsets.ptype(), |T| {
        let off = offsets.as_slice::<T>();
        matcher.scan_or_to_bitbuf(n, off, all_bytes, false)
    })
}

/// Run N separate single-pattern `FsstMatcher` scans and OR-merge them.
fn npass_or(fsst: &FSSTArray, patterns: &[&str]) -> vortex_buffer::BitBuffer {
    #[expect(deprecated)]
    use vortex_array::ToCanonical;
    use vortex_array::arrays::varbin::VarBinArrayExt;
    use vortex_array::match_each_integer_ptype;
    use vortex_fsst::FSSTArrayExt;
    use vortex_fsst::dfa::FsstMatcher;

    let symbols = fsst.symbols();
    let symbol_lengths = fsst.symbol_lengths();
    let codes = fsst.codes();
    #[expect(deprecated)]
    let offsets = codes.offsets().to_primitive();
    let all_bytes = codes.bytes();
    let all_bytes = all_bytes.as_slice();
    let n = codes.len();
    let mut acc: Option<vortex_buffer::BitBuffer> = None;
    for p in patterns {
        let m = FsstMatcher::try_new_with(
            symbols.as_slice(),
            symbol_lengths.as_slice(),
            p.as_bytes(),
            false,
        )
        .unwrap()
        .unwrap();
        let r = match_each_integer_ptype!(offsets.ptype(), |T| {
            let off = offsets.as_slice::<T>();
            m.scan_to_bitbuf(n, off, all_bytes, false)
        });
        acc = Some(match acc {
            Some(prev) => &prev | &r,
            None => r,
        });
    }
    acc.unwrap()
}

static NEEDLES_OR_3_URLS: &[&str] = &["%google%", "%yandex%", "%bing%"];
static NEEDLES_OR_8_URLS: &[&str] = &[
    "%google%", "%yandex%", "%bing%", "%duck%", "%wiki%", "%news%", "%blog%", "%shop%",
];
static NEEDLES_OR_16_URLS: &[&str] = &[
    "%google%", "%yandex%", "%bing%", "%duck%", "%wiki%", "%news%", "%blog%", "%shop%", "%com%",
    "%org%", "%net%", "%info%", "%http%", "%https%", "%www%", "%api%",
];

#[divan::bench]
fn fsst_contains_or_3_urls(bencher: Bencher) {
    bencher.bench(|| fat_teddy_or(&FSST_CB_URLS, NEEDLES_OR_3_URLS));
}

#[divan::bench]
fn fsst_contains_or_3_urls_npass(bencher: Bencher) {
    bencher.bench(|| npass_or(&FSST_CB_URLS, NEEDLES_OR_3_URLS));
}

#[divan::bench]
fn fsst_contains_or_8_urls(bencher: Bencher) {
    bencher.bench(|| fat_teddy_or(&FSST_CB_URLS, NEEDLES_OR_8_URLS));
}

#[divan::bench]
fn fsst_contains_or_8_urls_npass(bencher: Bencher) {
    bencher.bench(|| npass_or(&FSST_CB_URLS, NEEDLES_OR_8_URLS));
}

#[divan::bench]
fn fsst_contains_or_16_urls(bencher: Bencher) {
    bencher.bench(|| fat_teddy_or(&FSST_CB_URLS, NEEDLES_OR_16_URLS));
}

#[divan::bench]
fn fsst_contains_or_16_urls_npass(bencher: Bencher) {
    bencher.bench(|| npass_or(&FSST_CB_URLS, NEEDLES_OR_16_URLS));
}
