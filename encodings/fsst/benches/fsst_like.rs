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
    bencher.bench_local(|| {
        Like.try_new_array(len, LikeOptions::default(), [arr.clone(), pattern.clone()])
            .unwrap()
            .into_array()
            .execute::<Canonical>(&mut SESSION.create_execution_ctx())
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
