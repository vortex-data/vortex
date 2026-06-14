// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
//! LIKE-pushdown microbenchmarks for the OnPair Vortex array.
//!
//! Compares two ways of evaluating `LIKE 'prefix%'` / `LIKE '%needle%'` on an
//! OnPair-encoded URL column:
//!
//! * `pushdown` — the compressed-domain per-code DFA kernel
//!   (`<OnPair as LikeKernel>::like`), which never decompresses.
//! * `fallback` — what the engine does when no pushdown kernel handles the
//!   predicate: canonicalise to `VarBinViewArray` (decompress) and run the
//!   standard string LIKE.
//!
//! Both arms produce the same boolean result; the delta is the cost the
//! pushdown avoids.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_lossless,
    clippy::panic,
    clippy::tests_outside_test_module,
    clippy::unwrap_used,
    clippy::expect_used
)]

use std::sync::LazyLock;

use divan::Bencher;
use divan::black_box;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::scalar_fn::ScalarFnFactoryExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::scalar_fn::fns::like::Like;
use vortex_array::scalar_fn::fns::like::LikeKernel;
use vortex_array::scalar_fn::fns::like::LikeOptions;
use vortex_array::session::ArraySession;
use vortex_onpair::DEFAULT_DICT12_CONFIG;
use vortex_onpair::OnPair;
use vortex_onpair::OnPairArray;
use vortex_onpair::onpair_compress;
use vortex_session::VortexSession;

fn main() {
    divan::main();
}

const N_ROWS: usize = 200_000;

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

fn ctx() -> ExecutionCtx {
    SESSION.create_execution_ctx()
}

/// A ClickBench-shaped URL corpus: high lexical overlap, a handful of hosts and
/// path templates, so OnPair's learned dictionary applies.
static CORPUS: LazyLock<OnPairArray> = LazyLock::new(|| {
    let hosts = [
        "https://www.example.com",
        "http://ads.adriver.ru",
        "https://yandex.ru/search",
        "https://www.google.com/maps",
        "http://shop.bonprix.ru",
        "https://video.yandex.ru/users",
    ];
    let paths = [
        "/page/",
        "/product/",
        "/category/",
        "/search?text=",
        "/index?id=",
        "/cart/item/",
    ];
    let rows: Vec<String> = (0..N_ROWS)
        .map(|i| {
            let h = hosts[i % hosts.len()];
            let p = paths[(i / 7) % paths.len()];
            format!("{h}{p}{}", i % 9973)
        })
        .collect();
    let refs: Vec<Option<&str>> = rows.iter().map(|s| Some(s.as_str())).collect();
    let varbin = VarBinArray::from_iter(refs, DType::Utf8(Nullability::NonNullable));
    let len = varbin.len();
    let dtype = varbin.dtype().clone();
    onpair_compress(&varbin, len, &dtype, DEFAULT_DICT12_CONFIG).unwrap()
});

/// A small (4k-row) corpus with a full-sized dictionary, so the per-call DFA
/// table construction dominates — the dict-encoded ClickBench regime, where the
/// kernel scans only the deduplicated values.
static CORPUS_SMALL: LazyLock<OnPairArray> = LazyLock::new(|| {
    let rows: Vec<String> = (0..4000)
        .map(|i| {
            format!(
                "https://host{}.example.com/path/{}/item/{}?ref={}",
                i % 89,
                (i * 7) % 1000,
                (i * 13) % 5000,
                i
            )
        })
        .collect();
    let refs: Vec<Option<&str>> = rows.iter().map(|s| Some(s.as_str())).collect();
    let varbin = VarBinArray::from_iter(refs, DType::Utf8(Nullability::NonNullable));
    let len = varbin.len();
    let dtype = varbin.dtype().clone();
    onpair_compress(&varbin, len, &dtype, DEFAULT_DICT12_CONFIG).unwrap()
});

fn pattern_array(pattern: &str) -> ArrayRef {
    ConstantArray::new(pattern, CORPUS.len()).into_array()
}

/// Run the pushdown kernel on a given corpus (build + scan).
fn run_pushdown_on(corpus: &OnPairArray, pattern: &str) -> ArrayRef {
    let mut ctx = ctx();
    let p = ConstantArray::new(pattern, corpus.len()).into_array();
    <OnPair as LikeKernel>::like(corpus.as_view(), &p, LikeOptions::default(), &mut ctx)
        .unwrap()
        .expect("OnPair pushdown should handle this pattern")
}

/// Build-dominated: needles range from common bytes (many relevant codes) to
/// rare bytes (almost all codes skipped).
#[divan::bench(args = ["%google%", "%example%", "%zqxj%"])]
fn contains_pushdown_buildheavy(bencher: Bencher, pattern: &str) {
    bencher.bench(|| black_box(run_pushdown_on(&CORPUS_SMALL, pattern)));
}

/// Compressed-domain pushdown: the DFA kernel, no decompression.
fn run_pushdown(pattern: &ArrayRef) -> ArrayRef {
    let mut ctx = ctx();
    <OnPair as LikeKernel>::like(CORPUS.as_view(), pattern, LikeOptions::default(), &mut ctx)
        .unwrap()
        .expect("OnPair pushdown should handle this pattern")
}

/// Fallback: decompress to `VarBinViewArray`, then run the standard LIKE.
fn run_fallback(pattern: &ArrayRef) -> ArrayRef {
    let mut ctx = ctx();
    let canonical = CORPUS
        .clone()
        .into_array()
        .execute::<VarBinViewArray>(&mut ctx)
        .unwrap()
        .into_array();
    Like.try_new_array(
        CORPUS.len(),
        LikeOptions::default(),
        [canonical, pattern.clone()],
    )
    .unwrap()
    .into_array()
    .execute::<Canonical>(&mut ctx)
    .unwrap()
    .into_array()
}

#[divan::bench(args = ["https://www.example.com%", "https://www.google.com/maps%"])]
fn prefix_pushdown(bencher: Bencher, pattern: &str) {
    let p = pattern_array(pattern);
    bencher.bench(|| black_box(run_pushdown(&p)));
}

#[divan::bench(args = ["https://www.example.com%", "https://www.google.com/maps%"])]
fn prefix_fallback(bencher: Bencher, pattern: &str) {
    let p = pattern_array(pattern);
    bencher.bench(|| black_box(run_fallback(&p)));
}

#[divan::bench(args = ["%yandex%", "%/search?text=%", "%bonprix%"])]
fn contains_pushdown(bencher: Bencher, pattern: &str) {
    let p = pattern_array(pattern);
    bencher.bench(|| black_box(run_pushdown(&p)));
}

#[divan::bench(args = ["%yandex%", "%/search?text=%", "%bonprix%"])]
fn contains_fallback(bencher: Bencher, pattern: &str) {
    let p = pattern_array(pattern);
    bencher.bench(|| black_box(run_fallback(&p)));
}
