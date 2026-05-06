// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Focused microbenchmark for ClickBench Q22:
//!
//! ```sql
//! SELECT COUNT(*) FROM hits WHERE "URL" LIKE '%google%';
//! ```
//!
//! Mirrors the actual ClickBench Q22 microbenchmark on synthetic ClickBench-style
//! URL data, exercising only the FSST `%needle%` DFA path. This benchmark
//! deliberately strips away scan / planning / count overhead so we can iterate
//! on the FSST contains DFA in isolation.

#![expect(clippy::unwrap_used)]

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
use vortex_fsst::test_utils::make_fsst_clickbench_urls;
use vortex_session::VortexSession;

fn main() {
    divan::main();
}

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

/// Larger than the shared `fsst_like` bench (100K) so we get stable numbers
/// for the inner-loop scan and so % overhead from setup is small.
const N: usize = 1_000_000;

const PATTERN: &str = "%google%";

static FSST_CB_URLS: LazyLock<FSSTArray> = LazyLock::new(|| make_fsst_clickbench_urls(N));

/// Full path: build the LIKE expression and execute it through the session.
/// This is the closest analogue to the real ClickBench query.
#[divan::bench]
fn like_google_full(bencher: Bencher) {
    let fsst = &*FSST_CB_URLS;
    let len = fsst.len();
    let arr = fsst.clone().into_array();
    let pattern = ConstantArray::new(PATTERN, len).into_array();
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
