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
//!
//! ## Variants
//!
//! - `like_google_full`: end-to-end `LIKE` expression evaluation. The closest
//!   analogue to the real ClickBench query. Includes per-string call dispatch,
//!   `BitBuffer` packing, and matcher construction — i.e., everything the real
//!   query executor pays.
//! - `dfa_inner_only`: per-string `FsstMatcher::matches` call, accumulating a
//!   count. Strips the `BitBuffer::collect_bool` closure indirection, so the
//!   delta vs `like_google_full` is dispatch + bit packing overhead.
//! - `memmem_per_string`: decompress each FSST string into a stack scratch
//!   buffer, then `memchr::memmem::Finder::find`. Apples-to-apples with
//!   `dfa_inner_only` (same per-string call shape, same accumulating-count).
//! - `memmem_concat_corpus`: decompress the entire corpus once into one big
//!   buffer, then run a single `memmem` over it. Theoretical floor for the
//!   matching work alone, with zero per-string overhead.

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use memchr::memmem::Finder;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::scalar_fn::ScalarFnFactoryExt;
use vortex_array::scalar_fn::fns::like::Like;
use vortex_array::scalar_fn::fns::like::LikeOptions;
use vortex_array::session::ArraySession;
use vortex_fsst::FSSTArray;
use vortex_fsst::FSSTArrayExt;
use vortex_fsst::dfa::FsstMatcher;
use vortex_fsst::test_utils::generate_clickbench_urls;
use vortex_fsst::test_utils::make_fsst_clickbench_urls;
use vortex_session::VortexSession;

fn main() {
    divan::main();
}

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

const N: usize = 1_000_000;

const PATTERN: &str = "%google%";
const NEEDLE: &[u8] = b"google";
const LIKE_PATTERN_BYTES: &[u8] = b"%google%";

static FSST_CB_URLS: LazyLock<FSSTArray> = LazyLock::new(|| make_fsst_clickbench_urls(N));

/// Decompressed corpus, one `Vec<u8>` per URL, in the same order as the FSST array.
static DECOMPRESSED_PER_STRING: LazyLock<Vec<Vec<u8>>> = LazyLock::new(|| {
    generate_clickbench_urls(N)
        .into_iter()
        .map(String::into_bytes)
        .collect()
});

/// Decompressed corpus concatenated into one buffer (no separators) plus the
/// per-string offsets, for the global `memmem` variant.
static DECOMPRESSED_CONCAT: LazyLock<Vec<u8>> = LazyLock::new(|| {
    let mut total = 0usize;
    for s in DECOMPRESSED_PER_STRING.iter() {
        total += s.len();
    }
    let mut out = Vec::with_capacity(total);
    for s in DECOMPRESSED_PER_STRING.iter() {
        out.extend_from_slice(s);
    }
    out
});

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

/// DFA inner loop only: call the FSST matcher per string and count hits.
/// No bitbuf, no expression dispatch. Isolates the FSST DFA matching work.
///
/// We reach into the codes layout the same way the real `like` kernel does, so
/// the per-string slicing cost is included.
#[divan::bench]
fn dfa_inner_only(bencher: Bencher) {
    #[expect(deprecated)]
    use vortex_array::ToCanonical;
    use vortex_array::arrays::varbin::VarBinArrayExt;

    let fsst = &*FSST_CB_URLS;
    let view = fsst.as_view();
    let codes = view.codes();
    #[expect(deprecated)]
    let offsets = codes.offsets().to_primitive();
    let offsets: Vec<u32> = offsets
        .as_slice::<i32>()
        .iter()
        .map(|&v| v as u32)
        .collect();
    let bytes = codes.bytes().as_slice().to_vec();

    let matcher = FsstMatcher::try_new(
        view.symbols().as_slice(),
        view.symbol_lengths().as_slice(),
        LIKE_PATTERN_BYTES,
    )
    .unwrap()
    .unwrap();

    bencher.bench_local(|| {
        let mut count: u64 = 0;
        let mut start = offsets[0] as usize;
        for i in 0..N {
            let end = offsets[i + 1] as usize;
            if matcher.matches(&bytes[start..end]) {
                count += 1;
            }
            start = end;
        }
        count
    });
}

/// memmem against per-string decompressed bytes. Same per-string call shape as
/// `dfa_inner_only`, but on plain UTF-8 text. Reuses one `Finder`.
#[divan::bench]
fn memmem_per_string(bencher: Bencher) {
    let strings = &*DECOMPRESSED_PER_STRING;
    let finder = Finder::new(NEEDLE);
    bencher.bench_local(|| {
        let mut count: u64 = 0;
        for s in strings.iter() {
            if finder.find(s).is_some() {
                count += 1;
            }
        }
        count
    });
}

/// One global memmem over the entire concatenated decompressed corpus.
/// No per-string overhead — theoretical floor for the matching work alone.
/// (Counts total occurrences, not strings; that's fine for floor measurement.)
#[divan::bench]
fn memmem_concat_corpus(bencher: Bencher) {
    let bytes = &*DECOMPRESSED_CONCAT;
    let finder = Finder::new(NEEDLE);
    bencher.bench_local(|| finder.find_iter(bytes).count() as u64);
}
