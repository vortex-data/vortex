// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
//! `OnPair` (compact-on-every-op) vs `OnPairView` (metadata-only) for
//! `filter`/`take`, with the cost split into the **ops** phase and the
//! **canonicalise-to-`VarBinViewArray`** phase.
//!
//! Two pipelines, same logical result:
//!
//! 1. **OnPair**: each `filter`/`take` rebuilds (compacts) the surviving `codes`
//!    token stream into a new `OnPairArray`; finally canonicalise to VarBinView.
//! 2. **OnPairView**: convert once, then each `filter`/`take` only rewrites the
//!    per-row `offsets`/`sizes` (the shared `codes` buffer is untouched); finally
//!    canonicalise to VarBinView, compacting the live windows there iff needed.
//!
//! Kernels are invoked directly (`<OnPair as FilterKernel>::filter`,
//! `onpair_take_compact`, `<OnPairView as FilterKernel>::filter`,
//! `<OnPairView as TakeExecute>::take`) rather than through a query plan.
//!
//! Inputs are ~2 MB uncompressed: one corpus of many short strings, one of
//! fewer long strings.

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
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::dict::TakeExecute;
use vortex_array::arrays::filter::FilterKernel;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::session::ArraySession;
use vortex_buffer::Buffer;
use vortex_mask::Mask;
use vortex_onpair::DEFAULT_DICT12_CONFIG;
use vortex_onpair::OnPair;
use vortex_onpair::OnPairArray;
use vortex_onpair::OnPairView;
use vortex_onpair::OnPairViewArray;
use vortex_onpair::OnPairViewDecodeMode;
use vortex_onpair::canonicalize_with;
use vortex_onpair::onpair_compress;
use vortex_onpair::onpair_take_compact;
use vortex_session::VortexSession;

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

// ─── Corpora (≈ 2 MB uncompressed each) ──────────────────────────────────

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Corpus {
    /// Many short strings (~11 bytes/row).
    ManyShort,
    /// Fewer long strings (~115 bytes/row).
    FewLong,
}

const TARGET_BYTES: usize = 2 * 1024 * 1024;

fn corpus(c: Corpus) -> Vec<String> {
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state
    };
    let mut out = Vec::new();
    let mut bytes = 0usize;
    match c {
        Corpus::ManyShort => {
            let words: &[&str] = &[
                "alpha", "beta", "gamma", "delta", "eps", "zeta", "eta", "theta",
            ];
            while bytes < TARGET_BYTES {
                let s = words[(next() as usize) % words.len()].to_string();
                bytes += s.len();
                out.push(s);
            }
        }
        Corpus::FewLong => {
            let templates: &[&str] = &[
                "2026-05-14T12:34:56.789012Z INFO  request_id={id} method=GET path=/api/v1/users/{id}/profile status=200",
                "2026-05-14T12:34:56.789012Z WARN  request_id={id} method=POST path=/api/v1/users/{id}/sessions status=429",
                "2026-05-14T12:34:56.789012Z ERROR request_id={id} method=PUT  path=/api/v1/users/{id}/settings status=500",
            ];
            while bytes < TARGET_BYTES {
                let s = next();
                let pick = (s as usize) % templates.len();
                let line = templates[pick].replace("{id}", &format!("{:08x}", s as u32));
                bytes += line.len();
                out.push(line);
            }
        }
    }
    out
}

fn compress(c: Corpus) -> OnPairArray {
    let strings = corpus(c);
    let varbin = VarBinArray::from_iter(
        strings.iter().map(|s| Some(s.as_bytes())),
        DType::Utf8(Nullability::NonNullable),
    );
    onpair_compress(&varbin, varbin.len(), varbin.dtype(), DEFAULT_DICT12_CONFIG)
        .unwrap_or_else(|e| panic!("compress: {e}"))
}

struct Base {
    onpair: OnPairArray,
    view: OnPairViewArray,
}

fn base(c: Corpus) -> &'static Base {
    static MANY_SHORT: LazyLock<Base> = LazyLock::new(|| make(Corpus::ManyShort));
    static FEW_LONG: LazyLock<Base> = LazyLock::new(|| make(Corpus::FewLong));
    fn make(c: Corpus) -> Base {
        let onpair = compress(c);
        let mut ctx = SESSION.create_execution_ctx();
        let view = OnPairView::from_onpair(&onpair, &mut ctx).expect("from_onpair");
        Base { onpair, view }
    }
    match c {
        Corpus::ManyShort => &MANY_SHORT,
        Corpus::FewLong => &FEW_LONG,
    }
}

// ─── Scenarios ────────────────────────────────────────────────────────────

#[derive(Copy, Clone, Debug)]
enum Scenario {
    /// Keep ~1 % of rows.
    FilterSelective,
    /// Keep ~90 % of rows.
    FilterNonSelective,
    /// Take a full permutation (all rows, reordered).
    TakeShuffle,
    /// Take ~1 % of rows at random.
    TakeSelective,
    /// Take all rows in order.
    TakeNonSelective,
    /// Keep ~90 % then take ~1 % — a scan followed by a gather.
    FilterThenTake,
}

fn mask_selective(n: usize) -> Mask {
    Mask::from_iter((0..n).map(|i| i % 100 == 0))
}

fn mask_nonselective(n: usize) -> Mask {
    Mask::from_iter((0..n).map(|i| i % 10 != 0))
}

fn lcg(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *seed
}

fn shuffle_indices(n: usize) -> ArrayRef {
    let mut idx: Vec<u64> = (0..n as u64).collect();
    let mut seed = 0x1234_5678_9abc_def0_u64;
    for i in (1..n).rev() {
        let j = (lcg(&mut seed) as usize) % (i + 1);
        idx.swap(i, j);
    }
    Buffer::from(idx).into_array()
}

fn selective_indices(n: usize) -> ArrayRef {
    let count = (n / 100).max(1);
    let mut seed = 0x0bad_c0de_dead_beef_u64;
    let idx: Vec<u64> = (0..count)
        .map(|_| (lcg(&mut seed) as usize % n) as u64)
        .collect();
    Buffer::from(idx).into_array()
}

fn identity_indices(n: usize) -> ArrayRef {
    Buffer::from((0..n as u64).collect::<Vec<_>>()).into_array()
}

fn onpair_filter(
    arr: &OnPairArray,
    mask: &Mask,
    ctx: &mut vortex_array::ExecutionCtx,
) -> OnPairArray {
    <OnPair as FilterKernel>::filter(arr.as_view(), mask, ctx)
        .unwrap()
        .unwrap()
        .try_downcast::<OnPair>()
        .ok()
        .unwrap()
}

fn view_filter(
    arr: &OnPairViewArray,
    mask: &Mask,
    ctx: &mut vortex_array::ExecutionCtx,
) -> OnPairViewArray {
    <OnPairView as FilterKernel>::filter(arr.as_view(), mask, ctx)
        .unwrap()
        .unwrap()
        .try_downcast::<OnPairView>()
        .ok()
        .unwrap()
}

fn view_take(
    arr: &OnPairViewArray,
    indices: &ArrayRef,
    ctx: &mut vortex_array::ExecutionCtx,
) -> OnPairViewArray {
    <OnPairView as TakeExecute>::take(arr.as_view(), indices, ctx)
        .unwrap()
        .unwrap()
        .try_downcast::<OnPairView>()
        .ok()
        .unwrap()
}

/// Apply the scenario in `OnPair` space (compacting on every op).
fn run_onpair(
    base: &OnPairArray,
    scenario: Scenario,
    ctx: &mut vortex_array::ExecutionCtx,
) -> OnPairArray {
    let n = base.len();
    match scenario {
        Scenario::FilterSelective => onpair_filter(base, &mask_selective(n), ctx),
        Scenario::FilterNonSelective => onpair_filter(base, &mask_nonselective(n), ctx),
        Scenario::TakeShuffle => onpair_take_compact(base, &shuffle_indices(n), ctx).unwrap(),
        Scenario::TakeSelective => onpair_take_compact(base, &selective_indices(n), ctx).unwrap(),
        Scenario::TakeNonSelective => onpair_take_compact(base, &identity_indices(n), ctx).unwrap(),
        Scenario::FilterThenTake => {
            let filtered = onpair_filter(base, &mask_nonselective(n), ctx);
            let m = filtered.len();
            onpair_take_compact(&filtered, &selective_indices(m), ctx).unwrap()
        }
    }
}

/// Apply the scenario in `OnPairView` space (metadata-only on every op).
fn run_view(
    base: &OnPairViewArray,
    scenario: Scenario,
    ctx: &mut vortex_array::ExecutionCtx,
) -> OnPairViewArray {
    let n = base.len();
    match scenario {
        Scenario::FilterSelective => view_filter(base, &mask_selective(n), ctx),
        Scenario::FilterNonSelective => view_filter(base, &mask_nonselective(n), ctx),
        Scenario::TakeShuffle => view_take(base, &shuffle_indices(n), ctx),
        Scenario::TakeSelective => view_take(base, &selective_indices(n), ctx),
        Scenario::TakeNonSelective => view_take(base, &identity_indices(n), ctx),
        Scenario::FilterThenTake => {
            let filtered = view_filter(base, &mask_nonselective(n), ctx);
            let m = filtered.len();
            view_take(&filtered, &selective_indices(m), ctx)
        }
    }
}

fn canonicalize(arr: ArrayRef, ctx: &mut vortex_array::ExecutionCtx) -> VarBinViewArray {
    arr.execute::<VarBinViewArray>(ctx).unwrap()
}

const CASES: &[(Corpus, Scenario)] = &[
    (Corpus::ManyShort, Scenario::FilterSelective),
    (Corpus::ManyShort, Scenario::FilterNonSelective),
    (Corpus::ManyShort, Scenario::TakeShuffle),
    (Corpus::ManyShort, Scenario::TakeSelective),
    (Corpus::ManyShort, Scenario::TakeNonSelective),
    (Corpus::ManyShort, Scenario::FilterThenTake),
    (Corpus::FewLong, Scenario::FilterSelective),
    (Corpus::FewLong, Scenario::FilterNonSelective),
    (Corpus::FewLong, Scenario::TakeShuffle),
    (Corpus::FewLong, Scenario::TakeSelective),
    (Corpus::FewLong, Scenario::TakeNonSelective),
    (Corpus::FewLong, Scenario::FilterThenTake),
];

// ─── Ops phase ──────────────────────────────────────────────────────────

#[divan::bench(args = CASES)]
fn onpair_ops(bencher: Bencher, case: (Corpus, Scenario)) {
    let (c, scenario) = case;
    let base = base(c);
    bencher.bench_local(|| {
        let mut ctx = SESSION.create_execution_ctx();
        divan::black_box(run_onpair(&base.onpair, scenario, &mut ctx));
    });
}

#[divan::bench(args = CASES)]
fn view_ops(bencher: Bencher, case: (Corpus, Scenario)) {
    let (c, scenario) = case;
    let base = base(c);
    bencher.bench_local(|| {
        let mut ctx = SESSION.create_execution_ctx();
        divan::black_box(run_view(&base.view, scenario, &mut ctx));
    });
}

// ─── Canonicalise phase ──────────────────────────────────────────────────

#[divan::bench(args = CASES)]
fn onpair_canonicalize(bencher: Bencher, case: (Corpus, Scenario)) {
    let (c, scenario) = case;
    let base = base(c);
    bencher
        .with_inputs(|| {
            let mut ctx = SESSION.create_execution_ctx();
            run_onpair(&base.onpair, scenario, &mut ctx).into_array()
        })
        .bench_local_values(|arr| {
            let mut ctx = SESSION.create_execution_ctx();
            divan::black_box(canonicalize(arr, &mut ctx));
        });
}

#[divan::bench(args = CASES)]
fn view_canonicalize(bencher: Bencher, case: (Corpus, Scenario)) {
    let (c, scenario) = case;
    let base = base(c);
    bencher
        .with_inputs(|| {
            let mut ctx = SESSION.create_execution_ctx();
            run_view(&base.view, scenario, &mut ctx).into_array()
        })
        .bench_local_values(|arr| {
            let mut ctx = SESSION.create_execution_ctx();
            divan::black_box(canonicalize(arr, &mut ctx));
        });
}

// ─── Span-decode vs gather sweep over gap density ────────────────────────
//
// A filter keeping `p`% of rows (in clustered runs) leaves a span whose live
// fraction is ~`p`%. Decoding the span carries the dead gap bytes; gathering
// avoids them at the cost of random reads. This sweep finds the crossover.

const KEEP_PCT: &[(Corpus, u32)] = &[
    (Corpus::ManyShort, 95),
    (Corpus::ManyShort, 75),
    (Corpus::ManyShort, 50),
    (Corpus::ManyShort, 25),
    (Corpus::ManyShort, 10),
    (Corpus::ManyShort, 2),
    (Corpus::FewLong, 50),
    (Corpus::FewLong, 10),
];

fn filtered_view(c: Corpus, keep_pct: u32) -> OnPairViewArray {
    let base = base(c);
    let n = base.view.len();
    let mask = Mask::from_iter((0..n).map(|i| (i as u32 % 100) < keep_pct));
    let mut ctx = SESSION.create_execution_ctx();
    view_filter(&base.view, &mask, &mut ctx)
}

#[divan::bench(args = KEEP_PCT)]
fn canon_span(bencher: Bencher, case: (Corpus, u32)) {
    let (c, keep) = case;
    bencher
        .with_inputs(|| filtered_view(c, keep))
        .bench_local_values(|view| {
            let mut ctx = SESSION.create_execution_ctx();
            divan::black_box(
                canonicalize_with(view.as_view(), OnPairViewDecodeMode::SpanWithDead, &mut ctx)
                    .unwrap(),
            );
        });
}

#[divan::bench(args = KEEP_PCT)]
fn canon_gather(bencher: Bencher, case: (Corpus, u32)) {
    let (c, keep) = case;
    bencher
        .with_inputs(|| filtered_view(c, keep))
        .bench_local_values(|view| {
            let mut ctx = SESSION.create_execution_ctx();
            divan::black_box(
                canonicalize_with(view.as_view(), OnPairViewDecodeMode::Gather, &mut ctx).unwrap(),
            );
        });
}

fn main() {
    divan::main();
}
