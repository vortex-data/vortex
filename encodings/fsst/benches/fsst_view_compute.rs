// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compares the two ways to run a `filter`/`take` pipeline that ends in a `VarBinViewArray`:
//!
//! 1. **fsst pipeline**: stay in [`FSSTArray`] at every step, compacting the codes into a fresh
//!    [`FSSTArray`] each time (the kernels delegate to `VarBin`, rewriting the byte heap), then
//!    canonicalize to a [`VarBinViewArray`] at the end.
//! 2. **fsstview pipeline**: convert to [`FSSTViewArray`] and apply the metadata-only kernels
//!    (offsets/sizes only — the byte heap is never touched), then canonicalize to a
//!    [`VarBinViewArray`] at the end.
//!
//! Kernels are invoked directly (no Vortex execution/dispatch) so each part is measured in
//! isolation: the `_step` benches measure just the filter/take hop; the `_pipeline` benches
//! measure the hop plus the final canonicalization. For the fsstview pipeline the final
//! canonicalization is measured under each [`FsstViewCompaction`] strategy so the compaction
//! trade-off is visible directly.
//!
//! Two ~2 MiB (uncompressed) inputs are used: one with **many short** strings and one with
//! **fewer long** strings.
//!
//! Observed (medians): the fsstview hop is far cheaper in both cases (no heap rewrite) — e.g.
//! `take many_short/shuffle` is ~650 µs vs ~2.84 ms for fsst. For the final canonicalization,
//! `GatherBulk` (compact) beats `PerElement` (no compact) across the whole range, short *and*
//! long strings, because it pays FSST's slow decode-tail once instead of once per element; that's
//! why `Auto` compacts whenever the codes aren't contiguous.

#![expect(clippy::unwrap_used)]

use divan::Bencher;
use divan::black_box;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::dict::TakeExecute;
use vortex_array::arrays::filter::FilterKernel;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_fsst::FSST;
use vortex_fsst::FSSTArray;
use vortex_fsst::FSSTView;
use vortex_fsst::FSSTViewArray;
use vortex_fsst::FsstViewCompaction;
use vortex_fsst::canonicalize_fsstview_to_varbin;
use vortex_fsst::canonicalize_fsstview_with;
use vortex_fsst::fsst_compress;
use vortex_fsst::fsst_train_compressor;
use vortex_fsst::fsstview_from_fsst;
use vortex_mask::Mask;

fn main() {
    divan::main();
}

/// ~2 MiB of uncompressed string data, in two shapes.
const TARGET_UNCOMPRESSED: usize = 2 * 1024 * 1024;

#[derive(Clone, Copy, Debug)]
enum Shape {
    /// Many short strings (~12 bytes each) — small per-element work.
    ManyShort,
    /// Fewer long strings (~256 bytes each) — large per-element work.
    FewLong,
}

impl Shape {
    fn avg_len(self) -> usize {
        match self {
            Shape::ManyShort => 12,
            Shape::FewLong => 256,
        }
    }

    fn count(self) -> usize {
        TARGET_UNCOMPRESSED / self.avg_len()
    }

    fn name(self) -> &'static str {
        match self {
            Shape::ManyShort => "many_short",
            Shape::FewLong => "few_long",
        }
    }
}

/// Build a ~2 MiB input. We use a small alphabet so FSST finds good symbols (realistic
/// compression), with some shared substrings to mimic real string columns.
fn generate(shape: Shape) -> VarBinArray {
    let mut rng = StdRng::seed_from_u64(42);
    let count = shape.count();
    let avg_len = shape.avg_len();
    let mut strings: Vec<Box<[u8]>> = Vec::with_capacity(count);

    const WORDS: &[&str] = &[
        "https://", "example", "vortex", ".com/", "path", "query=", "value", "data", "alpha",
        "bravo", "charlie", "delta", "_", "-", "/", "0123",
    ];

    for _ in 0..count {
        let target = avg_len * rng.random_range(70..=130) / 100;
        let mut s = String::with_capacity(target + 8);
        while s.len() < target {
            s.push_str(WORDS[rng.random_range(0..WORDS.len())]);
        }
        s.truncate(target.max(1));
        strings.push(s.into_bytes().into_boxed_slice());
    }

    VarBinArray::from_iter(
        strings.into_iter().map(Some),
        DType::Utf8(Nullability::NonNullable),
    )
}

fn compress(varbin: &VarBinArray, ctx: &mut ExecutionCtx) -> FSSTArray {
    let compressor = fsst_train_compressor(varbin);
    fsst_compress(varbin, varbin.len(), varbin.dtype(), &compressor, ctx)
}

/// A selective mask keeps ~10% of rows; a non-selective mask keeps ~90%.
fn make_mask(len: usize, keep_fraction: f64) -> Mask {
    let mut rng = StdRng::seed_from_u64(7);
    Mask::from_iter((0..len).map(|_| rng.random_bool(keep_fraction)))
}

#[derive(Clone, Copy, Debug)]
enum TakeKind {
    /// A full shuffle (permutation of all rows) — same length, reordered.
    Shuffle,
    /// Very selective — pick ~5% of rows at random (with possible repeats).
    Selective,
    /// Not selective — pick ~150% of rows at random (duplicates, output grows).
    Dense,
}

impl TakeKind {
    fn name(self) -> &'static str {
        match self {
            TakeKind::Shuffle => "shuffle",
            TakeKind::Selective => "selective",
            TakeKind::Dense => "dense",
        }
    }
}

fn compaction_name(strategy: FsstViewCompaction) -> &'static str {
    match strategy {
        FsstViewCompaction::Auto => "auto",
        FsstViewCompaction::Direct => "direct",
        FsstViewCompaction::GatherBulk => "gather_bulk",
        FsstViewCompaction::PerElement => "per_element",
        FsstViewCompaction::RunCoalesce => "run_coalesce",
    }
}

fn make_indices(len: usize, kind: TakeKind) -> ArrayRef {
    let mut rng = StdRng::seed_from_u64(11);
    let indices: Vec<u64> = match kind {
        TakeKind::Shuffle => {
            let mut v: Vec<u64> = (0..len as u64).collect();
            // Fisher-Yates.
            for i in (1..v.len()).rev() {
                v.swap(i, rng.random_range(0..=i));
            }
            v
        }
        TakeKind::Selective => (0..(len / 20).max(1))
            .map(|_| rng.random_range(0..len as u64))
            .collect(),
        TakeKind::Dense => (0..(len * 3 / 2))
            .map(|_| rng.random_range(0..len as u64))
            .collect(),
    };
    PrimitiveArray::from_iter(indices).into_array()
}

// ----- direct kernel wrappers (no Vortex dispatch) ---------------------------------------------

fn fsst_filter(array: &FSSTArray, mask: &Mask, ctx: &mut ExecutionCtx) -> FSSTArray {
    <FSST as FilterKernel>::filter(array.as_view(), mask, ctx)
        .unwrap()
        .unwrap()
        .try_downcast::<FSST>()
        .ok()
        .unwrap()
}

fn fsst_take(array: &FSSTArray, indices: &ArrayRef, ctx: &mut ExecutionCtx) -> FSSTArray {
    <FSST as TakeExecute>::take(array.as_view(), indices, ctx)
        .unwrap()
        .unwrap()
        .try_downcast::<FSST>()
        .ok()
        .unwrap()
}

fn view_filter(array: &FSSTArray, mask: &Mask, ctx: &mut ExecutionCtx) -> ArrayRef {
    let view = fsstview_from_fsst(array, ctx).unwrap();
    <FSSTView as FilterKernel>::filter(view.as_view(), mask, ctx)
        .unwrap()
        .unwrap()
}

fn view_take(array: &FSSTArray, indices: &ArrayRef, ctx: &mut ExecutionCtx) -> ArrayRef {
    let view = fsstview_from_fsst(array, ctx).unwrap();
    <FSSTView as TakeExecute>::take(view.as_view(), indices, ctx)
        .unwrap()
        .unwrap()
}

fn fsst_to_canonical(array: &FSSTArray, ctx: &mut ExecutionCtx) -> ArrayRef {
    // Decompress straight to a VarBinView via the VarBin codes (the FSST canonical path).
    array
        .clone()
        .into_array()
        .execute::<VarBinViewArray>(ctx)
        .unwrap()
        .into_array()
}

const SHAPES: &[Shape] = &[Shape::ManyShort, Shape::FewLong];

// =============================== FILTER ========================================================

/// Filter masks to exercise: selective (~10% kept) and non-selective (~90% kept).
const FILTER_KEEP: &[(&str, f64)] = &[("selective_10pct", 0.10), ("nonselective_90pct", 0.90)];

#[divan::bench(args = filter_args())]
fn filter_step_fsst(bencher: Bencher, args: FilterArg) {
    let varbin = generate(args.shape);
    let fsst = compress(&varbin, &mut LEGACY_SESSION.create_execution_ctx());
    let mask = make_mask(fsst.len(), args.keep);
    bencher
        .with_inputs(|| (&fsst, &mask, LEGACY_SESSION.create_execution_ctx()))
        .bench_refs(|(fsst, mask, ctx)| black_box(fsst_filter(fsst, mask, ctx)));
}

#[divan::bench(args = filter_args())]
fn filter_step_view(bencher: Bencher, args: FilterArg) {
    let varbin = generate(args.shape);
    let fsst = compress(&varbin, &mut LEGACY_SESSION.create_execution_ctx());
    let mask = make_mask(fsst.len(), args.keep);
    bencher
        .with_inputs(|| (&fsst, &mask, LEGACY_SESSION.create_execution_ctx()))
        .bench_refs(|(fsst, mask, ctx)| black_box(view_filter(fsst, mask, ctx)));
}

/// Metadata-only filter measured in isolation (conversion hoisted out). See `take_op_only_view`.
#[divan::bench(args = filter_args())]
fn filter_op_only_view(bencher: Bencher, args: FilterArg) {
    let varbin = generate(args.shape);
    let fsst = compress(&varbin, &mut LEGACY_SESSION.create_execution_ctx());
    let view = fsstview_from_fsst(&fsst, &mut LEGACY_SESSION.create_execution_ctx()).unwrap();
    let mask = make_mask(view.len(), args.keep);
    bencher
        .with_inputs(|| (&view, &mask, LEGACY_SESSION.create_execution_ctx()))
        .bench_refs(|(view, mask, ctx)| {
            black_box(
                <FSSTView as FilterKernel>::filter(view.as_view(), mask, ctx)
                    .unwrap()
                    .unwrap(),
            )
        });
}

/// Full pipeline: filter (compacting into another FSSTArray) then canonicalize to VarBinView.
#[divan::bench(args = filter_args())]
fn filter_pipeline_fsst(bencher: Bencher, args: FilterArg) {
    let varbin = generate(args.shape);
    let fsst = compress(&varbin, &mut LEGACY_SESSION.create_execution_ctx());
    let mask = make_mask(fsst.len(), args.keep);
    bencher
        .with_inputs(|| (&fsst, &mask, LEGACY_SESSION.create_execution_ctx()))
        .bench_refs(|(fsst, mask, ctx)| {
            let filtered = fsst_filter(fsst, mask, ctx);
            black_box(fsst_to_canonical(&filtered, ctx))
        });
}

/// Full pipeline: filter to FSSTView then canonicalize, once per compaction strategy.
#[divan::bench(args = filter_view_pipeline_args())]
fn filter_pipeline_view(bencher: Bencher, args: FilterViewArg) {
    let varbin = generate(args.shape);
    let fsst = compress(&varbin, &mut LEGACY_SESSION.create_execution_ctx());
    let mask = make_mask(fsst.len(), args.keep);
    bencher
        .with_inputs(|| (&fsst, &mask, LEGACY_SESSION.create_execution_ctx()))
        .bench_refs(|(fsst, mask, ctx)| {
            let view = view_filter(fsst, mask, ctx)
                .try_downcast::<FSSTView>()
                .ok()
                .unwrap();
            black_box(canonicalize_fsstview_with(view.as_view(), args.strategy, ctx).unwrap())
        });
}

// =============================== TAKE ==========================================================

const TAKE_KINDS: &[TakeKind] = &[TakeKind::Shuffle, TakeKind::Selective, TakeKind::Dense];

#[divan::bench(args = take_args())]
fn take_step_fsst(bencher: Bencher, args: TakeArg) {
    let varbin = generate(args.shape);
    let fsst = compress(&varbin, &mut LEGACY_SESSION.create_execution_ctx());
    let indices = make_indices(fsst.len(), args.kind);
    bencher
        .with_inputs(|| (&fsst, &indices, LEGACY_SESSION.create_execution_ctx()))
        .bench_refs(|(fsst, indices, ctx)| black_box(fsst_take(fsst, indices, ctx)));
}

#[divan::bench(args = take_args())]
fn take_step_view(bencher: Bencher, args: TakeArg) {
    let varbin = generate(args.shape);
    let fsst = compress(&varbin, &mut LEGACY_SESSION.create_execution_ctx());
    let indices = make_indices(fsst.len(), args.kind);
    bencher
        .with_inputs(|| (&fsst, &indices, LEGACY_SESSION.create_execution_ctx()))
        .bench_refs(|(fsst, indices, ctx)| black_box(view_take(fsst, indices, ctx)));
}

/// The metadata-only take measured *in isolation*: the FSST→view conversion is hoisted out of the
/// timed loop (a chain converts once), so this is the apples-to-apples "is the view op itself as
/// cheap as a ListView op" comparison. The `*_step_view` bench above instead folds the one-time
/// conversion into every op, which only the first op of a chain actually pays.
#[divan::bench(args = take_args())]
fn take_op_only_view(bencher: Bencher, args: TakeArg) {
    let varbin = generate(args.shape);
    let fsst = compress(&varbin, &mut LEGACY_SESSION.create_execution_ctx());
    let view = fsstview_from_fsst(&fsst, &mut LEGACY_SESSION.create_execution_ctx()).unwrap();
    let indices = make_indices(view.len(), args.kind);
    bencher
        .with_inputs(|| (&view, &indices, LEGACY_SESSION.create_execution_ctx()))
        .bench_refs(|(view, indices, ctx)| {
            black_box(
                <FSSTView as TakeExecute>::take(view.as_view(), indices, ctx)
                    .unwrap()
                    .unwrap(),
            )
        });
}

#[divan::bench(args = take_args())]
fn take_pipeline_fsst(bencher: Bencher, args: TakeArg) {
    let varbin = generate(args.shape);
    let fsst = compress(&varbin, &mut LEGACY_SESSION.create_execution_ctx());
    let indices = make_indices(fsst.len(), args.kind);
    bencher
        .with_inputs(|| (&fsst, &indices, LEGACY_SESSION.create_execution_ctx()))
        .bench_refs(|(fsst, indices, ctx)| {
            let taken = fsst_take(fsst, indices, ctx);
            black_box(fsst_to_canonical(&taken, ctx))
        });
}

#[divan::bench(args = take_view_pipeline_args())]
fn take_pipeline_view(bencher: Bencher, args: TakeViewArg) {
    let varbin = generate(args.shape);
    let fsst = compress(&varbin, &mut LEGACY_SESSION.create_execution_ctx());
    let indices = make_indices(fsst.len(), args.kind);
    bencher
        .with_inputs(|| (&fsst, &indices, LEGACY_SESSION.create_execution_ctx()))
        .bench_refs(|(fsst, indices, ctx)| {
            let view = view_take(fsst, indices, ctx)
                .try_downcast::<FSSTView>()
                .ok()
                .unwrap();
            black_box(canonicalize_fsstview_with(view.as_view(), args.strategy, ctx).unwrap())
        });
}

// =============================== COMBINATION ===================================================

/// A filter (selective) followed by a take (shuffle) — the realistic "scan then reorder" shape.
/// fsst path compacts twice; fsstview path stays metadata-only until the final canonicalize.
#[divan::bench(args = SHAPES)]
fn combo_pipeline_fsst(bencher: Bencher, shape: Shape) {
    let varbin = generate(shape);
    let fsst = compress(&varbin, &mut LEGACY_SESSION.create_execution_ctx());
    let mask = make_mask(fsst.len(), 0.10);
    bencher
        .with_inputs(|| (&fsst, &mask, LEGACY_SESSION.create_execution_ctx()))
        .bench_refs(|(fsst, mask, ctx)| {
            let filtered = fsst_filter(fsst, mask, ctx);
            let indices = make_indices(filtered.len(), TakeKind::Shuffle);
            let taken = fsst_take(&filtered, &indices, ctx);
            black_box(fsst_to_canonical(&taken, ctx))
        });
}

#[divan::bench(args = SHAPES)]
fn combo_pipeline_view(bencher: Bencher, shape: Shape) {
    let varbin = generate(shape);
    let fsst = compress(&varbin, &mut LEGACY_SESSION.create_execution_ctx());
    let mask = make_mask(fsst.len(), 0.10);
    bencher
        .with_inputs(|| (&fsst, &mask, LEGACY_SESSION.create_execution_ctx()))
        .bench_refs(|(fsst, mask, ctx)| {
            // filter -> view, then take on the view (both metadata-only), then canonicalize.
            let filtered = view_filter(fsst, mask, ctx)
                .try_downcast::<FSSTView>()
                .ok()
                .unwrap();
            let indices = make_indices(filtered.len(), TakeKind::Shuffle);
            let taken = <FSSTView as TakeExecute>::take(filtered.as_view(), &indices, ctx)
                .unwrap()
                .unwrap()
                .try_downcast::<FSSTView>()
                .ok()
                .unwrap();
            black_box(
                canonicalize_fsstview_with(taken.as_view(), FsstViewCompaction::Auto, ctx).unwrap(),
            )
        });
}

// =============================== CHAIN =========================================================

/// Number of ops in the chain benchmark.
const CHAIN_LEN: usize = 5;

/// A chain of `CHAIN_LEN` alternating filter/take ops ending in a canonicalization.
///
/// This is where the view model is meant to dominate: each fsst op re-compacts the byte heap,
/// so the cost compounds with chain length, whereas the view converts to offsets+sizes *once*
/// and every subsequent op is metadata-only, deferring the single gather+decode to the final
/// canonicalize. We keep every op only mildly selective (filter keeps 80%, take is a shuffle)
/// so there's still substantial data at the end — i.e. the heap rewrites the fsst path pays are
/// real work, not optimized away to nothing.
#[divan::bench(args = SHAPES)]
fn chain_pipeline_fsst(bencher: Bencher, shape: Shape) {
    let varbin = generate(shape);
    let fsst = compress(&varbin, &mut LEGACY_SESSION.create_execution_ctx());
    bencher
        .with_inputs(|| (&fsst, LEGACY_SESSION.create_execution_ctx()))
        .bench_refs(|(fsst, ctx)| {
            let mut cur = (*fsst).clone();
            for op in 0..CHAIN_LEN {
                if op % 2 == 0 {
                    let mask = make_mask(cur.len(), 0.80);
                    cur = fsst_filter(&cur, &mask, ctx);
                } else {
                    let indices = make_indices(cur.len(), TakeKind::Shuffle);
                    cur = fsst_take(&cur, &indices, ctx);
                }
            }
            black_box(fsst_to_canonical(&cur, ctx))
        });
}

#[divan::bench(args = SHAPES)]
fn chain_pipeline_view(bencher: Bencher, shape: Shape) {
    let varbin = generate(shape);
    let fsst = compress(&varbin, &mut LEGACY_SESSION.create_execution_ctx());
    bencher
        .with_inputs(|| (&fsst, LEGACY_SESSION.create_execution_ctx()))
        .bench_refs(|(fsst, ctx)| {
            // Convert to the view once, then chain metadata-only ops, canonicalize once at the end.
            let mut cur = fsstview_from_fsst(fsst, ctx).unwrap();
            for op in 0..CHAIN_LEN {
                let next = if op % 2 == 0 {
                    let mask = make_mask(cur.len(), 0.80);
                    <FSSTView as FilterKernel>::filter(cur.as_view(), &mask, ctx)
                        .unwrap()
                        .unwrap()
                } else {
                    let indices = make_indices(cur.len(), TakeKind::Shuffle);
                    <FSSTView as TakeExecute>::take(cur.as_view(), &indices, ctx)
                        .unwrap()
                        .unwrap()
                };
                cur = next.try_downcast::<FSSTView>().ok().unwrap();
            }
            black_box(
                canonicalize_fsstview_with(cur.as_view(), FsstViewCompaction::Auto, ctx).unwrap(),
            )
        });
}

// =============================== SINGLE FILTER + EXPORT (2x2) ==================================
//
// A single filter, then export to a canonical string array. The matrix is
// {fsst, fsstview} x {VarBinView, VarBin}:
//  - fsst path:     filter rewrites the compressed heap (VarBin filter on codes), then decode.
//  - fsstview path: filter is metadata-only, then decode (coalesced gather + bulk) at export.
//  - VarBinView export: build a 16-byte view per element.
//  - VarBin export:     build len+1 cumulative offsets over the contiguous decoded bytes.

fn export_view(array: &FSSTArray, mask: &Mask, ctx: &mut ExecutionCtx) -> FSSTViewArray {
    view_filter(array, mask, ctx)
        .try_downcast::<FSSTView>()
        .ok()
        .unwrap()
}

#[divan::bench(args = filter_args())]
fn export_fsst_to_varbinview(bencher: Bencher, args: FilterArg) {
    let varbin = generate(args.shape);
    let fsst = compress(&varbin, &mut LEGACY_SESSION.create_execution_ctx());
    let mask = make_mask(fsst.len(), args.keep);
    bencher
        .with_inputs(|| (&fsst, &mask, LEGACY_SESSION.create_execution_ctx()))
        .bench_refs(|(fsst, mask, ctx)| {
            let filtered = fsst_filter(fsst, mask, ctx);
            black_box(fsst_to_canonical(&filtered, ctx))
        });
}

#[divan::bench(args = filter_args())]
fn export_fsst_to_varbin(bencher: Bencher, args: FilterArg) {
    let varbin = generate(args.shape);
    let fsst = compress(&varbin, &mut LEGACY_SESSION.create_execution_ctx());
    let mask = make_mask(fsst.len(), args.keep);
    bencher
        .with_inputs(|| (&fsst, &mask, LEGACY_SESSION.create_execution_ctx()))
        .bench_refs(|(fsst, mask, ctx)| {
            // Filter stays in FSST; reach VarBin by reinterpreting the (now contiguous) codes as a
            // view and exporting offsets+bytes.
            let filtered = fsst_filter(fsst, mask, ctx);
            let view = fsstview_from_fsst(&filtered, ctx).unwrap();
            black_box(
                canonicalize_fsstview_to_varbin(view.as_view(), FsstViewCompaction::Auto, ctx)
                    .unwrap(),
            )
        });
}

#[divan::bench(args = filter_args())]
fn export_view_to_varbinview(bencher: Bencher, args: FilterArg) {
    let varbin = generate(args.shape);
    let fsst = compress(&varbin, &mut LEGACY_SESSION.create_execution_ctx());
    let mask = make_mask(fsst.len(), args.keep);
    bencher
        .with_inputs(|| (&fsst, &mask, LEGACY_SESSION.create_execution_ctx()))
        .bench_refs(|(fsst, mask, ctx)| {
            let view = export_view(fsst, mask, ctx);
            black_box(
                canonicalize_fsstview_with(view.as_view(), FsstViewCompaction::Auto, ctx).unwrap(),
            )
        });
}

#[divan::bench(args = filter_args())]
fn export_view_to_varbin(bencher: Bencher, args: FilterArg) {
    let varbin = generate(args.shape);
    let fsst = compress(&varbin, &mut LEGACY_SESSION.create_execution_ctx());
    let mask = make_mask(fsst.len(), args.keep);
    bencher
        .with_inputs(|| (&fsst, &mask, LEGACY_SESSION.create_execution_ctx()))
        .bench_refs(|(fsst, mask, ctx)| {
            let view = export_view(fsst, mask, ctx);
            black_box(
                canonicalize_fsstview_to_varbin(view.as_view(), FsstViewCompaction::Auto, ctx)
                    .unwrap(),
            )
        });
}

// =============================== arg plumbing ==================================================

#[derive(Clone, Copy)]
struct FilterArg {
    shape: Shape,
    keep: f64,
    label: &'static str,
}

impl std::fmt::Display for FilterArg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", self.shape.name(), self.label)
    }
}

fn filter_args() -> Vec<FilterArg> {
    let mut v = Vec::new();
    for &shape in SHAPES {
        for &(label, keep) in FILTER_KEEP {
            v.push(FilterArg { shape, keep, label });
        }
    }
    v
}

#[derive(Clone, Copy)]
struct FilterViewArg {
    shape: Shape,
    keep: f64,
    label: &'static str,
    strategy: FsstViewCompaction,
}

impl std::fmt::Display for FilterViewArg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}/{}/{}",
            self.shape.name(),
            self.label,
            compaction_name(self.strategy)
        )
    }
}

fn filter_view_pipeline_args() -> Vec<FilterViewArg> {
    let mut v = Vec::new();
    for &shape in SHAPES {
        for &(label, keep) in FILTER_KEEP {
            for &strategy in COMPACTIONS {
                v.push(FilterViewArg {
                    shape,
                    keep,
                    label,
                    strategy,
                });
            }
        }
    }
    v
}

#[derive(Clone, Copy)]
struct TakeArg {
    shape: Shape,
    kind: TakeKind,
}

impl std::fmt::Display for TakeArg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", self.shape.name(), self.kind.name())
    }
}

fn take_args() -> Vec<TakeArg> {
    let mut v = Vec::new();
    for &shape in SHAPES {
        for &kind in TAKE_KINDS {
            v.push(TakeArg { shape, kind });
        }
    }
    v
}

#[derive(Clone, Copy)]
struct TakeViewArg {
    shape: Shape,
    kind: TakeKind,
    strategy: FsstViewCompaction,
}

impl std::fmt::Display for TakeViewArg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}/{}/{}",
            self.shape.name(),
            self.kind.name(),
            compaction_name(self.strategy)
        )
    }
}

const COMPACTIONS: &[FsstViewCompaction] = &[
    FsstViewCompaction::Auto,
    FsstViewCompaction::GatherBulk,
    FsstViewCompaction::PerElement,
    FsstViewCompaction::RunCoalesce,
];

fn take_view_pipeline_args() -> Vec<TakeViewArg> {
    let mut v = Vec::new();
    for &shape in SHAPES {
        for &kind in TAKE_KINDS {
            for &strategy in COMPACTIONS {
                v.push(TakeViewArg {
                    shape,
                    kind,
                    strategy,
                });
            }
        }
    }
    v
}
