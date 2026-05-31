// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! FSST vs FSSTView on synthetic string data, ending in a `VarBinViewArray`.
//!
//! `fsst` rewrites the compressed code heap on every `filter`/`take` (it delegates to `VarBin`);
//! `fsstview` keeps those ops metadata-only and decodes once at canonicalize. This bench measures
//! both a single filter and a 5-op filter/take chain, over two shapes — many short strings and
//! fewer long strings — with a clustered selection (the realistic shape, where survivors form runs
//! the view's `RunDecode` export exploits). For a benchmark on real FineWeb columns, see
//! `fsst_view_fineweb`.

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
use vortex_fsst::FsstViewCompaction;
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
    /// Many short strings (~12 bytes each).
    ManyShort,
    /// Fewer long strings (~256 bytes each).
    FewLong,
}

impl Shape {
    fn avg_len(self) -> usize {
        match self {
            Shape::ManyShort => 12,
            Shape::FewLong => 256,
        }
    }
}

const SHAPES: &[Shape] = &[Shape::ManyShort, Shape::FewLong];

/// Build a ~2 MiB input from a small alphabet so FSST finds good symbols, with shared substrings
/// to mimic real string columns.
fn generate(shape: Shape) -> VarBinArray {
    let mut rng = StdRng::seed_from_u64(42);
    let avg_len = shape.avg_len();
    let count = TARGET_UNCOMPRESSED / avg_len;
    const WORDS: &[&str] = &[
        "https://", "example", "vortex", ".com/", "path", "query=", "value", "data", "alpha",
        "bravo", "charlie", "delta", "_", "-", "/", "0123",
    ];
    let mut strings: Vec<Box<[u8]>> = Vec::with_capacity(count);
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

/// Clustered selection (32 bursts, ~`keep` fraction) — survivors form runs, the realistic shape.
fn clustered_mask(len: usize, keep: f64) -> Mask {
    let mut rng = StdRng::seed_from_u64(9);
    #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let total = (len as f64 * keep) as usize;
    let burst_len = (total / 32).max(1);
    let mut keep_set = vec![false; len];
    for _ in 0..32 {
        let start = rng.random_range(0..len.saturating_sub(burst_len).max(1));
        for j in start..(start + burst_len).min(len) {
            keep_set[j] = true;
        }
    }
    Mask::from_iter(keep_set)
}

/// Sorted-index take (~`keep` fraction) — an index lookup; preserves heap order.
fn sorted_take(len: usize, keep: f64) -> ArrayRef {
    let mut rng = StdRng::seed_from_u64(13);
    #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let n = (len as f64 * keep) as usize;
    let mut idx: Vec<u64> = (0..n).map(|_| rng.random_range(0..len as u64)).collect();
    idx.sort_unstable();
    PrimitiveArray::from_iter(idx).into_array()
}

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

fn fsst_to_vbv(array: &FSSTArray, ctx: &mut ExecutionCtx) -> ArrayRef {
    array
        .clone()
        .into_array()
        .execute::<VarBinViewArray>(ctx)
        .unwrap()
        .into_array()
}

// =============================== SINGLE FILTER -> VarBinView ===================================

#[divan::bench(args = SHAPES)]
fn single_filter_fsst(bencher: Bencher, shape: Shape) {
    let fsst = compress(&generate(shape), &mut LEGACY_SESSION.create_execution_ctx());
    let mask = clustered_mask(fsst.len(), 0.10);
    bencher
        .with_inputs(|| (&fsst, &mask, LEGACY_SESSION.create_execution_ctx()))
        .bench_refs(|(fsst, mask, ctx)| {
            let filtered = fsst_filter(fsst, mask, ctx);
            black_box(fsst_to_vbv(&filtered, ctx))
        });
}

#[divan::bench(args = SHAPES)]
fn single_filter_view(bencher: Bencher, shape: Shape) {
    let fsst = compress(&generate(shape), &mut LEGACY_SESSION.create_execution_ctx());
    let mask = clustered_mask(fsst.len(), 0.10);
    bencher
        .with_inputs(|| (&fsst, &mask, LEGACY_SESSION.create_execution_ctx()))
        .bench_refs(|(fsst, mask, ctx)| {
            let view = fsstview_from_fsst(fsst, ctx).unwrap();
            let filtered = <FSSTView as FilterKernel>::filter(view.as_view(), mask, ctx)
                .unwrap()
                .unwrap()
                .try_downcast::<FSSTView>()
                .ok()
                .unwrap();
            black_box(
                canonicalize_fsstview_with(filtered.as_view(), FsstViewCompaction::Auto, ctx)
                    .unwrap(),
            )
        });
}

// =============================== CHAIN (convert once, N ops, export once) ======================

const CHAIN_LEN: usize = 5;

#[divan::bench(args = SHAPES)]
fn chain_fsst(bencher: Bencher, shape: Shape) {
    let fsst = compress(&generate(shape), &mut LEGACY_SESSION.create_execution_ctx());
    bencher
        .with_inputs(|| (&fsst, LEGACY_SESSION.create_execution_ctx()))
        .bench_refs(|(fsst, ctx)| {
            let mut cur = (*fsst).clone();
            for op in 0..CHAIN_LEN {
                if op % 2 == 0 {
                    let mask = clustered_mask(cur.len(), 0.80);
                    cur = fsst_filter(&cur, &mask, ctx);
                } else {
                    let indices = sorted_take(cur.len(), 0.80);
                    cur = fsst_take(&cur, &indices, ctx);
                }
            }
            black_box(fsst_to_vbv(&cur, ctx))
        });
}

#[divan::bench(args = SHAPES)]
fn chain_view(bencher: Bencher, shape: Shape) {
    let fsst = compress(&generate(shape), &mut LEGACY_SESSION.create_execution_ctx());
    bencher
        .with_inputs(|| (&fsst, LEGACY_SESSION.create_execution_ctx()))
        .bench_refs(|(fsst, ctx)| {
            // Convert once, then chain metadata-only ops, canonicalize once.
            let mut cur = fsstview_from_fsst(fsst, ctx).unwrap();
            for op in 0..CHAIN_LEN {
                let next = if op % 2 == 0 {
                    let mask = clustered_mask(cur.len(), 0.80);
                    <FSSTView as FilterKernel>::filter(cur.as_view(), &mask, ctx)
                        .unwrap()
                        .unwrap()
                } else {
                    let indices = sorted_take(cur.len(), 0.80);
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
