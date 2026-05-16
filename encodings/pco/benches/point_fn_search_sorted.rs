// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmark `search_sorted` on a PCO-encoded sorted array, comparing the legacy
//! path (per-probe `ExecutionCtx` construction, no caching) against a [`PointSession`]
//! (one ctx, scalar cache for repeated probes).
//!
//! Phase 1b: this benchmark measures the win from execution-context reuse and the
//! scalar cache, before PCO is ported to use [`PointDispatch::cached_block`]. Phase
//! 1d/e will add a block-cache benchmark once the PCO kernel is migrated.

#![expect(clippy::unwrap_used)]

use divan::Bencher;
use rand::RngExt;
use rand::SeedableRng;
use rand::distr::Uniform;
use rand::prelude::StdRng;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::point_fn::PointDispatch;
use vortex_array::point_fn::PointRuntime;
use vortex_array::point_fn::PointSession;
use vortex_array::scalar::Scalar;
use vortex_array::search_sorted::SearchSorted;
use vortex_array::search_sorted::SearchSortedSide;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_pco::Pco;

fn main() {
    divan::main();
}

const SIZES: &[usize] = &[1024, 16_384, 262_144];

fn build_pco_sorted(len: usize) -> (ArrayRef, Scalar) {
    let mut rng = StdRng::seed_from_u64(0);
    let range = Uniform::new(0i32, i32::MAX / 2).unwrap();
    let mut data: Vec<i32> = (0..len).map(|_| rng.sample(range)).collect();
    data.sort();
    let target = Scalar::from(data[len / 3]);
    let primitive = PrimitiveArray::new(Buffer::copy_from(&data), Validity::NonNullable);
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let pco = Pco::from_primitive(primitive.as_view(), 8, 1024, &mut ctx).unwrap();
    (pco.into_array(), target)
}

#[divan::bench(args = SIZES)]
fn legacy_search_sorted(bencher: Bencher, &len: &usize) {
    let (arr, target) = build_pco_sorted(len);
    bencher
        .with_inputs(|| (&arr, &target))
        .bench_refs(|(arr, target)| {
            // Legacy path: IndexOrd<Scalar> for ArrayRef constructs a fresh
            // ExecutionCtx per probe via LEGACY_SESSION.create_execution_ctx().
            arr.search_sorted(target, SearchSortedSide::Left).unwrap()
        });
}

#[divan::bench(args = SIZES)]
fn point_runtime_search_sorted(bencher: Bencher, &len: &usize) {
    let (arr, target) = build_pco_sorted(len);
    bencher
        .with_inputs(|| (&arr, &target))
        .bench_refs(|(arr, target)| {
            // PointRuntime: one ctx reused across all probes. No caching.
            let mut ctx = LEGACY_SESSION.create_execution_ctx();
            let mut rt = PointRuntime::new(&mut ctx);
            rt.search_sorted(arr, target, SearchSortedSide::Left).unwrap()
        });
}

#[divan::bench(args = SIZES)]
fn point_session_search_sorted(bencher: Bencher, &len: &usize) {
    let (arr, target) = build_pco_sorted(len);
    bencher
        .with_inputs(|| (&arr, &target))
        .bench_refs(|(arr, target)| {
            // PointSession: one ctx + scalar cache (helps the side-refinement pass).
            let mut ctx = LEGACY_SESSION.create_execution_ctx();
            let mut session = PointSession::new(&mut ctx);
            session
                .search_sorted(arr, target, SearchSortedSide::Left)
                .unwrap()
        });
}

/// Many searches reusing the same session — the realistic "batch of probes"
/// pattern that benefits most from the session caches.
#[divan::bench(args = SIZES)]
fn point_session_batched(bencher: Bencher, &len: &usize) {
    let (arr, _) = build_pco_sorted(len);
    let mut rng = StdRng::seed_from_u64(42);
    let range = Uniform::new(0i32, i32::MAX / 2).unwrap();
    let targets: Vec<Scalar> = (0..32).map(|_| Scalar::from(rng.sample(range))).collect();

    bencher
        .with_inputs(|| (&arr, &targets))
        .bench_refs(|(arr, targets)| {
            let mut ctx = LEGACY_SESSION.create_execution_ctx();
            let mut session = PointSession::new(&mut ctx);
            let mut result = 0usize;
            for t in *targets {
                result += session
                    .search_sorted(arr, t, SearchSortedSide::Left)
                    .unwrap()
                    .to_index();
            }
            result
        });
}

#[divan::bench(args = SIZES)]
fn legacy_batched(bencher: Bencher, &len: &usize) {
    let (arr, _) = build_pco_sorted(len);
    let mut rng = StdRng::seed_from_u64(42);
    let range = Uniform::new(0i32, i32::MAX / 2).unwrap();
    let targets: Vec<Scalar> = (0..32).map(|_| Scalar::from(rng.sample(range))).collect();

    bencher
        .with_inputs(|| (&arr, &targets))
        .bench_refs(|(arr, targets)| {
            let mut result = 0usize;
            for t in *targets {
                result += arr
                    .search_sorted(t, SearchSortedSide::Left)
                    .unwrap()
                    .to_index();
            }
            result
        });
}
