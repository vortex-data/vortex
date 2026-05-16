// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks for `point_search_sorted` overrides on the structural encodings
//! ported in phase 2 of the `point_fn` migration:
//!
//!   - Constant: O(1) closed-form
//!   - Slice:    one child search + clamp
//!   - Dict:     dict.search_sorted + codes.search_sorted (small dict win)
//!   - Chunked:  zone-map prune to candidate chunk
//!
//! Each comparison is legacy `arr.search_sorted(target, side)` (generic binary
//! search calling `execute_scalar` per probe, with fresh `ExecutionCtx`) vs
//! `PointSession::search_sorted` (uses the encoding's vtable override).

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
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::SliceArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::point_fn::PointDispatch;
use vortex_array::point_fn::PointSession;
use vortex_array::scalar::Scalar;
use vortex_array::search_sorted::SearchSorted;
use vortex_array::search_sorted::SearchSortedSide;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;

fn main() {
    divan::main();
}

const SIZES: &[usize] = &[1024, 16_384, 262_144];

// ────────────────────────────────────────────────────────────────────────────
// Constant
// ────────────────────────────────────────────────────────────────────────────

fn build_constant(len: usize) -> (ArrayRef, Scalar) {
    let arr = ConstantArray::new(Scalar::from(42i32), len).into_array();
    (arr, Scalar::from(42i32))
}

#[divan::bench(args = SIZES)]
fn constant_legacy(bencher: Bencher, &len: &usize) {
    let (arr, target) = build_constant(len);
    bencher
        .with_inputs(|| (&arr, &target))
        .bench_refs(|(arr, target)| arr.search_sorted(target, SearchSortedSide::Left).unwrap());
}

#[divan::bench(args = SIZES)]
fn constant_session(bencher: Bencher, &len: &usize) {
    let (arr, target) = build_constant(len);
    bencher
        .with_inputs(|| (&arr, &target))
        .bench_refs(|(arr, target)| {
            let mut ctx = LEGACY_SESSION.create_execution_ctx();
            let mut session = PointSession::new(&mut ctx);
            session
                .search_sorted(arr, target, SearchSortedSide::Left)
                .unwrap()
        });
}

// ────────────────────────────────────────────────────────────────────────────
// Slice(Primitive)
// ────────────────────────────────────────────────────────────────────────────

fn build_sorted_primitive(len: usize) -> ArrayRef {
    let mut rng = StdRng::seed_from_u64(0);
    let range = Uniform::new(0i32, i32::MAX / 2).unwrap();
    let mut data: Vec<i32> = (0..len).map(|_| rng.sample(range)).collect();
    data.sort();
    PrimitiveArray::new(Buffer::copy_from(&data), Validity::NonNullable).into_array()
}

fn build_slice(len: usize) -> (ArrayRef, Scalar) {
    let inner = build_sorted_primitive(len * 2);
    let slice = SliceArray::new(inner, (len / 2)..(len / 2 + len)).into_array();
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let target = slice.execute_scalar(len / 3, &mut ctx).unwrap();
    (slice, target)
}

#[divan::bench(args = SIZES)]
fn slice_legacy(bencher: Bencher, &len: &usize) {
    let (arr, target) = build_slice(len);
    bencher
        .with_inputs(|| (&arr, &target))
        .bench_refs(|(arr, target)| arr.search_sorted(target, SearchSortedSide::Left).unwrap());
}

#[divan::bench(args = SIZES)]
fn slice_session(bencher: Bencher, &len: &usize) {
    let (arr, target) = build_slice(len);
    bencher
        .with_inputs(|| (&arr, &target))
        .bench_refs(|(arr, target)| {
            let mut ctx = LEGACY_SESSION.create_execution_ctx();
            let mut session = PointSession::new(&mut ctx);
            session
                .search_sorted(arr, target, SearchSortedSide::Left)
                .unwrap()
        });
}

// ────────────────────────────────────────────────────────────────────────────
// Dict (sorted dict + sorted codes)
// ────────────────────────────────────────────────────────────────────────────

/// Build a sorted Dict array: tiny dict (~64 unique values), large codes
/// array distributing those values across `len` rows in sorted order.
fn build_sorted_dict(len: usize) -> (ArrayRef, Scalar) {
    const DICT_SIZE: usize = 64;
    let mut rng = StdRng::seed_from_u64(0);
    let range = Uniform::new(0i32, i32::MAX / 2).unwrap();

    let mut dict_values: Vec<i32> = (0..DICT_SIZE).map(|_| rng.sample(range)).collect();
    dict_values.sort();
    dict_values.dedup();
    let actual_dict_size = dict_values.len();

    // Distribute `len` rows across the dict entries roughly evenly, in sorted order.
    let mut codes: Vec<u32> = Vec::with_capacity(len);
    for i in 0..len {
        let code = (i * actual_dict_size / len) as u32;
        codes.push(code.min(actual_dict_size as u32 - 1));
    }

    let dict =
        PrimitiveArray::new(Buffer::copy_from(&dict_values), Validity::NonNullable).into_array();
    let codes_arr =
        PrimitiveArray::new(Buffer::copy_from(&codes), Validity::NonNullable).into_array();

    let arr = DictArray::try_new(codes_arr, dict).unwrap().into_array();
    let target = Scalar::from(dict_values[actual_dict_size / 3]);
    (arr, target)
}

#[divan::bench(args = SIZES)]
fn dict_legacy(bencher: Bencher, &len: &usize) {
    let (arr, target) = build_sorted_dict(len);
    bencher
        .with_inputs(|| (&arr, &target))
        .bench_refs(|(arr, target)| arr.search_sorted(target, SearchSortedSide::Left).unwrap());
}

#[divan::bench(args = SIZES)]
fn dict_session(bencher: Bencher, &len: &usize) {
    let (arr, target) = build_sorted_dict(len);
    bencher
        .with_inputs(|| (&arr, &target))
        .bench_refs(|(arr, target)| {
            let mut ctx = LEGACY_SESSION.create_execution_ctx();
            let mut session = PointSession::new(&mut ctx);
            session
                .search_sorted(arr, target, SearchSortedSide::Left)
                .unwrap()
        });
}

// ────────────────────────────────────────────────────────────────────────────
// Chunked (cross-chunk monotonic)
// ────────────────────────────────────────────────────────────────────────────

/// Build a chunked array of `nchunks` sorted chunks, cross-chunk monotonic.
/// Total logical length ≈ `len`.
fn build_sorted_chunked(len: usize, nchunks: usize) -> (ArrayRef, Scalar) {
    let chunk_size = len / nchunks;
    let mut rng = StdRng::seed_from_u64(0);
    let range = Uniform::new(0i32, i32::MAX / 2).unwrap();
    let mut all: Vec<i32> = (0..(chunk_size * nchunks))
        .map(|_| rng.sample(range))
        .collect();
    all.sort();

    let chunks: Vec<ArrayRef> = (0..nchunks)
        .map(|i| {
            let start = i * chunk_size;
            let end = start + chunk_size;
            PrimitiveArray::new(Buffer::copy_from(&all[start..end]), Validity::NonNullable)
                .into_array()
        })
        .collect();

    let arr = ChunkedArray::try_new(
        chunks,
        DType::Primitive(PType::I32, Nullability::NonNullable),
    )
    .unwrap()
    .into_array();

    // Target near the middle.
    let target = Scalar::from(all[all.len() / 3]);
    (arr, target)
}

#[divan::bench(args = SIZES)]
fn chunked_legacy(bencher: Bencher, &len: &usize) {
    let (arr, target) = build_sorted_chunked(len, 16);
    bencher
        .with_inputs(|| (&arr, &target))
        .bench_refs(|(arr, target)| arr.search_sorted(target, SearchSortedSide::Left).unwrap());
}

#[divan::bench(args = SIZES)]
fn chunked_session(bencher: Bencher, &len: &usize) {
    let (arr, target) = build_sorted_chunked(len, 16);
    bencher
        .with_inputs(|| (&arr, &target))
        .bench_refs(|(arr, target)| {
            let mut ctx = LEGACY_SESSION.create_execution_ctx();
            let mut session = PointSession::new(&mut ctx);
            session
                .search_sorted(arr, target, SearchSortedSide::Left)
                .unwrap()
        });
}
