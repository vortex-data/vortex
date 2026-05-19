// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Decompression-with-filter benches across stacks beyond Dict(RunEnd).
//!
//! The motivating workload is "take first 10 elements of each row from a list column,
//! then sum" - common in array/JSON workloads. We test:
//!
//! 1. **Sum aggregate on RunEnd<i64>** - does pushdown beat decode-then-dense-sum?
//! 2. **Strided take on RunEnd<i64>** (first-10-per-row pattern from a ListView):
//!    take 10 elements out of every 100, total `N_TOTAL` rows. Take indices known up front.
//! 3. **Strided take on Dict(RunEnd)** - same, but elements live in a dict.
//! 4. **Predicate on RunEnd<i64>** as a baseline.
//!
//! `target-cpu=native` for AVX-512 in the hot paths.

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation)]

use divan::Bencher;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::fns::sum::sum;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_runend::RunEnd;

fn main() {
    divan::main();
}

const N_ELEMENTS: usize = 1_000_000;
const ROW_LEN: usize = 100;
const TAKE_PER_ROW: usize = 10;
const NUM_ROWS: usize = N_ELEMENTS / ROW_LEN; // 10_000 rows
const DICT_SIZE: usize = 64;

// =====================================================================================
// Builders
// =====================================================================================

/// Build a `RunEnd<i64>` of `n` elements with run length `r`.
/// Run values cycle through a small set (so sums stay deterministic, not too large).
fn build_runend_i64(n: usize, r: usize) -> vortex_array::ArrayRef {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let num_runs = n.div_ceil(r);
    let ends: Buffer<u32> = (1..=num_runs)
        .map(|i| std::cmp::min(i * r, n) as u32)
        .collect();
    let values: Buffer<i64> = (0..num_runs).map(|i| (i % 64) as i64).collect();
    let ends_arr = PrimitiveArray::new(ends, Validity::NonNullable).into_array();
    let values_arr = PrimitiveArray::new(values, Validity::NonNullable).into_array();
    RunEnd::new(ends_arr, values_arr, &mut ctx).into_array()
}

/// Build `Dict(Primitive<i64>[64], RunEnd<u8>)` of `n` rows with the codes RunEnd-encoded
/// at run length `r`.
fn build_dict_runend_i64(n: usize, r: usize) -> vortex_array::ArrayRef {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let num_runs = n.div_ceil(r);
    let ends: Buffer<u32> = (1..=num_runs)
        .map(|i| std::cmp::min(i * r, n) as u32)
        .collect();
    let runend_codes: Buffer<u8> = (0..num_runs).map(|i| (i % DICT_SIZE) as u8).collect();
    let ends_arr = PrimitiveArray::new(ends, Validity::NonNullable).into_array();
    let runend_codes_arr =
        PrimitiveArray::new(runend_codes, Validity::NonNullable).into_array();
    let codes = RunEnd::new(ends_arr, runend_codes_arr, &mut ctx).into_array();
    let dict_values: Buffer<i64> = (0..DICT_SIZE as i64).collect();
    let dict_values_arr = PrimitiveArray::new(dict_values, Validity::NonNullable).into_array();
    DictArray::new(codes, dict_values_arr).into_array()
}

/// Build the "first 10 of each row" index pattern: for each of `num_rows`,
/// indices `row * row_len + 0..take_per_row`. Total `num_rows * take_per_row` indices.
fn build_strided_indices() -> vortex_array::ArrayRef {
    let indices: Buffer<u32> = (0..NUM_ROWS as u32)
        .flat_map(|row| (0..TAKE_PER_ROW as u32).map(move |k| row * ROW_LEN as u32 + k))
        .collect();
    PrimitiveArray::new(indices, Validity::NonNullable).into_array()
}

// =====================================================================================
// Sum aggregate on RunEnd<i64>
// =====================================================================================

const RUN_LENGTHS: &[usize] = &[10, 100];

// Note: `sum()` caches `Stat::Sum` on the array's statistics on first compute and
// short-circuits on subsequent calls. We rebuild the array per-iter on the tree path so
// each measurement is a fresh first-compute. (The canonical path is unaffected because
// `execute::<Canonical>()` returns a fresh array each call, so its stats are never reused.)

#[divan::bench(args = RUN_LENGTHS, sample_count = 30)]
fn sum_runend_tree(bencher: Bencher, r: usize) {
    bencher
        .with_inputs(|| {
            (
                build_runend_i64(N_ELEMENTS, r),
                LEGACY_SESSION.create_execution_ctx(),
            )
        })
        .bench_values(|(col, mut ctx)| sum(&col, &mut ctx).unwrap());
}

#[divan::bench(args = RUN_LENGTHS, sample_count = 30)]
fn sum_runend_canonical(bencher: Bencher, r: usize) {
    bencher
        .with_inputs(|| {
            (
                build_runend_i64(N_ELEMENTS, r),
                LEGACY_SESSION.create_execution_ctx(),
            )
        })
        .bench_values(|(col, mut ctx)| {
            let canon = col.execute::<Canonical>(&mut ctx).unwrap().into_array();
            sum(&canon, &mut ctx).unwrap()
        });
}

#[divan::bench(args = RUN_LENGTHS, sample_count = 30)]
fn sum_dict_runend_tree(bencher: Bencher, r: usize) {
    bencher
        .with_inputs(|| {
            (
                build_dict_runend_i64(N_ELEMENTS, r),
                LEGACY_SESSION.create_execution_ctx(),
            )
        })
        .bench_values(|(col, mut ctx)| sum(&col, &mut ctx).unwrap());
}

#[divan::bench(args = RUN_LENGTHS, sample_count = 30)]
fn sum_dict_runend_canonical(bencher: Bencher, r: usize) {
    bencher
        .with_inputs(|| {
            (
                build_dict_runend_i64(N_ELEMENTS, r),
                LEGACY_SESSION.create_execution_ctx(),
            )
        })
        .bench_values(|(col, mut ctx)| {
            let canon = col.execute::<Canonical>(&mut ctx).unwrap().into_array();
            sum(&canon, &mut ctx).unwrap()
        });
}

// =====================================================================================
// Strided take + sum: "first 10 of each row"
// =====================================================================================

/// Tree path: take strided indices from RunEnd column, then sum the result.
#[divan::bench(args = RUN_LENGTHS, sample_count = 30)]
fn first_n_runend_tree(bencher: Bencher, r: usize) {
    let col = build_runend_i64(N_ELEMENTS, r);
    let indices = build_strided_indices();
    bencher
        .with_inputs(|| {
            (
                col.clone(),
                indices.clone(),
                LEGACY_SESSION.create_execution_ctx(),
            )
        })
        .bench_values(|(col, indices, mut ctx)| {
            let taken = col.take(indices).unwrap();
            sum(&taken, &mut ctx).unwrap()
        });
}

/// All-in-one path: canonicalise RunEnd to dense, then strided take, then sum.
#[divan::bench(args = RUN_LENGTHS, sample_count = 30)]
fn first_n_runend_canonical(bencher: Bencher, r: usize) {
    let col = build_runend_i64(N_ELEMENTS, r);
    let indices = build_strided_indices();
    bencher
        .with_inputs(|| {
            (
                col.clone(),
                indices.clone(),
                LEGACY_SESSION.create_execution_ctx(),
            )
        })
        .bench_values(|(col, indices, mut ctx)| {
            let canon = col.execute::<Canonical>(&mut ctx).unwrap().into_array();
            let taken = canon.take(indices).unwrap();
            sum(&taken, &mut ctx).unwrap()
        });
}

/// Tree path on Dict(RunEnd): take strided indices, then sum.
#[divan::bench(args = RUN_LENGTHS, sample_count = 30)]
fn first_n_dict_runend_tree(bencher: Bencher, r: usize) {
    let col = build_dict_runend_i64(N_ELEMENTS, r);
    let indices = build_strided_indices();
    bencher
        .with_inputs(|| {
            (
                col.clone(),
                indices.clone(),
                LEGACY_SESSION.create_execution_ctx(),
            )
        })
        .bench_values(|(col, indices, mut ctx)| {
            let taken = col.take(indices).unwrap();
            sum(&taken, &mut ctx).unwrap()
        });
}

/// All-in-one on Dict(RunEnd): canonicalise, take, sum.
#[divan::bench(args = RUN_LENGTHS, sample_count = 30)]
fn first_n_dict_runend_canonical(bencher: Bencher, r: usize) {
    let col = build_dict_runend_i64(N_ELEMENTS, r);
    let indices = build_strided_indices();
    bencher
        .with_inputs(|| {
            (
                col.clone(),
                indices.clone(),
                LEGACY_SESSION.create_execution_ctx(),
            )
        })
        .bench_values(|(col, indices, mut ctx)| {
            let canon = col.execute::<Canonical>(&mut ctx).unwrap().into_array();
            let taken = canon.take(indices).unwrap();
            sum(&taken, &mut ctx).unwrap()
        });
}

// Note: a `predicate_runend_tree` bench with `i64` values needs `RUST_MIN_STACK=8388608`
// to avoid stack overflow on the small worker thread divan uses. Root cause: the i64
// RunEnd compare path uses larger stack frames than i32 and overflows divan's default
// 2MB thread stack. Pass `RUST_MIN_STACK=8388608` when running, or fix Vortex's i64
// compare to use a smaller stack.
//
// The i32 predicate (`block_execution.rs::predicate_dict_runend_chunked`) runs without
// the env var.

#[divan::bench(args = [10], sample_count = 5)]
fn predicate_runend_tree_i64(bencher: Bencher, r: usize) {
    use vortex_array::arrays::ConstantArray;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::scalar_fn::fns::operators::Operator;
    bencher
        .with_inputs(|| {
            (
                build_runend_i64(N_ELEMENTS, r),
                LEGACY_SESSION.create_execution_ctx(),
            )
        })
        .bench_values(|(col, mut ctx)| {
            let five = ConstantArray::new(5i64, N_ELEMENTS).into_array();
            let cmp = col.binary(five, Operator::Gt).unwrap();
            cmp.execute::<Canonical>(&mut ctx).unwrap()
        });
}
