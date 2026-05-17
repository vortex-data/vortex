// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmark `search_sorted` on a RunEnd-encoded sorted array, comparing:
//!   - legacy path (generic binary search calling `execute_scalar` per probe,
//!     which in turn calls `find_physical_index` on the ends array)
//!   - PointSession path with `point_search_sorted` override that descends
//!     directly into `values` and resolves the run boundary via `ends`.
//!
//! Phase 2d: this validates the structural search_sorted win for RunEnd —
//! `O(log num_runs)` vs `O(log n × log num_runs)`.

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation)]
#![expect(clippy::cast_precision_loss)]

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
use vortex_array::scalar::Scalar;
use vortex_array::search_sorted::SearchSorted;
use vortex_array::search_sorted::SearchSortedSide;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_runend::RunEnd;

fn main() {
    divan::main();
}

// Parameter pairs: (logical_len, num_runs).
const PARAMS: &[(usize, usize)] = &[(1024, 64), (16_384, 256), (262_144, 1024)];

/// Build a sorted RunEnd-encoded array of the given logical length with
/// approximately the given number of runs. Values are monotonically
/// nondecreasing, so the array is sorted.
fn build_sorted_runend(logical_len: usize, num_runs: usize) -> (ArrayRef, Scalar) {
    let mut rng = StdRng::seed_from_u64(0);
    let range = Uniform::new(0i32, i32::MAX / 2).unwrap();

    // Generate `num_runs` sorted distinct values.
    let mut values: Vec<i32> = (0..num_runs).map(|_| rng.sample(range)).collect();
    values.sort();
    values.dedup();

    // Distribute logical_len across the runs roughly evenly.
    let mut ends: Vec<u32> = Vec::with_capacity(values.len());
    for i in 1..=values.len() {
        let e = (logical_len as f64 * i as f64 / values.len() as f64).round() as u32;
        ends.push(e.max(*ends.last().unwrap_or(&0) + 1));
    }
    if let Some(last) = ends.last_mut() {
        *last = logical_len as u32;
    }

    let ends_arr =
        PrimitiveArray::new(Buffer::copy_from(&ends), Validity::NonNullable).into_array();
    let values_arr =
        PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable).into_array();

    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let runend = RunEnd::try_new(ends_arr, values_arr, &mut ctx).unwrap();

    // Pick a target value from the middle of the run.
    let target = Scalar::from(values[values.len() / 3]);
    (runend.into_array(), target)
}

#[divan::bench(args = PARAMS)]
fn legacy_search_sorted(bencher: Bencher, &(len, runs): &(usize, usize)) {
    let (arr, target) = build_sorted_runend(len, runs);
    bencher
        .with_inputs(|| (&arr, &target))
        .bench_refs(|(arr, target)| arr.search_sorted(target, SearchSortedSide::Left).unwrap());
}

#[divan::bench(args = PARAMS)]
fn repeated_access_search_sorted(bencher: Bencher, &(len, runs): &(usize, usize)) {
    let (arr, target) = build_sorted_runend(len, runs);
    bencher
        .with_inputs(|| (&arr, &target))
        .bench_refs(|(arr, target)| {
            let mut ctx = LEGACY_SESSION.create_execution_ctx();
            arr.repeated_access(&mut ctx)
                .search_sorted(target, SearchSortedSide::Left)
                .unwrap()
        });
}
