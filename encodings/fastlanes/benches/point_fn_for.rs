// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmark `search_sorted` on a FoR-encoded sorted array.
//!
//! Compares the legacy generic binary search (per-probe add of reference)
//! against PointSession's `point_search_sorted` override (one-shot subtract
//! of reference, then push to encoded). Phase 2e.

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
use vortex_array::scalar::Scalar;
use vortex_array::search_sorted::SearchSorted;
use vortex_array::search_sorted::SearchSortedSide;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_fastlanes::FoRData;

fn main() {
    divan::main();
}

const SIZES: &[usize] = &[1024, 16_384, 262_144];

fn build_sorted_for(len: usize) -> (ArrayRef, Scalar) {
    // Generate values in a tight range so FoR compression is meaningful.
    // base = ~1_000_000, deltas in [0, 65536).
    let base: i32 = 1_000_000;
    let mut rng = StdRng::seed_from_u64(0);
    let range = Uniform::new(0i32, 65_536).unwrap();
    let mut data: Vec<i32> = (0..len).map(|_| base + rng.sample(range)).collect();
    data.sort();
    let target = Scalar::from(data[len / 3]);
    let primitive = PrimitiveArray::new(Buffer::copy_from(&data), Validity::NonNullable);
    let for_arr = FoRData::encode(primitive).unwrap().into_array();
    (for_arr, target)
}

#[divan::bench(args = SIZES)]
fn for_legacy(bencher: Bencher, &len: &usize) {
    let (arr, target) = build_sorted_for(len);
    bencher
        .with_inputs(|| (&arr, &target))
        .bench_refs(|(arr, target)| arr.search_sorted(target, SearchSortedSide::Left).unwrap());
}

#[divan::bench(args = SIZES)]
fn for_repeated_access(bencher: Bencher, &len: &usize) {
    let (arr, target) = build_sorted_for(len);
    bencher
        .with_inputs(|| (&arr, &target))
        .bench_refs(|(arr, target)| {
            let mut ctx = LEGACY_SESSION.create_execution_ctx();
            arr.repeated_access(&mut ctx)
                .search_sorted(target, SearchSortedSide::Left)
                .unwrap()
        });
}
