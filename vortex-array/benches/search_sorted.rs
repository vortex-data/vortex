// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]

use divan::Bencher;
use rand::RngExt;
use rand::SeedableRng;
use rand::distr::Uniform;
use rand::prelude::StdRng;
use vortex_array::search_sorted::SearchSorted;
use vortex_array::search_sorted::SearchSortedSide;

fn main() {
    divan::main();
}

#[divan::bench]
fn binary_search_std(bencher: Bencher) {
    let (sorted_array, target) = fixture();
    bencher
        .with_inputs(|| (&sorted_array, &target))
        .bench_refs(|(array, target)| array.binary_search(target));
}

#[divan::bench]
fn binary_search_vortex(bencher: Bencher) {
    let (sorted_array, target) = fixture();
    bencher
        .with_inputs(|| (&sorted_array, &target))
        .bench_refs(|(array, target)| array.search_sorted(target, SearchSortedSide::Left).unwrap());
}

fn fixture() -> (Vec<i32>, i32) {
    let mut rng = StdRng::seed_from_u64(0);
    let range = Uniform::new(0, 65_536).unwrap();
    let mut data: Vec<i32> = (0..65_536).map(|_| rng.sample(range)).collect();
    data.sort();

    (data, rng.sample(range))
}
