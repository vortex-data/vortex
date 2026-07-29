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

/// One search over 65536 elements is ~16 comparisons, small enough that the measurement is mostly
/// harness overhead and the two implementations below become indistinguishable. Search a batch per
/// iteration instead; the targets differ, so the calls cannot be folded together.
const SEARCH_TARGETS: usize = 64;

#[divan::bench]
fn binary_search_std(bencher: Bencher) {
    let (sorted_array, targets) = fixture();
    bencher
        .with_inputs(|| (&sorted_array, &targets))
        .bench_refs(|(array, targets)| {
            let mut found = 0;
            for target in targets.iter() {
                found += array.binary_search(target).unwrap_or_else(|idx| idx);
            }
            found
        });
}

#[divan::bench]
fn binary_search_vortex(bencher: Bencher) {
    let (sorted_array, targets) = fixture();
    bencher
        .with_inputs(|| (&sorted_array, &targets))
        .bench_refs(|(array, targets)| {
            let mut found = 0;
            for target in targets.iter() {
                found += array
                    .search_sorted(target, SearchSortedSide::Left)
                    .unwrap()
                    .to_index();
            }
            found
        });
}

fn fixture() -> (Vec<i32>, Vec<i32>) {
    let mut rng = StdRng::seed_from_u64(0);
    let range = Uniform::new(0, 65_536).unwrap();
    let mut data: Vec<i32> = (0..65_536).map(|_| rng.sample(range)).collect();
    data.sort();

    let targets = (0..SEARCH_TARGETS).map(|_| rng.sample(range)).collect();

    (data, targets)
}
