// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks for the `list_sum` scalar function over `List`, `ListView`, and
//! `FixedSizeList` inputs.

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation)]

use std::sync::LazyLock;

use divan::Bencher;
use rand::RngExt;
use rand::SeedableRng;
use rand::distr::Uniform;
use rand::rngs::StdRng;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::expr::list_sum;
use vortex_array::expr::root;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_session::VortexSession;

fn main() {
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(vortex_array::array_session);

const BASE_LIST_SIZE: usize = 8;

// Sized to keep the CodSpeed simulation under 1ms per benchmark.
const SMALL: usize = 100;
const MEDIUM: usize = 250;
const LARGE: usize = 500;

/// A uniformly-random partition of `num_lists * BASE_LIST_SIZE` elements into `num_lists`
/// lists, plus a validity mask with ~1/8 of lists null at random positions.
fn random_lists(num_lists: usize) -> (Vec<i32>, Validity) {
    let mut rng = StdRng::seed_from_u64(num_lists as u64);
    let total = (num_lists * BASE_LIST_SIZE) as i32;

    let cut_dist = Uniform::new_inclusive(0i32, total).unwrap();
    let mut cuts: Vec<i32> = (0..num_lists - 1).map(|_| rng.sample(cut_dist)).collect();
    cuts.sort_unstable();
    let mut sizes = Vec::with_capacity(num_lists);
    let mut prev = 0i32;
    for cut in cuts {
        sizes.push(cut - prev);
        prev = cut;
    }
    sizes.push(total - prev);

    let null_dist = Uniform::new(0u32, 8).unwrap();
    let valid = (0..num_lists).map(|_| rng.sample(null_dist) != 0);
    (
        sizes,
        Validity::Array(BoolArray::from_iter(valid).into_array()),
    )
}

/// `0..total` as `i32` elements, with ~1/8 nulled at random positions when `nullable`.
fn make_elements(total: i32, nullable: bool) -> ArrayRef {
    if !nullable {
        return PrimitiveArray::from_iter(0..total).into_array();
    }
    let mut rng = StdRng::seed_from_u64(total as u64);
    let null_dist = Uniform::new(0u32, 8).unwrap();
    PrimitiveArray::from_option_iter((0..total).map(|v| (rng.sample(null_dist) != 0).then_some(v)))
        .into_array()
}

/// A canonical `List<i32>` of `num_lists` variable-length lists, ~1/8 of them null.
fn make_list(num_lists: usize, nullable_elements: bool) -> ArrayRef {
    let (sizes, validity) = random_lists(num_lists);
    let total: i32 = sizes.iter().sum();
    let elements = make_elements(total, nullable_elements);
    let offsets: Buffer<i32> = std::iter::once(0)
        .chain(sizes.iter().scan(0i32, |acc, &s| {
            *acc += s;
            Some(*acc)
        }))
        .collect();
    ListArray::try_new(elements, offsets.into_array(), validity)
        .unwrap()
        .into_array()
}

/// A gapless `ListView<i32>` of `num_lists` variable-length lists, ~1/8 of them null.
fn make_listview(num_lists: usize) -> ArrayRef {
    let (sizes, validity) = random_lists(num_lists);
    let total: i32 = sizes.iter().sum();
    let elements = make_elements(total, false);
    let offsets: Buffer<i32> = sizes
        .iter()
        .scan(0i32, |acc, &s| {
            let start = *acc;
            *acc += s;
            Some(start)
        })
        .collect();
    let sizes: Buffer<i32> = sizes.into_iter().collect();
    ListViewArray::new(elements, offsets.into_array(), sizes.into_array(), validity).into_array()
}

/// A `FixedSizeList<i32, 8>` of `num_lists` lists, ~1/8 of them null.
fn make_fsl(num_lists: usize) -> ArrayRef {
    let mut rng = StdRng::seed_from_u64(num_lists as u64);
    let total = (num_lists * BASE_LIST_SIZE) as i32;
    let elements = make_elements(total, false);
    let null_dist = Uniform::new(0u32, 8).unwrap();
    let valid = (0..num_lists).map(|_| rng.sample(null_dist) != 0);
    let validity = Validity::Array(BoolArray::from_iter(valid).into_array());
    FixedSizeListArray::new(elements, BASE_LIST_SIZE as u32, validity, num_lists).into_array()
}

/// Apply `list_sum(root())` and materialize the result.
fn run(bencher: Bencher, array: ArrayRef) {
    let expr = list_sum(root());
    bencher
        .with_inputs(|| (&array, SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| {
            array
                .clone()
                .apply(&expr)
                .unwrap()
                .execute::<Canonical>(ctx)
                .unwrap()
        });
}

#[divan::bench]
fn list_sum_small(bencher: Bencher) {
    run(bencher, make_list(SMALL, false));
}

#[divan::bench]
fn list_sum_medium(bencher: Bencher) {
    run(bencher, make_list(MEDIUM, false));
}

#[divan::bench]
fn list_sum_large(bencher: Bencher) {
    run(bencher, make_list(LARGE, false));
}

#[divan::bench]
fn list_sum_nullable_elements_medium(bencher: Bencher) {
    run(bencher, make_list(MEDIUM, true));
}

#[divan::bench]
fn list_sum_nullable_elements_large(bencher: Bencher) {
    run(bencher, make_list(LARGE, true));
}

#[divan::bench]
fn listview_sum_small(bencher: Bencher) {
    run(bencher, make_listview(SMALL));
}

#[divan::bench]
fn listview_sum_medium(bencher: Bencher) {
    run(bencher, make_listview(MEDIUM));
}

#[divan::bench]
fn listview_sum_large(bencher: Bencher) {
    run(bencher, make_listview(LARGE));
}

#[divan::bench]
fn fsl_sum_small(bencher: Bencher) {
    run(bencher, make_fsl(SMALL));
}

#[divan::bench]
fn fsl_sum_medium(bencher: Bencher) {
    run(bencher, make_fsl(MEDIUM));
}

#[divan::bench]
fn fsl_sum_large(bencher: Bencher) {
    run(bencher, make_fsl(LARGE));
}
