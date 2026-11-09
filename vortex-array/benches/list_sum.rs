// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks for the `list_sum` scalar function over `List`, `ListView`, and
//! `FixedSizeList` inputs, and over lists of bool, run-end and chunked elements.

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation)]

use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use rand::RngExt;
use rand::SeedableRng;
use rand::distr::Uniform;
use rand::rngs::StdRng;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::session::AggregateFnSession;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::RunEnd;
use vortex_array::expr::list_sum;
use vortex_array::expr::root;
use vortex_array::memory::MemorySession;
use vortex_array::optimizer::kernels::KernelSession;
use vortex_array::test_harness::mark_optimized;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    VortexSession::empty()
        .with::<KernelSession>()
        .with::<AggregateFnSession>()
        .with::<MemorySession>()
});

const BASE_LIST_SIZE: usize = 8;

// Sized to keep the CodSpeed simulation under 1ms per benchmark.
const SMALL: usize = 100;
const MEDIUM: usize = 250;
const LARGE: usize = 500;

/// Average list lengths: short lists weigh per-list overhead, long lists per-element work. The
/// longest is about the length of statpopgen's genotype lists.
const LIST_SIZES: &[usize] = &[8, 256, 4_096];
/// Element chunks in each chunked benchmark.
const NUM_CHUNKS: usize = 4;

/// Elements for lists of `list_size`: at least 4096, and at least 16 lists.
fn num_elements(list_size: usize) -> usize {
    (16 * list_size).max(4_096)
}

/// A uniformly-random partition of `total` elements into `num_lists` lists, plus a validity
/// mask with ~1/8 of lists null at random positions.
fn random_lists(num_lists: usize, total: usize) -> (Vec<i32>, Validity) {
    let mut rng = StdRng::seed_from_u64(num_lists as u64);
    let total = total as i32;

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
    let elements = make_elements((num_lists * BASE_LIST_SIZE) as i32, nullable_elements);
    list_over(elements, BASE_LIST_SIZE)
}

/// A `List` over `elements` with `list_size` elements per list on average, ~1/8 of lists null.
fn list_over(elements: ArrayRef, list_size: usize) -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    let (sizes, validity) = random_lists(elements.len() / list_size, elements.len());
    let offsets: Buffer<i32> = std::iter::once(0)
        .chain(sizes.iter().scan(0i32, |acc, &s| {
            *acc += s;
            Some(*acc)
        }))
        .collect();
    ListArray::try_new(elements, offsets.into_array(), validity, &mut ctx)
        .unwrap()
        .into_array()
}

/// A gapless `ListView<i32>` of `num_lists` variable-length lists, ~1/8 of them null.
fn make_listview(num_lists: usize) -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    let (sizes, validity) = random_lists(num_lists, num_lists * BASE_LIST_SIZE);
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
    ListViewArray::new(
        elements,
        offsets.into_array(),
        sizes.into_array(),
        validity,
        &mut ctx,
    )
    .into_array()
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

/// `len` random bools, ~1/8 null.
fn bool_elements(len: usize, seed: u64) -> ArrayRef {
    let mut rng = StdRng::seed_from_u64(seed);
    let null_dist = Uniform::new(0u32, 8).unwrap();
    BoolArray::from_iter(
        (0..len).map(|_| (rng.sample(null_dist) != 0).then(|| rng.random_bool(0.5))),
    )
    .into_array()
}

/// `len` run-end encoded bools in runs of 1 to 32, ~1/8 of runs null.
fn run_end_bool_elements(len: usize, seed: u64) -> ArrayRef {
    let mut rng = StdRng::seed_from_u64(seed);
    let run_dist = Uniform::new_inclusive(1usize, 32).unwrap();
    let mut ends = Vec::new();
    let mut end = 0;
    while end < len {
        end = (end + rng.sample(run_dist)).min(len);
        ends.push(end as u32);
    }
    let values = bool_elements(ends.len(), seed);
    let ends: Buffer<u32> = ends.into_iter().collect();
    RunEnd::new(
        ends.into_array(),
        values,
        &mut SESSION.create_execution_ctx(),
    )
    .into_array()
}

/// `len` elements in `NUM_CHUNKS` chunks, each built by `chunk(chunk_len, seed)`.
fn chunked(len: usize, chunk: impl Fn(usize, u64) -> ArrayRef) -> ArrayRef {
    let chunk_len = len.div_ceil(NUM_CHUNKS);
    let chunks: Vec<ArrayRef> = (0..len)
        .step_by(chunk_len)
        .map(|start| chunk(chunk_len.min(len - start), start as u64))
        .collect();
    let dtype = chunks[0].dtype().clone();
    ChunkedArray::try_new(chunks, dtype).unwrap().into_array()
}

/// Apply `list_sum(root())` and materialize the result.
fn run(bencher: Bencher, array: ArrayRef) {
    mark_optimized(&array, &SESSION).unwrap();

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

#[divan::bench(args = LIST_SIZES)]
fn bool_sum(bencher: Bencher, list_size: usize) {
    run(
        bencher,
        list_over(bool_elements(num_elements(list_size), 0), list_size),
    );
}

#[divan::bench(args = LIST_SIZES)]
fn run_end_bool_sum(bencher: Bencher, list_size: usize) {
    run(
        bencher,
        list_over(run_end_bool_elements(num_elements(list_size), 0), list_size),
    );
}

#[divan::bench(args = LIST_SIZES)]
fn chunked_i32_sum(bencher: Bencher, list_size: usize) {
    let elements = chunked(num_elements(list_size), |len, _| {
        make_elements(len as i32, true)
    });
    run(bencher, list_over(elements, list_size));
}

#[divan::bench(args = LIST_SIZES)]
fn chunked_bool_sum(bencher: Bencher, list_size: usize) {
    run(
        bencher,
        list_over(chunked(num_elements(list_size), bool_elements), list_size),
    );
}

#[divan::bench(args = LIST_SIZES)]
fn chunked_run_end_bool_sum(bencher: Bencher, list_size: usize) {
    run(
        bencher,
        list_over(
            chunked(num_elements(list_size), run_end_bool_elements),
            list_size,
        ),
    );
}
