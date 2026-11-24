// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks for the `list_transform` scalar function over `List` and `ListView` inputs, alone
//! and as a chain of transforms, each over the result of the one before.

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
use vortex_array::IntoArray;
use vortex_array::RecursiveCanonical;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::session::AggregateFnSession;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::session::DTypeSession;
use vortex_array::expr::Expression;
use vortex_array::expr::Lambda;
use vortex_array::expr::checked_add;
use vortex_array::expr::lit;
use vortex_array::expr::root;
use vortex_array::expr::var;
use vortex_array::memory::MemorySession;
use vortex_array::optimizer::kernels::KernelSession;
use vortex_array::scalar_fn::session::ScalarFnSession;
use vortex_array::session::ArraySession;
use vortex_array::test_harness::mark_optimized;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_functions::expr::list_transform;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty()
        .with::<ArraySession>()
        .with::<KernelSession>()
        .with::<DTypeSession>()
        .with::<ScalarFnSession>()
        .with::<AggregateFnSession>()
        .with::<MemorySession>();
    vortex_functions::initialize(&session);
    session
});

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

/// A canonical `List<i32>` of `num_lists` variable-length lists, ~1/8 of them null, over the
/// elements `0..total`.
fn make_list(num_lists: usize) -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    let (sizes, validity) = random_lists(num_lists);
    let total: i32 = sizes.iter().sum();
    let offsets: Buffer<i32> = std::iter::once(0)
        .chain(sizes.iter().scan(0i32, |acc, &s| {
            *acc += s;
            Some(*acc)
        }))
        .collect();
    ListArray::try_new(
        PrimitiveArray::from_iter(0..total).into_array(),
        offsets.into_array(),
        validity,
        &mut ctx,
    )
    .unwrap()
    .into_array()
}

/// A gapless `ListView<i32>` of `num_lists` variable-length lists, ~1/8 of them null, over the
/// elements `0..total`.
fn make_listview(num_lists: usize) -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    let (sizes, validity) = random_lists(num_lists);
    let total: i32 = sizes.iter().sum();
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
        PrimitiveArray::from_iter(0..total).into_array(),
        offsets.into_array(),
        sizes.into_array(),
        validity,
        &mut ctx,
    )
    .into_array()
}

/// `depth` transforms of `x -> x + 1`, each over the result of the one before. The lambda can
/// overflow, so every transform has to keep it off the elements of null lists.
fn transforms(depth: usize) -> Expression {
    (0..depth).fold(root(), |list, _| {
        let plus_one = Lambda::new(["x"], checked_add(var("x"), lit(1_i32))).unwrap();
        list_transform(list, plus_one)
    })
}

/// Apply `depth` transforms and materialize the lists and their elements.
fn run(bencher: Bencher, array: ArrayRef, depth: usize) {
    mark_optimized(&array, &SESSION).unwrap();

    let expr = transforms(depth);
    bencher
        .with_inputs(|| (&array, SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| {
            array
                .clone()
                .apply(&expr)
                .unwrap()
                .execute::<RecursiveCanonical>(ctx)
                .unwrap()
        });
}

#[divan::bench]
fn list_transform_small(bencher: Bencher) {
    run(bencher, make_list(SMALL), 1);
}

#[divan::bench]
fn list_transform_medium(bencher: Bencher) {
    run(bencher, make_list(MEDIUM), 1);
}

#[divan::bench]
fn list_transform_large(bencher: Bencher) {
    run(bencher, make_list(LARGE), 1);
}

#[divan::bench]
fn listview_transform_medium(bencher: Bencher) {
    run(bencher, make_listview(MEDIUM), 1);
}

#[divan::bench]
fn list_two_transforms_small(bencher: Bencher) {
    run(bencher, make_list(SMALL), 2);
}

#[divan::bench]
fn list_two_transforms_medium(bencher: Bencher) {
    run(bencher, make_list(MEDIUM), 2);
}

#[divan::bench]
fn list_two_transforms_large(bencher: Bencher) {
    run(bencher, make_list(LARGE), 2);
}

#[divan::bench]
fn listview_two_transforms_medium(bencher: Bencher) {
    run(bencher, make_listview(MEDIUM), 2);
}

#[divan::bench]
fn list_four_transforms_medium(bencher: Bencher) {
    run(bencher, make_list(MEDIUM), 4);
}
