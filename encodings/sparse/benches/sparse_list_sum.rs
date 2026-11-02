// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks for `list_sum` over lists of sparse bool elements, the shape a comparison on a
//! sparse column produces.

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation)]

use std::iter::once;
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
use vortex_array::arrays::ListArray;
use vortex_array::dtype::Nullability;
use vortex_array::expr::list_sum;
use vortex_array::expr::root;
use vortex_array::memory::MemorySession;
use vortex_array::optimizer::kernels::KernelSession;
use vortex_array::scalar::Scalar;
use vortex_array::session::ArraySession;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_session::VortexSession;
use vortex_sparse::Sparse;

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
        .with::<AggregateFnSession>()
        .with::<MemorySession>();
    vortex_sparse::initialize(&session);
    session
});

/// Average list lengths: short lists weigh per-list overhead, long lists per-element work. The
/// longest is about the length of statpopgen's genotype lists.
const LIST_SIZES: &[usize] = &[8, 256, 4_096];
/// Element chunks in each chunked benchmark.
const NUM_CHUNKS: usize = 4;

/// Elements for lists of `list_size`: at least 4096, and at least 16 lists.
fn num_elements(list_size: usize) -> usize {
    (16 * list_size).max(4_096)
}

/// `len` nullable bools that are `false` except at ~28% patch positions, ~3/4 of patches null.
fn sparse_bool_elements(len: usize, seed: u64) -> ArrayRef {
    let mut rng = StdRng::seed_from_u64(seed);
    let indices: Buffer<u32> = (0..len as u32).filter(|_| rng.random_bool(0.28)).collect();
    let values = BoolArray::from_iter(
        (0..indices.len()).map(|_| rng.random_bool(0.25).then(|| rng.random_bool(0.5))),
    )
    .into_array();
    Sparse::try_new(
        indices.into_array(),
        values,
        len,
        Scalar::bool(false, Nullability::Nullable),
        &mut SESSION.create_execution_ctx(),
    )
    .unwrap()
    .into_array()
}

/// `len` sparse bools in `NUM_CHUNKS` chunks.
fn chunked_sparse_bool_elements(len: usize) -> ArrayRef {
    let chunk_len = len.div_ceil(NUM_CHUNKS);
    let chunks: Vec<ArrayRef> = (0..len)
        .step_by(chunk_len)
        .map(|start| sparse_bool_elements(chunk_len.min(len - start), start as u64))
        .collect();
    let dtype = chunks[0].dtype().clone();
    ChunkedArray::try_new(chunks, dtype).unwrap().into_array()
}

/// A `List` over `elements` with `list_size` elements per list on average, ~1/8 of lists null.
fn list_over(elements: ArrayRef, list_size: usize) -> ArrayRef {
    let len = elements.len();
    let num_lists = len / list_size;
    let mut rng = StdRng::seed_from_u64(num_lists as u64);
    let cut_dist = Uniform::new_inclusive(0i32, len as i32).unwrap();
    let mut cuts: Vec<i32> = (0..num_lists - 1).map(|_| rng.sample(cut_dist)).collect();
    cuts.sort_unstable();
    let offsets: Buffer<i32> = once(0).chain(cuts).chain(once(len as i32)).collect();
    let null_dist = Uniform::new(0u32, 8).unwrap();
    let validity = Validity::from_iter((0..num_lists).map(|_| rng.sample(null_dist) != 0));
    ListArray::try_new(
        elements,
        offsets.into_array(),
        validity,
        &mut SESSION.create_execution_ctx(),
    )
    .unwrap()
    .into_array()
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

#[divan::bench(args = LIST_SIZES)]
fn sparse_bool_sum(bencher: Bencher, list_size: usize) {
    run(
        bencher,
        list_over(sparse_bool_elements(num_elements(list_size), 0), list_size),
    );
}

#[divan::bench(args = LIST_SIZES)]
fn chunked_sparse_bool_sum(bencher: Bencher, list_size: usize) {
    run(
        bencher,
        list_over(
            chunked_sparse_bool_elements(num_elements(list_size)),
            list_size,
        ),
    );
}
