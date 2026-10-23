// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Baselines for the naive `Unary` encoding, so each later optimisation can be measured
//! against them. Keep the names and inputs stable: CodSpeed compares benchmarks by name.
//!
//! Each shape is what an Elias-Fano high part will hold for `LEN` sorted values: one count per
//! bucket of how many values fall in it. All four are small enough for the encoder's size rule:
//!
//! * `Dense`: values drawn from `[0, LEN)`, so about one value per bucket.
//! * `Spread`: values drawn from `[0, 2 * LEN)`, about half a value per bucket, the sparsest an
//!   Elias-Fano high part gets.
//! * `Runs`: `LEN / 32` values, 32 rows each, so every count is 32 and set bits come in long runs.
//! * `Gap`: dense halves either side of one jump of `GAP` values, a long run of empty buckets that
//!   every scan crossing it has to walk.

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_elias_fano::unary_encode;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    vortex_elias_fano::initialize(&session);
    session
});

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Shape {
    Dense,
    Spread,
    Runs,
    Gap,
}

const SHAPES: &[Shape] = &[Shape::Dense, Shape::Spread, Shape::Runs, Shape::Gap];

/// Sorted values behind each shape, few enough that a whole decode stays inside the 1 ms
/// per-iteration budget.
const LEN: usize = 1 << 16;

/// The jump in `Gap`, in values.
const GAP: usize = LEN;

/// Rows per value in `Runs`. At 64, each count would take 65 bits, past the size rule.
const ROWS_PER_VALUE: usize = 32;

/// Probes per iteration. Each is a linear scan from the first bit, so 256 keep an iteration near
/// the budget, as in `vortex-array`'s probe benchmark.
const PROBES: usize = 256;

/// How many of `values` fall in each of `buckets` buckets, one value wide.
fn bucket_counts(values: impl Iterator<Item = usize>, buckets: usize) -> Vec<u64> {
    let mut counts = vec![0; buckets];
    for value in values {
        counts[value] += 1;
    }
    counts
}

fn counts(shape: Shape) -> Vec<u64> {
    let mut rng = StdRng::seed_from_u64(0);
    match shape {
        Shape::Dense => bucket_counts((0..LEN).map(|_| rng.random_range(0..LEN)), LEN),
        Shape::Spread => bucket_counts((0..LEN).map(|_| rng.random_range(0..2 * LEN)), 2 * LEN),
        Shape::Runs => bucket_counts(
            (0..LEN).map(|row| row / ROWS_PER_VALUE),
            LEN / ROWS_PER_VALUE,
        ),
        Shape::Gap => bucket_counts(
            (0..LEN).map(|row| {
                let offset = if row < LEN / 2 { 0 } else { GAP };
                offset + rng.random_range(0..LEN / 2)
            }),
            GAP + LEN / 2,
        ),
    }
}

fn primitive(shape: Shape) -> PrimitiveArray {
    PrimitiveArray::from_iter(counts(shape))
}

fn encoded(shape: Shape) -> ArrayRef {
    unary_encode(primitive(shape).as_ref().as_::<Primitive>())
        .unwrap()
        .unwrap()
        .into_array()
}

#[divan::bench(args = SHAPES)]
fn encode(bencher: Bencher, shape: Shape) {
    let array = primitive(shape);
    bencher.bench_local(|| {
        divan::black_box(unary_encode(array.as_ref().as_::<Primitive>()).unwrap());
    });
}

/// The whole array, in one walk over the unset bits.
#[divan::bench(args = SHAPES)]
fn decode(bencher: Bencher, shape: Shape) {
    let array = encoded(shape);
    bencher
        .with_inputs(|| (array.clone(), SESSION.create_execution_ctx()))
        .bench_local_values(|(array, mut ctx)| {
            divan::black_box(array.execute::<PrimitiveArray>(&mut ctx).unwrap());
        });
}

/// Point reads in a random order, each a scan from the first bit.
#[divan::bench(args = SHAPES)]
fn scalar_at_random(bencher: Bencher, shape: Shape) {
    let array = encoded(shape);
    let mut rng = StdRng::seed_from_u64(1);
    let indices: Vec<usize> = (0..PROBES)
        .map(|_| rng.random_range(0..array.len()))
        .collect();
    bencher
        .with_inputs(|| SESSION.create_execution_ctx())
        .bench_local_values(|mut ctx| {
            for &index in &indices {
                divan::black_box(array.execute_scalar(index, &mut ctx).unwrap());
            }
        });
}

/// Consecutive point reads from the middle, which a cursor could answer from the last position.
#[divan::bench(args = SHAPES)]
fn scalar_at_sequential(bencher: Bencher, shape: Shape) {
    let array = encoded(shape);
    let start = array.len() / 2;
    bencher
        .with_inputs(|| SESSION.create_execution_ctx())
        .bench_local_values(|mut ctx| {
            for index in start..start + PROBES {
                divan::black_box(array.execute_scalar(index, &mut ctx).unwrap());
            }
        });
}

/// The middle half, through the default slice path, which decodes the whole array first.
#[divan::bench(args = SHAPES)]
fn slice_decode(bencher: Bencher, shape: Shape) {
    let array = encoded(shape);
    let len = array.len();
    bencher
        .with_inputs(|| (array.clone(), SESSION.create_execution_ctx()))
        .bench_local_values(|(array, mut ctx)| {
            let sliced = array.slice(len / 4..3 * len / 4).unwrap();
            divan::black_box(sliced.execute::<PrimitiveArray>(&mut ctx).unwrap());
        });
}

fn main() {
    divan::main();
}
