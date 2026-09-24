// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
//! Microbenchmarks for the Elias-Fano array's read and write paths.
//!
//! `scalar_at` is the per-element path and reads the low-bits child once per probe, so it is what a
//! change to that path has to be judged against. `encode` and `decode_bulk` cover the two batch
//! paths.
//!
//! Three shapes, because the layout behaves differently in each:
//!
//! * `Sparse` — a wide universe, so `lower_width` is large and nearly every read touches the child.
//! * `Dense` — a universe no wider than the row count, so `lower_width` is zero and the low-bits
//!   child is never read at all. The difference against `Sparse` is the child's whole cost.
//! * `Duplicates` — few distinct values over a wide universe, so each occupied high-part bucket is
//!   deep. Random data almost never produces this.

#![allow(
    clippy::cast_possible_truncation,
    clippy::expect_used,
    clippy::tests_outside_test_module,
    clippy::unwrap_used
)]

use std::sync::LazyLock;

use divan::Bencher;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_elias_fano::EliasFanoArray;
use vortex_elias_fano::elias_fano_encode;
use vortex_session::VortexSession;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    vortex_fastlanes::initialize(&session);
    vortex_elias_fano::initialize(&session);
    session
});

/// Deterministic xorshift, so a run is reproducible without a `rand` dependency.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, bound: u64) -> u64 {
        self.next_u64() % bound
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Shape {
    Sparse,
    Dense,
    Duplicates,
}

/// The universe every shape draws from, wide enough that `Sparse` gets ten low bits at 2^20 rows.
const UNIVERSE: u64 = 1 << 30;

/// Distinct values in the `Duplicates` shape: at 2^20 rows that is ~64 rows per value, so an
/// occupied bucket is deep enough for `search_bucket` to bisect rather than walk.
const LEVELS: u64 = 1 << 14;

fn values(n: usize, shape: Shape) -> Vec<u64> {
    let mut rng = Rng(0x5EED_1234_ABCD_0001);
    let mut out: Vec<u64> = match shape {
        Shape::Sparse => (0..n).map(|_| rng.below(UNIVERSE)).collect(),
        // Universe == row count, which drives `lower_width` to zero.
        Shape::Dense => (0..n).map(|_| rng.below(n as u64)).collect(),
        Shape::Duplicates => {
            let step = UNIVERSE / LEVELS;
            (0..n).map(|_| rng.below(LEVELS) * step).collect()
        }
    };
    out.sort_unstable();
    out
}

fn encoded(n: usize, shape: Shape) -> EliasFanoArray {
    let array = PrimitiveArray::from_iter(values(n, shape));
    let mut ctx = SESSION.create_execution_ctx();
    elias_fano_encode(array.as_ref().as_::<Primitive>(), &mut ctx).expect("encode")
}

/// Probe count held fixed across shapes and row counts, so the reported figure is comparable.
const PROBES: usize = 4096;

const CASES: &[(Shape, usize)] = &[
    (Shape::Sparse, 1 << 16),
    (Shape::Sparse, 1 << 20),
    (Shape::Dense, 1 << 20),
    (Shape::Duplicates, 1 << 20),
];

fn random_indices(n: usize) -> Vec<usize> {
    let mut rng = Rng(0xA11C_E000_0000_0001);
    (0..PROBES).map(|_| rng.below(n as u64) as usize).collect()
}

/// Point lookups through `OperationsVTable::scalar_at`: one sampled `select1` and one low-bits read
/// apiece, in a random order so nothing about locality is being measured by accident.
#[divan::bench(args = CASES)]
fn scalar_at(bencher: Bencher, case: (Shape, usize)) {
    let (shape, n) = case;
    let array: ArrayRef = encoded(n, shape).into_array();
    let indices = random_indices(n);
    bencher
        .with_inputs(|| SESSION.create_execution_ctx())
        .bench_local_values(|mut ctx| {
            for &index in &indices {
                divan::black_box(array.execute_scalar(index, &mut ctx).unwrap());
            }
        });
}

/// Whole-array decode, which walks the upper array once and reads the low bits a FastLanes block at
/// a time rather than one element at a time.
#[divan::bench(args = CASES)]
fn decode_bulk(bencher: Bencher, case: (Shape, usize)) {
    let (shape, n) = case;
    let array = encoded(n, shape);
    bencher
        .with_inputs(|| (array.clone().into_array(), SESSION.create_execution_ctx()))
        .bench_local_values(|(array, mut ctx)| {
            divan::black_box(array.execute::<PrimitiveArray>(&mut ctx).unwrap());
        });
}

#[divan::bench(args = CASES)]
fn encode(bencher: Bencher, case: (Shape, usize)) {
    let (shape, n) = case;
    let array = PrimitiveArray::from_iter(values(n, shape));
    bencher
        .with_inputs(|| SESSION.create_execution_ctx())
        .bench_local_values(|mut ctx| {
            divan::black_box(
                elias_fano_encode(array.as_ref().as_::<Primitive>(), &mut ctx).unwrap(),
            );
        });
}

/// Row counts for the scaling benches below. Each measures an operation whose work should not
/// depend on the row count, so a figure that grows along this axis is the regression being shown.
const SCALING: &[usize] = &[1 << 16, 1 << 20, 1 << 24];

/// Rows in the fixed-size window [`slice_window`] cuts out and the fixed number of indices
/// [`take_few`] gathers.
const WINDOW: usize = 16;

/// Row count per chunk in [`slice_into_chunks`], roughly what a writer cuts a column into.
const CHUNK: usize = 1 << 16;

/// Slice a fixed 16-row window out of the middle of an `n`-row array.
///
/// A slice only records `first_rank`, so this should cost the same at every `n`. It does not:
/// `SliceReduce` rebuilds the array through `try_new`, which re-runs `VTable::validate`, and that
/// walks every entry of both sample tables, about `n / 256 + zeros / 512` of them.
#[divan::bench(args = SCALING)]
fn slice_window(bencher: Bencher, n: usize) {
    let array: ArrayRef = encoded(n, Shape::Sparse).into_array();
    let start = n / 2;
    bencher.bench_local(|| divan::black_box(array.slice(start..start + WINDOW).unwrap()));
}

/// Baseline for [`slice_window`]: the same slice of the same values, uncompressed.
#[divan::bench(args = SCALING)]
fn slice_window_primitive(bencher: Bencher, n: usize) {
    let array = PrimitiveArray::from_iter(values(n, Shape::Sparse)).into_array();
    let start = n / 2;
    bencher.bench_local(|| divan::black_box(array.slice(start..start + WINDOW).unwrap()));
}

/// Cut an `n`-row array into `CHUNK`-row slices, as a writer repartitioning a column does.
///
/// There are `n / CHUNK` slices and each re-validates the whole layout, so the total grows with
/// `n^2` rather than `n`.
#[divan::bench(args = SCALING)]
fn slice_into_chunks(bencher: Bencher, n: usize) {
    let array: ArrayRef = encoded(n, Shape::Sparse).into_array();
    bencher.bench_local(|| {
        for start in (0..n).step_by(CHUNK) {
            divan::black_box(array.slice(start..(start + CHUNK).min(n)).unwrap());
        }
    });
}

/// Baseline for [`slice_into_chunks`]: the same chunking of the same values, uncompressed.
#[divan::bench(args = SCALING)]
fn slice_into_chunks_primitive(bencher: Bencher, n: usize) {
    let array = PrimitiveArray::from_iter(values(n, Shape::Sparse)).into_array();
    bencher.bench_local(|| {
        for start in (0..n).step_by(CHUNK) {
            divan::black_box(array.slice(start..(start + CHUNK).min(n)).unwrap());
        }
    });
}

fn few_indices(n: usize) -> ArrayRef {
    let mut rng = Rng(0x7A4E_0000_0000_0001);
    PrimitiveArray::from_iter((0..WINDOW).map(|_| rng.below(n as u64))).into_array()
}

/// Gather 16 random rows from an `n`-row array and execute the result.
///
/// With no take kernel this decodes all `n` rows to keep 16, so it grows with `n` although the
/// point reads it needs are `O(1)` each; compare [`take_few_scalar_at`].
#[divan::bench(args = SCALING)]
fn take_few(bencher: Bencher, n: usize) {
    let array: ArrayRef = encoded(n, Shape::Sparse).into_array();
    let indices = few_indices(n);
    bencher
        .with_inputs(|| SESSION.create_execution_ctx())
        .bench_local_values(|mut ctx| {
            divan::black_box(
                array
                    .take(indices.clone())
                    .unwrap()
                    .execute::<PrimitiveArray>(&mut ctx)
                    .unwrap(),
            );
        });
}

/// What [`take_few`] could cost: the same 16 rows read one at a time through `scalar_at`.
#[divan::bench(args = SCALING)]
fn take_few_scalar_at(bencher: Bencher, n: usize) {
    let array: ArrayRef = encoded(n, Shape::Sparse).into_array();
    let mut rng = Rng(0x7A4E_0000_0000_0001);
    let indices: Vec<usize> = (0..WINDOW).map(|_| rng.below(n as u64) as usize).collect();
    bencher
        .with_inputs(|| SESSION.create_execution_ctx())
        .bench_local_values(|mut ctx| {
            for &index in &indices {
                divan::black_box(array.execute_scalar(index, &mut ctx).unwrap());
            }
        });
}

fn main() {
    divan::main();
}
