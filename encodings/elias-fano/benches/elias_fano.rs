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
use vortex_elias_fano::EliasFanoArray;
use vortex_elias_fano::elias_fano_encode;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    vortex_fastlanes::initialize(&session);
    vortex_elias_fano::initialize(&session);
    session
});

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Shape {
    Sparse,
    Dense,
    Duplicates,
}

const SHAPES: &[Shape] = &[Shape::Sparse, Shape::Dense, Shape::Duplicates];

/// Rows per array: small enough that encoding or decoding all of them stays inside the 1 ms
/// per-iteration budget.
const LEN: usize = 1 << 16;

/// The universe `Sparse` and `Duplicates` draw from.
const UNIVERSE: u64 = 1 << 30;

/// Rows per distinct value in `Duplicates`, so an occupied bucket fills a whole word of the upper
/// array.
const ROWS_PER_VALUE: usize = 64;

fn values(shape: Shape) -> Vec<u64> {
    let mut rng = StdRng::seed_from_u64(0);
    let mut out: Vec<u64> = match shape {
        Shape::Sparse => (0..LEN).map(|_| rng.random_range(0..UNIVERSE)).collect(),
        // Universe == row count, which drives `lower_width` to zero.
        Shape::Dense => (0..LEN).map(|_| rng.random_range(0..LEN as u64)).collect(),
        Shape::Duplicates => {
            let levels = (LEN / ROWS_PER_VALUE) as u64;
            let step = UNIVERSE / levels;
            (0..LEN)
                .map(|_| rng.random_range(0..levels) * step)
                .collect()
        }
    };
    out.sort_unstable();
    out
}

fn encoded(shape: Shape) -> EliasFanoArray {
    let array = PrimitiveArray::from_iter(values(shape));
    let mut ctx = SESSION.create_execution_ctx();
    elias_fano_encode(array.as_ref().as_::<Primitive>(), &mut ctx).unwrap()
}

/// Probes per iteration, fixed across shapes so the figures compare. Each runs a whole
/// `execute_scalar`, hence a quarter of the codec benchmark's count.
const PROBES: usize = 1024;

fn random_indices() -> Vec<usize> {
    let mut rng = StdRng::seed_from_u64(1);
    (0..PROBES).map(|_| rng.random_range(0..LEN)).collect()
}

/// Point lookups through `OperationsVTable::scalar_at`: one sampled `select1` and one low-bits read
/// apiece, in a random order so nothing about locality is being measured by accident.
#[divan::bench(args = SHAPES)]
fn scalar_at(bencher: Bencher, shape: Shape) {
    let array: ArrayRef = encoded(shape).into_array();
    let indices = random_indices();
    bencher
        .with_inputs(|| SESSION.create_execution_ctx())
        .bench_local_values(|mut ctx| {
            for &index in &indices {
                divan::black_box(array.execute_scalar(index, &mut ctx).unwrap());
            }
        });
}

/// Whole-array decode, which walks the upper array once and reads the low bits in pieces rather
/// than one element at a time.
#[divan::bench(args = SHAPES)]
fn decode_bulk(bencher: Bencher, shape: Shape) {
    let array = encoded(shape);
    bencher
        .with_inputs(|| (array.clone().into_array(), SESSION.create_execution_ctx()))
        .bench_local_values(|(array, mut ctx)| {
            divan::black_box(array.execute::<PrimitiveArray>(&mut ctx).unwrap());
        });
}

#[divan::bench(args = SHAPES)]
fn encode(bencher: Bencher, shape: Shape) {
    let array = PrimitiveArray::from_iter(values(shape));
    bencher
        .with_inputs(|| SESSION.create_execution_ctx())
        .bench_local_values(|mut ctx| {
            divan::black_box(
                elias_fano_encode(array.as_ref().as_::<Primitive>(), &mut ctx).unwrap(),
            );
        });
}

fn main() {
    divan::main();
}
