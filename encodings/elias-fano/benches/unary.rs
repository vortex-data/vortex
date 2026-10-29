// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Baselines for the naive `Unary` encoding.
//!
//! Later work on `Unary`, such as an index, a slice kernel, or a cursor, will be measured against
//! these numbers. CodSpeed matches results by name, so keep the names, shapes, and sizes stable.
//!
//! Each input is the array that an Elias-Fano upper child holds for `LEN` sorted values.
//! Elias-Fano puts each value in a _bucket_ by its high bits, and the upper child stores one count
//! per bucket. The array under test therefore has one element per bucket, and its elements sum to
//! `LEN`. The fixtures use buckets that are one value wide:
//!
//! - `Dense`: values in `0..LEN`, so `LEN` buckets with about one value each.
//! - `Spread`: values in `0..2 * LEN`, so `2 * LEN` buckets with about half a value each.
//! - `Runs`: every count is 32, which is the duplicate-heavy case with fewer buckets than values.
//! - `Gap`: two dense halves with `LEN / 2` empty buckets between them.
//!
//! Elias-Fano chooses its low bits so that there are between `LEN` and `2 * LEN` buckets, and
//! `Dense` and `Spread` are the two ends of that range. Encoding and decoding cost about one step
//! per bucket, so `Spread` decides how large `LEN` can be within the 1 ms per-iteration budget.

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
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_elias_fano::unary_encode;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty();
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

/// Sorted values behind each shape. Encoding and decoding cost about one step per bucket, and
/// `Spread`, the largest shape, has `2 * LEN` buckets, so this keeps every iteration inside the
/// 1 ms per-iteration budget.
const LEN: usize = 1 << 14;

/// Where the second half of `Gap` starts, in values.
const GAP: usize = LEN;

/// Rows per value in `Runs`. At 64, each count would take 65 bits, past the size rule.
const ROWS_PER_VALUE: usize = 32;

/// Probes per iteration, as in `vortex-array`'s probe benchmark. Each is a linear scan from the
/// first bit, so 256 keep an iteration inside the budget.
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
    unary_encode(
        primitive(shape).as_ref().as_::<Primitive>(),
        &mut SESSION.create_execution_ctx(),
    )
    .unwrap()
    .unwrap()
    .into_array()
}

#[divan::bench(args = SHAPES)]
fn encode(bencher: Bencher, shape: Shape) {
    let values = Buffer::from(counts(shape));
    // A fresh array each time, so `sum` cannot reuse the statistic cached on the last one.
    bencher
        .with_inputs(|| {
            (
                PrimitiveArray::new(values.clone(), Validity::NonNullable),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_local_values(|(array, mut ctx)| {
            divan::black_box(unary_encode(array.as_ref().as_::<Primitive>(), &mut ctx).unwrap());
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

/// Consecutive point reads from the middle through one repeated probe, so a cursor kept in its
/// state could answer each from the last position.
#[divan::bench(args = SHAPES)]
fn scalar_at_sequential(bencher: Bencher, shape: Shape) {
    let array = encoded(shape);
    let start = array.len() / 2;
    bencher
        .with_inputs(|| (SESSION.create_execution_ctx(), array.repeated_probe()))
        .bench_local_refs(|(ctx, probe)| {
            for index in start..start + PROBES {
                divan::black_box(probe.execute_scalar(index, ctx).unwrap());
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
