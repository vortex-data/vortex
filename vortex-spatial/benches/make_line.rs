// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Microbenchmarks for native `ST_MakeLine`.
//!
//! The cases cover the normal paired-column operation, a broadcast point constant, and strict
//! null propagation. They execute the result to its canonical representation so the benchmark
//! includes construction of the two-vertex line storage.
//!
//! `ROWS` keeps each case near the roughly 1 ms iteration budget recommended for CodSpeed.
//!
//! Run with `cargo bench -p vortex-spatial --bench make_line`.

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use divan::counter::ItemsCount;
use mimalloc::MiMalloc;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_session::VortexSession;
use vortex_spatial::scalar_fn::make_line::SpatialMakeLine;
use vortex_spatial::test_harness::nullable_point_column;
use vortex_spatial::test_harness::point_column;
use vortex_spatial::test_harness::spatial_session;

// Scalar function execution allocates its output inside the timed region, so use the vendored
// allocator instead of measuring glibc differences between CodSpeed runner images.
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

static SESSION: LazyLock<VortexSession> = LazyLock::new(spatial_session);

const ROWS: usize = 512;

fn main() {
    divan::main();
}

/// Deterministic pseudo-random value in `[0, 1)`.
fn unit(i: usize) -> f64 {
    ((i.wrapping_mul(2_654_435_761) >> 8) % 10_000) as f64 / 10_000.0
}

fn points(offset: usize) -> ArrayRef {
    let xs = (0..ROWS)
        .map(|i| 300.0 * unit(i + offset) - 150.0)
        .collect();
    let ys = (0..ROWS)
        .map(|i| 300.0 * unit(i + offset + 1) - 150.0)
        .collect();
    point_column(xs, ys).unwrap()
}

fn nullable_points(offset: usize, null_every: usize) -> ArrayRef {
    nullable_point_column(
        (0..ROWS)
            .map(|i| {
                (!i.is_multiple_of(null_every)).then(|| {
                    (
                        300.0 * unit(i + offset) - 150.0,
                        300.0 * unit(i + offset + 1) - 150.0,
                    )
                })
            })
            .collect(),
    )
    .unwrap()
}

fn point_constant(ctx: &mut ExecutionCtx) -> ArrayRef {
    let scalar = point_column(vec![0.0], vec![0.0])
        .unwrap()
        .execute_scalar(0, ctx)
        .unwrap();
    ConstantArray::new(scalar, ROWS).into_array()
}

fn make_lines(starts: &ArrayRef, ends: &ArrayRef, ctx: &mut ExecutionCtx) -> ArrayRef {
    SpatialMakeLine::try_new_array(starts.clone(), ends.clone())
        .unwrap()
        .into_array()
        .execute::<Canonical>(ctx)
        .unwrap()
        .into_array()
}

#[divan::bench]
fn column_x_column(bencher: Bencher) {
    let starts = points(0);
    let ends = points(97);
    let mut ctx = SESSION.create_execution_ctx();
    bencher
        .counter(ItemsCount::new(ROWS))
        .bench_local(|| make_lines(&starts, &ends, &mut ctx));
}

#[divan::bench]
fn column_x_constant(bencher: Bencher) {
    let starts = points(0);
    let mut ctx = SESSION.create_execution_ctx();
    let end = point_constant(&mut ctx);
    bencher
        .counter(ItemsCount::new(ROWS))
        .bench_local(|| make_lines(&starts, &end, &mut ctx));
}

#[divan::bench]
fn nullable_columns(bencher: Bencher) {
    let starts = nullable_points(0, 8);
    let ends = nullable_points(97, 11);
    let mut ctx = SESSION.create_execution_ctx();
    bencher
        .counter(ItemsCount::new(ROWS))
        .bench_local(|| make_lines(&starts, &ends, &mut ctx));
}
