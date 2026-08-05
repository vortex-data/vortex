// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Production baselines for planar `ST_Distance` over native geometry columns.
//!
//! Row counts are set by the 1 ms per-iteration budget from `docs/developer-guide/benchmarking.md`,
//! measured against CodSpeed's CPU simulation rather than local wall clock. Polygon distance
//! rebuilds both geometry R-trees per call, so it costs an order of magnitude more per row than
//! point distance and takes a correspondingly smaller fixture.

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
use vortex_array::arrays::MaskedArray;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
use vortex_geo::scalar_fn::distance::GeoDistance;
use vortex_geo::test_harness::geo_session;
use vortex_geo::test_harness::point_column;
use vortex_geo::test_harness::polygon_column;
use vortex_session::VortexSession;

// Scalar function execution allocates its output inside the timed region, so use the vendored
// allocator instead of measuring glibc differences between CodSpeed runner images.
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

static SESSION: LazyLock<VortexSession> = LazyLock::new(geo_session);

const POINT_ROWS: usize = 512;
const POLYGON_ROWS: usize = 32;

fn main() {
    divan::main();
}

/// Deterministic pseudo-random value in `[0, 1)`.
fn unit(i: usize) -> f64 {
    ((i.wrapping_mul(2654435761) >> 8) % 10_000) as f64 / 10_000.0
}

fn points(rows: usize, offset: usize) -> ArrayRef {
    let xs = (0..rows)
        .map(|i| 300.0 * unit(i + offset) - 150.0)
        .collect();
    let ys = (0..rows)
        .map(|i| 300.0 * unit(i + offset + 1) - 150.0)
        .collect();
    point_column(xs, ys).unwrap()
}

/// A small square centered at `(cx, cy)`.
fn square(cx: f64, cy: f64) -> Vec<Vec<(f64, f64)>> {
    vec![vec![
        (cx - 1.0, cy - 1.0),
        (cx + 1.0, cy - 1.0),
        (cx + 1.0, cy + 1.0),
        (cx - 1.0, cy + 1.0),
        (cx - 1.0, cy - 1.0),
    ]]
}

fn polygons(rows: usize) -> ArrayRef {
    let rows = (0..rows)
        .map(|i| square(150.0 + 700.0 * unit(i), 150.0 + 700.0 * unit(i + 1)))
        .collect();
    polygon_column(rows).unwrap()
}

fn point_constant(x: f64, y: f64, rows: usize, ctx: &mut ExecutionCtx) -> ArrayRef {
    let scalar = point_column(vec![x], vec![y])
        .unwrap()
        .execute_scalar(0, ctx)
        .unwrap();
    ConstantArray::new(scalar, rows).into_array()
}

fn polygon_constant(rows: usize, ctx: &mut ExecutionCtx) -> ArrayRef {
    let scalar = polygon_column(vec![square(0.0, 0.0)])
        .unwrap()
        .execute_scalar(0, ctx)
        .unwrap();
    ConstantArray::new(scalar, rows).into_array()
}

fn execute(distance: VortexResult<impl IntoArray>, ctx: &mut ExecutionCtx) -> ArrayRef {
    distance
        .unwrap()
        .into_array()
        .execute::<Canonical>(ctx)
        .unwrap()
        .into_array()
}

fn bench_distance(bencher: Bencher, lhs: ArrayRef, rhs: ArrayRef) {
    let rows = lhs.len();
    let mut ctx = SESSION.create_execution_ctx();
    bencher.counter(ItemsCount::new(rows)).bench_local(|| {
        execute(
            GeoDistance::try_new_array(lhs.clone(), rhs.clone()),
            &mut ctx,
        )
    });
}

#[divan::bench]
fn point_column_x_point_column(bencher: Bencher) {
    bench_distance(bencher, points(POINT_ROWS, 0), points(POINT_ROWS, 97));
}

#[divan::bench]
fn point_column_x_constant_point(bencher: Bencher) {
    let mut ctx = SESSION.create_execution_ctx();
    let point = point_constant(0.0, 0.0, POINT_ROWS, &mut ctx);
    bench_distance(bencher, points(POINT_ROWS, 0), point);
}

#[divan::bench]
fn polygon_column_x_constant_polygon(bencher: Bencher) {
    let mut ctx = SESSION.create_execution_ctx();
    let polygon = polygon_constant(POLYGON_ROWS, &mut ctx);
    bench_distance(bencher, polygons(POLYGON_ROWS), polygon);
}

#[divan::bench]
fn nullable_point_column_x_constant_point(bencher: Bencher) {
    let validity = Validity::from_iter((0..POINT_ROWS).map(|i| !i.is_multiple_of(8)));
    let points = MaskedArray::try_new(points(POINT_ROWS, 0), validity)
        .unwrap()
        .into_array();
    let mut ctx = SESSION.create_execution_ctx();
    let point = point_constant(0.0, 0.0, POINT_ROWS, &mut ctx);
    bench_distance(bencher, points, point);
}
