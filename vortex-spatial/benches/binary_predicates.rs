// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Microbenchmarks for the binary geometry predicates `ST_Contains` and `ST_Intersects`, focused
//! on the cost of a batch-constant operand.
//!
//! The constant is a 128-vertex query polygon, the shape a spatial filter broadcasts against a
//! column. Arms pair it with a point column (the external `geo` crate answers those pairings with
//! direct point-in-polygon algorithms) and with a small-polygon column (`geo` routes those through
//! bounding-box prechecks and `relate`), covering both a mostly-disjoint dataset where the bbox
//! early-out rejects nearly every row and an all-overlapping one where it never does. The
//! column-x-column arms are the control: no operand is constant, so a prepared path has nothing to
//! hoist and must not regress them.
//!
//! `contains` has no all-overlapping arm. One `contains(query polygon, contained square)` row
//! builds a topology graph over the constant's 128 edges, which CodSpeed's CPU simulation charges
//! around 120 µs, so no row count both fits the per-iteration budget and exercises the row loop.
//! [`intersects::polygons_overlapping_x_constant`] covers the never-rejects case instead.
//!
//! Run with `cargo bench -p vortex-spatial --bench binary_predicates`.

#![expect(clippy::unwrap_used)]

use std::f64::consts::TAU;
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
use vortex_session::VortexSession;
use vortex_spatial::scalar_fn::contains::SpatialContains;
use vortex_spatial::scalar_fn::intersects::SpatialIntersects;
use vortex_spatial::test_harness::point_column;
use vortex_spatial::test_harness::polygon_column;
use vortex_spatial::test_harness::spatial_session;

// Scalar function execution allocates its output inside the timed region, so use the vendored
// allocator instead of measuring glibc differences between CodSpeed runner images.
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(spatial_session);

/// The ordinary arms use the same row count so results are comparable across shapes. A geometry
/// predicate costs roughly a microsecond per row under CodSpeed's CPU simulation, which caps the
/// row count well below a typical batch to stay inside the 1 ms per-iteration budget from
/// `docs/developer-guide/benchmarking.md`.
const ROWS: usize = 1 << 5;

/// The all-overlapping polygon arm never rejects on bounding boxes, so every row pays for the full
/// pairwise predicate. It needs a smaller fixture than [`ROWS`] to stay inside the same budget.
const OVERLAPPING_POLYGON_ROWS: usize = 1 << 4;

/// Deterministic pseudo-random value in `[0, 1)`.
fn unit(i: usize) -> f64 {
    ((i.wrapping_mul(2654435761) >> 8) % 10_000) as f64 / 10_000.0
}

/// The exterior ring of a convex 128-gon of radius 100 centered at `(cx, cy)`: enough vertices
/// that per-row work proportional to the constant's size shows up clearly.
fn query_ring(cx: f64, cy: f64) -> Vec<(f64, f64)> {
    let n = 128;
    (0..=n)
        .map(|i| {
            let theta = (i % n) as f64 / n as f64 * TAU;
            (cx + 100.0 * theta.cos(), cy + 100.0 * theta.sin())
        })
        .collect()
}

/// The query polygon as a batch-constant operand: a top-level `ConstantArray` over the geometry
/// extension scalar, the shape that reaches the row loop's stride-0 path.
fn query_constant(ctx: &mut ExecutionCtx, rows: usize) -> ArrayRef {
    let scalar = polygon_column(vec![vec![query_ring(0.0, 0.0)]])
        .unwrap()
        .execute_scalar(0, ctx)
        .unwrap();
    ConstantArray::new(scalar, rows).into_array()
}

/// A batch-constant point operand with the requested row count.
fn point_constant(ctx: &mut ExecutionCtx, rows: usize) -> ArrayRef {
    let scalar = point_column(vec![0.0], vec![0.0])
        .unwrap()
        .execute_scalar(0, ctx)
        .unwrap();
    ConstantArray::new(scalar, rows).into_array()
}

/// A small square (side 2) centered at `(cx, cy)`.
fn square(cx: f64, cy: f64) -> Vec<Vec<(f64, f64)>> {
    vec![vec![
        (cx - 1.0, cy - 1.0),
        (cx + 1.0, cy - 1.0),
        (cx + 1.0, cy + 1.0),
        (cx - 1.0, cy + 1.0),
        (cx - 1.0, cy - 1.0),
    ]]
}

/// `rows` small squares whose centers avoid the query polygon almost always: the shape of a
/// selective spatial filter, where a bbox check rejects nearly every row.
fn squares_mostly_disjoint(rows: usize) -> ArrayRef {
    let rows = (0..rows)
        .map(|i| square(150.0 + 700.0 * unit(i), 150.0 + 700.0 * unit(i + 1)))
        .collect();
    polygon_column(rows).unwrap()
}

/// `rows` small squares whose centers all fall well inside the query polygon, so a bbox check
/// never rejects and the full pairwise predicate always runs.
fn squares_mostly_overlapping(rows: usize) -> ArrayRef {
    let rows = (0..rows)
        .map(|i| square(120.0 * unit(i) - 60.0, 120.0 * unit(i + 1) - 60.0))
        .collect();
    polygon_column(rows).unwrap()
}

/// `rows` points spread over `[-150, 150)^2`, mixing rows inside and outside the query polygon.
fn points(rows: usize) -> ArrayRef {
    let xs = (0..rows).map(|i| 300.0 * unit(i) - 150.0).collect();
    let ys = (0..rows).map(|i| 300.0 * unit(i + 1) - 150.0).collect();
    point_column(xs, ys).unwrap()
}

/// Execute `array` to completion.
fn execute(array: VortexResult<impl IntoArray>, ctx: &mut ExecutionCtx) -> ArrayRef {
    array
        .unwrap()
        .into_array()
        .execute::<Canonical>(ctx)
        .unwrap()
        .into_array()
}

/// Marks one row in `null_every` null without changing the geometry storage.
fn nullable_every(array: ArrayRef, null_every: usize) -> ArrayRef {
    let validity = Validity::from_iter((0..array.len()).map(|i| !i.is_multiple_of(null_every)));
    MaskedArray::try_new(array, validity).unwrap().into_array()
}

/// A deterministic 90%-null validity pattern.
fn ninety_percent_null(array: ArrayRef) -> ArrayRef {
    let validity = Validity::from_iter((0..array.len()).map(|i| i.is_multiple_of(10)));
    MaskedArray::try_new(array, validity).unwrap().into_array()
}

/// A deterministic 50% validity pattern with a distinct phase per operand. Adjacent phases make
/// the two columns jointly cover every valid/null combination once per four rows.
fn half_valid(array: ArrayRef, phase: usize) -> ArrayRef {
    let validity = Validity::from_iter((0..array.len()).map(|i| (i + phase) % 4 < 2));
    MaskedArray::try_new(array, validity).unwrap().into_array()
}

mod contains {
    use super::*;

    /// Control: no constant operand, direct point-in-polygon per row.
    #[divan::bench]
    fn column_x_column_points(bencher: Bencher) {
        let mut ctx = SESSION.create_execution_ctx();
        let polygons = squares_mostly_overlapping(ROWS);
        let points = points(ROWS);
        bencher.counter(ItemsCount::new(ROWS)).bench_local(|| {
            execute(
                SpatialContains::try_new_array(polygons.clone(), points.clone()),
                &mut ctx,
            )
        });
    }

    /// Control: no constant operand, relate-routed polygon pairs per row.
    #[divan::bench]
    fn column_x_column_polygons(bencher: Bencher) {
        let mut ctx = SESSION.create_execution_ctx();
        let a = squares_mostly_overlapping(ROWS);
        let b = squares_mostly_disjoint(ROWS);
        bencher.counter(ItemsCount::new(ROWS)).bench_local(|| {
            execute(
                SpatialContains::try_new_array(a.clone(), b.clone()),
                &mut ctx,
            )
        });
    }

    /// Constant container against a point column: the `geo` crate's direct point-in-polygon pairing.
    #[divan::bench]
    fn constant_x_points(bencher: Bencher) {
        let mut ctx = SESSION.create_execution_ctx();
        let query = query_constant(&mut ctx, ROWS);
        let points = points(ROWS);
        bencher.counter(ItemsCount::new(ROWS)).bench_local(|| {
            execute(
                SpatialContains::try_new_array(query.clone(), points.clone()),
                &mut ctx,
            )
        });
    }

    /// Constant container against mostly-disjoint polygons: relate-routed, and almost every row
    /// short-circuits on bounding boxes inside relate.
    #[divan::bench]
    fn constant_x_polygons_disjoint(bencher: Bencher) {
        let mut ctx = SESSION.create_execution_ctx();
        let query = query_constant(&mut ctx, ROWS);
        let polygons = squares_mostly_disjoint(ROWS);
        bencher.counter(ItemsCount::new(ROWS)).bench_local(|| {
            execute(
                SpatialContains::try_new_array(query.clone(), polygons.clone()),
                &mut ctx,
            )
        });
    }

    /// Constant container against a point column with one null row in eight.
    #[divan::bench]
    fn constant_x_nullable_points(bencher: Bencher) {
        let mut ctx = SESSION.create_execution_ctx();
        let query = query_constant(&mut ctx, ROWS);
        let points = nullable_every(points(ROWS), 8);
        bencher.counter(ItemsCount::new(ROWS)).bench_local(|| {
            execute(
                SpatialContains::try_new_array(query.clone(), points.clone()),
                &mut ctx,
            )
        });
    }

    /// Constant container against mostly-disjoint polygons with one null row in eight.
    #[divan::bench]
    fn constant_x_nullable_polygons_disjoint(bencher: Bencher) {
        let mut ctx = SESSION.create_execution_ctx();
        let query = query_constant(&mut ctx, ROWS);
        let polygons = nullable_every(squares_mostly_disjoint(ROWS), 8);
        bencher.counter(ItemsCount::new(ROWS)).bench_local(|| {
            execute(
                SpatialContains::try_new_array(query.clone(), polygons.clone()),
                &mut ctx,
            )
        });
    }

    /// Polygon columns against a constant point exercise the direct geometry pairing without a
    /// constant container.
    #[divan::bench]
    fn polygons_x_constant_point(bencher: Bencher) {
        let mut ctx = SESSION.create_execution_ctx();
        let polygons = squares_mostly_overlapping(ROWS);
        let point = point_constant(&mut ctx, ROWS);
        bencher.counter(ItemsCount::new(ROWS)).bench_local(|| {
            execute(
                SpatialContains::try_new_array(polygons.clone(), point.clone()),
                &mut ctx,
            )
        });
    }

    /// The same polygon-x-constant-point pairing with a deterministic 90% null polygon column.
    #[divan::bench]
    fn nullable_polygons_90pct_x_constant_point(bencher: Bencher) {
        let mut ctx = SESSION.create_execution_ctx();
        let polygons = ninety_percent_null(squares_mostly_overlapping(ROWS));
        let point = point_constant(&mut ctx, ROWS);
        bencher.counter(ItemsCount::new(ROWS)).bench_local(|| {
            execute(
                SpatialContains::try_new_array(polygons.clone(), point.clone()),
                &mut ctx,
            )
        });
    }

    /// Independently 50%-valid polygon and point columns cover mixed validity without test-only
    /// execution controls.
    #[divan::bench]
    fn nullable_polygons_x_nullable_points(bencher: Bencher) {
        let mut ctx = SESSION.create_execution_ctx();
        let polygons = half_valid(squares_mostly_overlapping(ROWS), 0);
        let points = half_valid(points(ROWS), 1);
        bencher.counter(ItemsCount::new(ROWS)).bench_local(|| {
            execute(
                SpatialContains::try_new_array(polygons.clone(), points.clone()),
                &mut ctx,
            )
        });
    }
}

mod intersects {
    use super::*;

    /// Control: no constant operand, polygon pairs per row.
    #[divan::bench]
    fn column_x_column_polygons(bencher: Bencher) {
        let mut ctx = SESSION.create_execution_ctx();
        let a = squares_mostly_overlapping(ROWS);
        let b = squares_mostly_disjoint(ROWS);
        bencher.counter(ItemsCount::new(ROWS)).bench_local(|| {
            execute(
                SpatialIntersects::try_new_array(a.clone(), b.clone()),
                &mut ctx,
            )
        });
    }

    /// Point column against the constant query: the `geo` crate answers point-x-polygon directly, with no
    /// bbox precheck to hoist.
    #[divan::bench]
    fn points_x_constant(bencher: Bencher) {
        let mut ctx = SESSION.create_execution_ctx();
        let points = points(ROWS);
        let query = query_constant(&mut ctx, ROWS);
        bencher.counter(ItemsCount::new(ROWS)).bench_local(|| {
            execute(
                SpatialIntersects::try_new_array(points.clone(), query.clone()),
                &mut ctx,
            )
        });
    }

    /// Mostly-disjoint polygons against the constant query: the bbox precheck rejects nearly
    /// every row, so the constant's per-row bounding-box fold dominates the baseline.
    #[divan::bench]
    fn polygons_disjoint_x_constant(bencher: Bencher) {
        let mut ctx = SESSION.create_execution_ctx();
        let polygons = squares_mostly_disjoint(ROWS);
        let query = query_constant(&mut ctx, ROWS);
        bencher.counter(ItemsCount::new(ROWS)).bench_local(|| {
            execute(
                SpatialIntersects::try_new_array(polygons.clone(), query.clone()),
                &mut ctx,
            )
        });
    }

    /// Mostly-overlapping polygons against the constant query: the bbox precheck never rejects, so
    /// every row still pays for the full pairwise predicate. That is several times the per-row cost
    /// of the other arms, so this one uses [`OVERLAPPING_POLYGON_ROWS`].
    #[divan::bench]
    fn polygons_overlapping_x_constant(bencher: Bencher) {
        let mut ctx = SESSION.create_execution_ctx();
        let polygons = squares_mostly_overlapping(OVERLAPPING_POLYGON_ROWS);
        let query = query_constant(&mut ctx, OVERLAPPING_POLYGON_ROWS);
        bencher
            .counter(ItemsCount::new(OVERLAPPING_POLYGON_ROWS))
            .bench_local(|| {
                execute(
                    SpatialIntersects::try_new_array(polygons.clone(), query.clone()),
                    &mut ctx,
                )
            });
    }

    /// Nullable point column against the constant query, with one null row in eight.
    #[divan::bench]
    fn nullable_points_x_constant(bencher: Bencher) {
        let mut ctx = SESSION.create_execution_ctx();
        let points = nullable_every(points(ROWS), 8);
        let query = query_constant(&mut ctx, ROWS);
        bencher.counter(ItemsCount::new(ROWS)).bench_local(|| {
            execute(
                SpatialIntersects::try_new_array(points.clone(), query.clone()),
                &mut ctx,
            )
        });
    }

    /// Mostly-disjoint nullable polygons against the constant query.
    #[divan::bench]
    fn nullable_polygons_disjoint_x_constant(bencher: Bencher) {
        let mut ctx = SESSION.create_execution_ctx();
        let polygons = nullable_every(squares_mostly_disjoint(ROWS), 8);
        let query = query_constant(&mut ctx, ROWS);
        bencher.counter(ItemsCount::new(ROWS)).bench_local(|| {
            execute(
                SpatialIntersects::try_new_array(polygons.clone(), query.clone()),
                &mut ctx,
            )
        });
    }
}
