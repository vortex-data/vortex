// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Microbenchmarks for native `ST_Area` over polygons and multipolygons.
//!
//! The cases separate the costs of vertex traversal, interior rings, nested polygons, and strict
//! null propagation. They execute through the scalar function and materialize the `f64` result.
//!
//! Run with `cargo bench -p vortex-spatial --bench area`.

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
use vortex_array::arrays::MaskedArray;
use vortex_array::validity::Validity;
use vortex_session::VortexSession;
use vortex_spatial::scalar_fn::area::SpatialArea;
use vortex_spatial::test_harness::multipolygon_column;
use vortex_spatial::test_harness::polygon_column;
use vortex_spatial::test_harness::spatial_session;

// Scalar function execution allocates its output inside the timed region, so use the vendored
// allocator instead of measuring glibc differences between CodSpeed runner images.
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

static SESSION: LazyLock<VortexSession> = LazyLock::new(spatial_session);

/// Sized so the multipolygon case, the slowest arm, stays inside the 1 ms per-iteration budget
/// from `docs/developer-guide/benchmarking.md` under CodSpeed's CPU simulation.
const ROWS: usize = 128;

fn main() {
    divan::main();
}

/// A closed square ring centered at `(cx, cy)`.
fn square(cx: f64, cy: f64, radius: f64) -> Vec<(f64, f64)> {
    vec![
        (cx - radius, cy - radius),
        (cx + radius, cy - radius),
        (cx + radius, cy + radius),
        (cx - radius, cy + radius),
        (cx - radius, cy - radius),
    ]
}

fn simple_polygons() -> ArrayRef {
    polygon_column(
        (0..ROWS)
            .map(|row| vec![square(row as f64, row as f64, 10.0)])
            .collect(),
    )
    .unwrap()
}

fn polygons_with_holes() -> ArrayRef {
    polygon_column(
        (0..ROWS)
            .map(|row| {
                let center = row as f64;
                vec![
                    square(center, center, 10.0),
                    square(center - 4.0, center, 1.0),
                    square(center + 4.0, center, 1.0),
                ]
            })
            .collect(),
    )
    .unwrap()
}

fn multipolygons() -> ArrayRef {
    multipolygon_column(
        (0..ROWS)
            .map(|row| {
                let center = row as f64;
                vec![
                    vec![square(center - 12.0, center, 5.0)],
                    vec![square(center, center, 5.0)],
                    vec![square(center + 12.0, center, 5.0)],
                ]
            })
            .collect(),
    )
    .unwrap()
}

fn areas(geometry: &ArrayRef, ctx: &mut ExecutionCtx) -> ArrayRef {
    SpatialArea::try_new(geometry.clone())
        .unwrap()
        .into_array()
        .execute::<Canonical>(ctx)
        .unwrap()
        .into_array()
}

fn bench_area(bencher: Bencher, geometry: ArrayRef) {
    let mut ctx = SESSION.create_execution_ctx();
    bencher
        .counter(ItemsCount::new(ROWS))
        .bench_local(|| areas(&geometry, &mut ctx));
}

#[divan::bench]
fn simple_polygon(bencher: Bencher) {
    bench_area(bencher, simple_polygons());
}

#[divan::bench]
fn polygon_with_holes(bencher: Bencher) {
    bench_area(bencher, polygons_with_holes());
}

#[divan::bench]
fn multipolygon(bencher: Bencher) {
    bench_area(bencher, multipolygons());
}

#[divan::bench]
fn nullable_polygon(bencher: Bencher) {
    let geometry = MaskedArray::try_new(
        simple_polygons(),
        Validity::from_iter((0..ROWS).map(|row| !row.is_multiple_of(8))),
    )
    .unwrap()
    .into_array();
    bench_area(bencher, geometry);
}
