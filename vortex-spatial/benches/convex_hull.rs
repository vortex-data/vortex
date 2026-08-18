// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Microbenchmarks for native `ST_ConvexHull` over `MultiPoint` rows.
//!
//! The cases separate ordinary small hulls, larger point sets, and strict null propagation. They
//! execute the result to its canonical polygon representation.
//!
//! Run with `cargo bench -p vortex-spatial --bench convex_hull`.

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
use vortex_array::arrays::MaskedArray;
use vortex_array::validity::Validity;
use vortex_session::VortexSession;
use vortex_spatial::scalar_fn::convex_hull::SpatialConvexHull;
use vortex_spatial::test_harness::multipoint_column;
use vortex_spatial::test_harness::spatial_session;

// Scalar function execution allocates its output inside the timed region, so use the vendored
// allocator instead of measuring glibc differences between CodSpeed runner images.
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

static SESSION: LazyLock<VortexSession> = LazyLock::new(spatial_session);

/// Hull cost under CodSpeed's CPU simulation is a few microseconds per row and grows with points
/// per row, so each case sizes its row count to stay inside the 1 ms per-iteration budget from
/// `docs/developer-guide/benchmarking.md`.
const ROWS: usize = 64;

/// Row count for the 64-point case, whose per-row hull costs roughly four times the 8-point one.
const LARGE_HULL_ROWS: usize = 16;

fn main() {
    divan::main();
}

fn multipoints(rows: usize, points_per_row: usize) -> ArrayRef {
    multipoint_column(
        (0..rows)
            .map(|row| {
                (0..points_per_row)
                    .map(|point| {
                        let angle = TAU * point as f64 / points_per_row as f64;
                        let radius = 10.0 + ((row + point) % 7) as f64;
                        (radius * angle.cos(), radius * angle.sin())
                    })
                    .collect()
            })
            .collect(),
    )
    .unwrap()
}

fn hulls(input: &ArrayRef, ctx: &mut ExecutionCtx) -> ArrayRef {
    SpatialConvexHull::try_new_array(input.clone())
        .unwrap()
        .into_array()
        .execute::<Canonical>(ctx)
        .unwrap()
        .into_array()
}

fn bench_hulls(bencher: Bencher, input: ArrayRef) {
    let mut ctx = SESSION.create_execution_ctx();
    bencher
        .counter(ItemsCount::new(input.len()))
        .bench_local(|| hulls(&input, &mut ctx));
}

#[divan::bench]
fn eight_points(bencher: Bencher) {
    bench_hulls(bencher, multipoints(ROWS, 8));
}

#[divan::bench]
fn sixty_four_points(bencher: Bencher) {
    bench_hulls(bencher, multipoints(LARGE_HULL_ROWS, 64));
}

#[divan::bench]
fn nullable_eight_points(bencher: Bencher) {
    let input = MaskedArray::try_new(
        multipoints(ROWS, 8),
        Validity::from_iter((0..ROWS).map(|row| !row.is_multiple_of(8))),
    )
    .unwrap()
    .into_array();
    bench_hulls(bencher, input);
}
