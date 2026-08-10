// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Microbenchmarks for native `ST_Intersection` over polygon pairs.
//!
//! The cases cover simple building-like rectangles, more detailed boundaries, and strict null
//! propagation. Inputs overlap because SpatialBench Q9 prefilters pairs with `ST_Intersects`.
//!
//! Run with `cargo bench -p vortex-spatial --bench intersection`.

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
use vortex_spatial::scalar_fn::intersection::SpatialIntersection;
use vortex_spatial::test_harness::polygon_column;
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

fn regular_polygon(cx: f64, cy: f64, radius: f64, vertices: usize) -> Vec<(f64, f64)> {
    (0..=vertices)
        .map(|vertex| {
            let angle = TAU * (vertex % vertices) as f64 / vertices as f64;
            (cx + radius * angle.cos(), cy + radius * angle.sin())
        })
        .collect()
}

fn polygon_pairs(vertices: usize) -> (ArrayRef, ArrayRef) {
    let left = polygon_column(
        (0..ROWS)
            .map(|row| vec![regular_polygon(row as f64, 0.0, 1.0, vertices)])
            .collect(),
    )
    .unwrap();
    let right = polygon_column(
        (0..ROWS)
            .map(|row| vec![regular_polygon(row as f64 + 0.5, 0.0, 1.0, vertices)])
            .collect(),
    )
    .unwrap();
    (left, right)
}

fn intersections(left: &ArrayRef, right: &ArrayRef, ctx: &mut ExecutionCtx) -> ArrayRef {
    SpatialIntersection::try_new_array(left.clone(), right.clone())
        .unwrap()
        .into_array()
        .execute::<Canonical>(ctx)
        .unwrap()
        .into_array()
}

fn bench_intersections(bencher: Bencher, left: ArrayRef, right: ArrayRef) {
    let mut ctx = SESSION.create_execution_ctx();
    bencher
        .counter(ItemsCount::new(ROWS))
        .bench_local(|| intersections(&left, &right, &mut ctx));
}

#[divan::bench]
fn rectangles(bencher: Bencher) {
    let (left, right) = polygon_pairs(4);
    bench_intersections(bencher, left, right);
}

#[divan::bench]
fn thirty_two_vertex_boundaries(bencher: Bencher) {
    let (left, right) = polygon_pairs(32);
    bench_intersections(bencher, left, right);
}

#[divan::bench]
fn nullable_rectangles(bencher: Bencher) {
    let (left, right) = polygon_pairs(4);
    let left = MaskedArray::try_new(
        left,
        Validity::from_iter((0..ROWS).map(|row| !row.is_multiple_of(8))),
    )
    .unwrap()
    .into_array();
    bench_intersections(bencher, left, right);
}
