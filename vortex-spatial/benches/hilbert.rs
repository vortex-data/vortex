// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Microbenchmarks for `vortex.st.hilbert` with constant `Rect` bounds.
//!
//! Run with `cargo bench -p vortex-spatial --bench hilbert`.

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
use vortex_array::scalar::Scalar;
use vortex_session::VortexSession;
use vortex_spatial::scalar_fn::hilbert::SpatialHilbert;
use vortex_spatial::test_harness::MultiPolygonRings;
use vortex_spatial::test_harness::multipolygon_column;
use vortex_spatial::test_harness::point_column;
use vortex_spatial::test_harness::rect_column;
use vortex_spatial::test_harness::spatial_session;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

static SESSION: LazyLock<VortexSession> = LazyLock::new(spatial_session);

const ROWS: usize = 1 << 9;

fn main() {
    divan::main();
}

fn ordinate(i: usize) -> f64 {
    (i.wrapping_mul(2_654_435_761) % 1_000) as f64
}

fn bounds(ctx: &mut ExecutionCtx) -> Scalar {
    rect_column(vec![(0.0, 0.0, 1_000.0, 1_000.0)])
        .unwrap()
        .execute_scalar(0, ctx)
        .unwrap()
}

fn hilbert(column: &ArrayRef, bounds: &Scalar, ctx: &mut ExecutionCtx) -> ArrayRef {
    SpatialHilbert::try_new_array(column.clone(), bounds.clone())
        .unwrap()
        .into_array()
        .execute::<Canonical>(ctx)
        .unwrap()
        .into_array()
}

fn bench_hilbert(bencher: Bencher, column: ArrayRef) {
    let mut ctx = SESSION.create_execution_ctx();
    let bounds = bounds(&mut ctx);
    bencher
        .counter(ItemsCount::new(ROWS))
        .bench_local(|| hilbert(&column, &bounds, &mut ctx));
}

#[divan::bench]
fn points(bencher: Bencher) {
    let column = point_column(
        (0..ROWS).map(ordinate).collect(),
        (0..ROWS).map(|row| ordinate(row + 1)).collect(),
    )
    .unwrap();
    bench_hilbert(bencher, column);
}

fn multipolygon(row: usize) -> MultiPolygonRings {
    let ring = |offset: usize| {
        (0..8)
            .map(|vertex| {
                (
                    ordinate(row + offset + vertex),
                    ordinate(row + offset + vertex + 1),
                )
            })
            .collect()
    };
    vec![vec![ring(0), ring(8)], vec![ring(16), ring(24)]]
}

#[divan::bench]
fn multipolygons(bencher: Bencher) {
    let column = multipolygon_column((0..ROWS).map(multipolygon).collect()).unwrap();
    bench_hilbert(bencher, column);
}
