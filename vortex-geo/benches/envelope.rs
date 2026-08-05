// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Microbenchmark for the `vortex.geo.envelope` scalar function: per-row bounding boxes over
//! native geometry storage.
//!
//! Cases vary the two axes that dominate the kernel's cost profile:
//! - nesting depth: `Point` (no `List` level, pure per-row reduction), `MultiPoint` (one level),
//!   `MultiPolygon` (three levels);
//! - validity: non-nullable operands, predictable sparse nulls (~10%, periodic), and
//!   unpredictable dense nulls (~50%, pseudo-random) — the worst case for branching on validity.
//!
//! All cases share the same row count, so numbers are comparable across shapes.
//!
//! Run with `cargo bench -p vortex-geo --bench envelope`.

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
use vortex_geo::scalar_fn::envelope::GeoEnvelope;
use vortex_geo::test_harness::MultiPolygonRings;
use vortex_geo::test_harness::geo_session;
use vortex_geo::test_harness::multipoint_column;
use vortex_geo::test_harness::multipolygon_column;
use vortex_geo::test_harness::nullable_multipolygon_column;
use vortex_geo::test_harness::nullable_point_column;
use vortex_geo::test_harness::point_column;
use vortex_session::VortexSession;

// Scalar function execution allocates its output inside the timed region, so use the vendored
// allocator instead of measuring glibc differences between CodSpeed runner images.
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(geo_session);

/// Every case has the same row count so results are comparable across shapes: differences then
/// reflect per-row cost (nesting depth, validity handling) rather than input size.
const ROWS: usize = 1 << 9;

/// Deterministic pseudo-random ordinate in `[0, 1000)`.
fn ordinate(i: usize) -> f64 {
    (i.wrapping_mul(2654435761) % 1000) as f64
}

/// A deterministic but unpredictable ~50% null pattern — the worst case for branching on
/// validity, since the branch predictor cannot learn it (unlike a periodic `i % k` pattern).
fn coin(i: usize) -> bool {
    (i.wrapping_mul(2654435761) >> 13) & 1 == 0
}

/// Execute the envelope of `column` to completion.
fn envelope(column: &ArrayRef, ctx: &mut ExecutionCtx) -> ArrayRef {
    GeoEnvelope::try_new_array(column.clone())
        .unwrap()
        .into_array()
        .execute::<Canonical>(ctx)
        .unwrap()
        .into_array()
}

#[divan::bench]
fn point_non_nullable(bencher: Bencher) {
    let xs = (0..ROWS).map(ordinate).collect();
    let ys = (0..ROWS).map(|i| ordinate(i + 1)).collect();
    let column = point_column(xs, ys).unwrap();
    let mut ctx = SESSION.create_execution_ctx();
    bencher
        .counter(ItemsCount::new(ROWS))
        .bench_local(|| envelope(&column, &mut ctx));
}

#[divan::bench]
fn point_mixed_validity(bencher: Bencher) {
    let points = (0..ROWS)
        .map(|i| (!i.is_multiple_of(10)).then(|| (ordinate(i), ordinate(i + 1))))
        .collect();
    let column = nullable_point_column(points).unwrap();
    let mut ctx = SESSION.create_execution_ctx();
    bencher
        .counter(ItemsCount::new(ROWS))
        .bench_local(|| envelope(&column, &mut ctx));
}

#[divan::bench]
fn point_random_nulls(bencher: Bencher) {
    let points = (0..ROWS)
        .map(|i| (!coin(i)).then(|| (ordinate(i), ordinate(i + 1))))
        .collect();
    let column = nullable_point_column(points).unwrap();
    let mut ctx = SESSION.create_execution_ctx();
    bencher
        .counter(ItemsCount::new(ROWS))
        .bench_local(|| envelope(&column, &mut ctx));
}

const POINTS_PER_ROW: usize = 32;

#[divan::bench]
fn multipoint_non_nullable(bencher: Bencher) {
    let rows = (0..ROWS)
        .map(|r| {
            (0..POINTS_PER_ROW)
                .map(|i| (ordinate(r + i), ordinate(r + i + 1)))
                .collect()
        })
        .collect();
    let column = multipoint_column(rows).unwrap();
    let mut ctx = SESSION.create_execution_ctx();
    bencher
        .counter(ItemsCount::new(ROWS))
        .bench_local(|| envelope(&column, &mut ctx));
}

const VERTICES_PER_RING: usize = 8;

/// A multipolygon of two polygons with two rings each (8 vertices per ring, 32 per row), so the
/// intermediate list levels have non-identity offsets.
fn multipolygon_row(r: usize) -> MultiPolygonRings {
    let ring = |p: usize| {
        (0..VERTICES_PER_RING)
            .map(|i| (ordinate(r + p + i), ordinate(r + p + i + 1)))
            .collect()
    };
    vec![vec![ring(0), ring(1)], vec![ring(2), ring(3)]]
}

#[divan::bench]
fn multipolygon_non_nullable(bencher: Bencher) {
    let rows = (0..ROWS).map(multipolygon_row).collect();
    let column = multipolygon_column(rows).unwrap();
    let mut ctx = SESSION.create_execution_ctx();
    bencher
        .counter(ItemsCount::new(ROWS))
        .bench_local(|| envelope(&column, &mut ctx));
}

#[divan::bench]
fn multipolygon_mixed_validity(bencher: Bencher) {
    let rows = (0..ROWS)
        .map(|r| (!r.is_multiple_of(10)).then(|| multipolygon_row(r)))
        .collect();
    let column = nullable_multipolygon_column(rows).unwrap();
    let mut ctx = SESSION.create_execution_ctx();
    bencher
        .counter(ItemsCount::new(ROWS))
        .bench_local(|| envelope(&column, &mut ctx));
}

#[divan::bench]
fn multipolygon_random_nulls(bencher: Bencher) {
    let rows = (0..ROWS)
        .map(|r| (!coin(r)).then(|| multipolygon_row(r)))
        .collect();
    let column = nullable_multipolygon_column(rows).unwrap();
    let mut ctx = SESSION.create_execution_ctx();
    bencher
        .counter(ItemsCount::new(ROWS))
        .bench_local(|| envelope(&column, &mut ctx));
}
