// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Microbenchmark for the bounding-box pre-check used by constant-vs-column geometry predicates.
//!
//! `disjoint` rows are rejected by their bounding boxes, while `candidate` rows overlap the
//! constant's box and therefore pay for both the pre-check and the exact predicate. Comparing the
//! candidate cases tracks the worst-case overhead of the optimization.
//!
//! Run with `cargo bench -p vortex-spatial --bench predicate_bbox`.

use std::f64::consts::TAU;
use std::sync::LazyLock;

use divan::Bencher;
use divan::counter::ItemsCount;
use geo::BoundingRect;
use geo::Contains;
use geo::Intersects;
use geo_types::Geometry;
use geo_types::LineString;
use geo_types::Point;
use geo_types::Polygon;

fn main() {
    divan::main();
}

/// A candidate row runs the exact predicate against a 128-vertex query polygon, which CodSpeed's
/// CPU simulation charges far more than a desktop does. This row count keeps the candidate arms
/// inside the 1 ms per-iteration budget from `docs/developer-guide/benchmarking.md`.
const ROWS: usize = 1 << 9;
const VERTICES: usize = 128;

static QUERY: LazyLock<Geometry<f64>> = LazyLock::new(|| {
    let mut ring: Vec<_> = (0..VERTICES)
        .map(|i| {
            let theta = TAU * i as f64 / VERTICES as f64;
            (10.0 * theta.cos(), 10.0 * theta.sin())
        })
        .collect();
    ring.push(ring[0]);
    Geometry::Polygon(Polygon::new(LineString::from(ring), vec![]))
});

static DISJOINT: LazyLock<Vec<Geometry<f64>>> = LazyLock::new(|| {
    (0..ROWS)
        .map(|i| Geometry::Point(Point::new(100.0 + ordinate(i), 100.0 + ordinate(i + 1))))
        .collect()
});

static CANDIDATES: LazyLock<Vec<Geometry<f64>>> = LazyLock::new(|| {
    (0..ROWS)
        .map(|i| {
            Geometry::Point(Point::new(
                20.0 * ordinate(i) - 10.0,
                20.0 * ordinate(i + 1) - 10.0,
            ))
        })
        .collect()
});

/// Deterministic pseudo-random value in `[0, 1)`.
fn ordinate(i: usize) -> f64 {
    (i.wrapping_mul(2_654_435_761) % 10_000) as f64 / 10_000.0
}

fn exact_intersects(query: &Geometry<f64>, rows: &[Geometry<f64>]) -> usize {
    rows.iter().filter(|row| query.intersects(*row)).count()
}

fn bbox_intersects(query: &Geometry<f64>, rows: &[Geometry<f64>]) -> usize {
    let query_rect = query.bounding_rect();
    rows.iter()
        .filter(|row| {
            let rejected = query_rect
                .zip(row.bounding_rect())
                .is_some_and(|(query, row)| !query.intersects(&row));
            !rejected && query.intersects(*row)
        })
        .count()
}

fn exact_contains(query: &Geometry<f64>, rows: &[Geometry<f64>]) -> usize {
    rows.iter().filter(|row| query.contains(*row)).count()
}

fn bbox_contains(query: &Geometry<f64>, rows: &[Geometry<f64>]) -> usize {
    let query_rect = query.bounding_rect();
    rows.iter()
        .filter(|row| {
            let rejected = query_rect
                .zip(row.bounding_rect())
                .is_some_and(|(query, row)| !query.contains(&row));
            !rejected && query.contains(*row)
        })
        .count()
}

#[divan::bench]
fn intersects_exact_disjoint(bencher: Bencher) {
    bencher
        .counter(ItemsCount::new(ROWS))
        .with_inputs(|| (&*QUERY, &*DISJOINT))
        .bench_refs(|(query, rows)| exact_intersects(query, rows));
}

#[divan::bench]
fn intersects_bbox_disjoint(bencher: Bencher) {
    bencher
        .counter(ItemsCount::new(ROWS))
        .with_inputs(|| (&*QUERY, &*DISJOINT))
        .bench_refs(|(query, rows)| bbox_intersects(query, rows));
}

#[divan::bench]
fn intersects_exact_candidate(bencher: Bencher) {
    bencher
        .counter(ItemsCount::new(ROWS))
        .with_inputs(|| (&*QUERY, &*CANDIDATES))
        .bench_refs(|(query, rows)| exact_intersects(query, rows));
}

#[divan::bench]
fn intersects_bbox_candidate(bencher: Bencher) {
    bencher
        .counter(ItemsCount::new(ROWS))
        .with_inputs(|| (&*QUERY, &*CANDIDATES))
        .bench_refs(|(query, rows)| bbox_intersects(query, rows));
}

#[divan::bench]
fn contains_exact_disjoint(bencher: Bencher) {
    bencher
        .counter(ItemsCount::new(ROWS))
        .with_inputs(|| (&*QUERY, &*DISJOINT))
        .bench_refs(|(query, rows)| exact_contains(query, rows));
}

#[divan::bench]
fn contains_bbox_disjoint(bencher: Bencher) {
    bencher
        .counter(ItemsCount::new(ROWS))
        .with_inputs(|| (&*QUERY, &*DISJOINT))
        .bench_refs(|(query, rows)| bbox_contains(query, rows));
}

#[divan::bench]
fn contains_exact_candidate(bencher: Bencher) {
    bencher
        .counter(ItemsCount::new(ROWS))
        .with_inputs(|| (&*QUERY, &*CANDIDATES))
        .bench_refs(|(query, rows)| exact_contains(query, rows));
}

#[divan::bench]
fn contains_bbox_candidate(bencher: Bencher) {
    bencher
        .counter(ItemsCount::new(ROWS))
        .with_inputs(|| (&*QUERY, &*CANDIDATES))
        .bench_refs(|(query, rows)| bbox_contains(query, rows));
}
