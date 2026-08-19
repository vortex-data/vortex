// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Microbenchmarks for native `ST_Length` over LineStrings.
//!
//! The two-vertex case tracks ordinary route segments, while the longer-line case captures the
//! per-vertex traversal cost. The nullable case measures strict null propagation separately.
//!
//! `ROWS` keeps each case near the roughly 1 ms iteration budget recommended for CodSpeed.
//!
//! Run with `cargo bench -p vortex-spatial --bench length`.

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
use vortex_spatial::scalar_fn::length::SpatialLength;
use vortex_spatial::test_harness::linestring_column;
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

/// A deterministic vertex ordinate.
fn ordinate(i: usize) -> f64 {
    (i.wrapping_mul(2_654_435_761) % 10_000) as f64 / 100.0
}

fn linestrings(vertices: usize) -> ArrayRef {
    linestring_column(
        (0..ROWS)
            .map(|row| {
                (0..vertices)
                    .map(|vertex| (ordinate(row + vertex), ordinate(row + vertex + 1)))
                    .collect()
            })
            .collect(),
    )
    .unwrap()
}

fn lengths(lines: &ArrayRef, ctx: &mut ExecutionCtx) -> ArrayRef {
    SpatialLength::try_new(lines.clone())
        .unwrap()
        .into_array()
        .execute::<Canonical>(ctx)
        .unwrap()
        .into_array()
}

#[divan::bench]
fn two_vertex_lines(bencher: Bencher) {
    let lines = linestrings(2);
    let mut ctx = SESSION.create_execution_ctx();
    bencher
        .counter(ItemsCount::new(ROWS))
        .bench_local(|| lengths(&lines, &mut ctx));
}

#[divan::bench]
fn sixteen_vertex_lines(bencher: Bencher) {
    let lines = linestrings(16);
    let mut ctx = SESSION.create_execution_ctx();
    bencher
        .counter(ItemsCount::new(ROWS))
        .bench_local(|| lengths(&lines, &mut ctx));
}

#[divan::bench]
fn nullable_two_vertex_lines(bencher: Bencher) {
    let lines = MaskedArray::try_new(
        linestrings(2),
        Validity::from_iter((0..ROWS).map(|i| !i.is_multiple_of(8))),
    )
    .unwrap()
    .into_array();
    let mut ctx = SESSION.create_execution_ctx();
    bencher
        .counter(ItemsCount::new(ROWS))
        .bench_local(|| lengths(&lines, &mut ctx));
}
