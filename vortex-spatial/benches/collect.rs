// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Microbenchmarks for native `ST_Collect` over homogeneous geometry lists.
//!
//! The cases cover each strict overload and the inner-null compaction path. They execute the
//! result to its canonical representation so the full multi-geometry construction is measured.
//!
//! Run with `cargo bench -p vortex-spatial --bench collect`.

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
use vortex_array::arrays::ListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_session::VortexSession;
use vortex_spatial::scalar_fn::collect::SpatialCollect;
use vortex_spatial::test_harness::linestring_column;
use vortex_spatial::test_harness::nullable_point_column;
use vortex_spatial::test_harness::point_column;
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

fn geometry_lists(elements: ArrayRef, elements_per_row: usize) -> ArrayRef {
    let offsets = PrimitiveArray::from_iter(
        (0..=ROWS).map(|row| u64::try_from(row * elements_per_row).unwrap()),
    )
    .into_array();
    ListArray::try_new(elements, offsets, Validity::NonNullable)
        .unwrap()
        .into_array()
}

fn point_lists(nullable: bool) -> ArrayRef {
    const POINTS_PER_ROW: usize = 8;
    let len = ROWS * POINTS_PER_ROW;
    let points = if nullable {
        nullable_point_column(
            (0..len)
                .map(|i| (!i.is_multiple_of(8)).then_some((i as f64, (i + 1) as f64)))
                .collect(),
        )
        .unwrap()
    } else {
        point_column(
            (0..len).map(|i| i as f64).collect(),
            (0..len).map(|i| (i + 1) as f64).collect(),
        )
        .unwrap()
    };
    geometry_lists(points, POINTS_PER_ROW)
}

fn linestring_lists() -> ArrayRef {
    const LINES_PER_ROW: usize = 4;
    let lines = linestring_column(
        (0..ROWS * LINES_PER_ROW)
            .map(|line| {
                (0..8)
                    .map(|vertex| {
                        let value = (line * 8 + vertex) as f64;
                        (value, value + 1.0)
                    })
                    .collect()
            })
            .collect(),
    )
    .unwrap();
    geometry_lists(lines, LINES_PER_ROW)
}

fn polygon_lists() -> ArrayRef {
    const POLYGONS_PER_ROW: usize = 2;
    let polygons = polygon_column(
        (0..ROWS * POLYGONS_PER_ROW)
            .map(|polygon| {
                let x = polygon as f64;
                vec![vec![
                    (x, 0.0),
                    (x + 1.0, 0.0),
                    (x + 1.0, 1.0),
                    (x, 1.0),
                    (x, 0.0),
                ]]
            })
            .collect(),
    )
    .unwrap();
    geometry_lists(polygons, POLYGONS_PER_ROW)
}

fn collect(input: &ArrayRef, ctx: &mut ExecutionCtx) -> ArrayRef {
    SpatialCollect::try_new_array(input.clone())
        .unwrap()
        .into_array()
        .execute::<Canonical>(ctx)
        .unwrap()
        .into_array()
}

fn bench_collect(bencher: Bencher, input: ArrayRef) {
    let mut ctx = SESSION.create_execution_ctx();
    bencher
        .counter(ItemsCount::new(ROWS))
        .bench_local(|| collect(&input, &mut ctx));
}

#[divan::bench]
fn points(bencher: Bencher) {
    bench_collect(bencher, point_lists(false));
}

#[divan::bench]
fn linestrings(bencher: Bencher) {
    bench_collect(bencher, linestring_lists());
}

#[divan::bench]
fn polygons(bencher: Bencher) {
    bench_collect(bencher, polygon_lists());
}

#[divan::bench]
fn nullable_points(bencher: Bencher) {
    bench_collect(bencher, point_lists(true));
}
