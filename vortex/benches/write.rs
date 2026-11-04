// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks writing Vortex files on a multi-thread Tokio runtime.
//!
//! Each shape is written two ways: from a thread outside the runtime through `block_on`, as
//! blocking callers do, and as a task spawned on the runtime, as async callers do. The writer's
//! output loop runs on the calling thread in the first case and on a worker in the second, which
//! changes the cost of every handoff between them.

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex::VortexSessionDefault;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::ChunkedArray;
use vortex::array::arrays::DictArray;
use vortex::array::arrays::ListArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::VarBinViewArray;
use vortex::array::validity::Validity;
use vortex::buffer::Buffer;
use vortex::encodings::runend::RunEnd;
use vortex::file::WriteOptionsSessionExt;
use vortex::io::session::RuntimeSessionExt;
use vortex::session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

const SHAPES: &[&str] = &["narrow_ints", "wide_ints", "wide_lists", "wide_sparse_utf8"];

fn main() {
    divan::main();
}

static TOKIO: LazyLock<tokio::runtime::Runtime> = LazyLock::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(8)
        .enable_all()
        .build()
        .unwrap()
});

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let _guard = TOKIO.enter();
    VortexSession::default().with_tokio()
});

/// A struct array of `columns` fields built by `column(chunk, column)`, in `chunks` chunks.
fn table(
    columns: usize,
    chunks: usize,
    mut column: impl FnMut(usize, usize) -> ArrayRef,
) -> ArrayRef {
    let names = (0..columns).map(|c| format!("c{c}")).collect::<Vec<_>>();
    ChunkedArray::from_iter((0..chunks).map(|chunk| {
        let fields = names
            .iter()
            .enumerate()
            .map(|(c, name)| (name.as_str(), column(chunk, c)))
            .collect::<Vec<_>>();
        StructArray::from_fields(&fields).unwrap().into_array()
    }))
    .into_array()
}

fn input(shape: &str) -> ArrayRef {
    let mut rng = StdRng::seed_from_u64(0);
    let mut ctx = SESSION.create_execution_ctx();
    match shape {
        // Few columns, many rows: little coordination per byte.
        "narrow_ints" => table(8, 8, |_, _| {
            let mut v = 0i64;
            Buffer::from_iter((0..125_000).map(|_| {
                v += rng.random_range(0..100);
                v
            }))
            .into_array()
        }),
        // Many dense primitive columns.
        "wide_ints" => table(1_000, 2, |_, _| {
            Buffer::from_iter((0..5_000).map(|_| rng.random_range(0..1_000_000i64))).into_array()
        }),
        // Many small list columns, as in compress-bench's wide tables.
        "wide_lists" => table(1_000, 1, |_, _| {
            let elements = Buffer::from_iter((0..1_000).map(|_| rng.random::<i64>())).into_array();
            let offsets = Buffer::from_iter(0..=1_000u32).into_array();
            ListArray::try_new(elements, offsets, Validity::AllValid, &mut ctx)
                .unwrap()
                .into_array()
        }),
        // Many mostly-null run-end-encoded dictionary string columns, ~5% of rows set.
        "wide_sparse_utf8" => table(500, 2, |_, c| {
            let rows = 50_000;
            let values = VarBinViewArray::from_iter_str((0..8).map(|v| format!("c{c}-v{v}")));
            let (mut ends, mut codes, mut end, mut set) = (Vec::new(), Vec::new(), 0, false);
            while end < rows {
                end = (end + rng.random_range(1..if set { 32 } else { 608 })).min(rows);
                ends.push(u32::try_from(end).unwrap());
                codes.push(set.then(|| rng.random_range(0..8u8)));
                set = !set;
            }
            let dict = DictArray::try_new(
                PrimitiveArray::from_option_iter(codes).into_array(),
                values.into_array(),
            )
            .unwrap();
            RunEnd::try_new(Buffer::from(ends).into_array(), dict.into_array(), &mut ctx)
                .unwrap()
                .into_array()
        }),
        _ => unreachable!("unknown shape {shape}"),
    }
}

async fn write(array: ArrayRef) -> Vec<u8> {
    let mut buf = Vec::new();
    SESSION
        .write_options()
        .write(&mut buf, array.to_array_stream())
        .await
        .unwrap();
    buf
}

/// The calling thread drives the write through `block_on`, as `block_in_place` callers do.
#[divan::bench(args = SHAPES, sample_count = 10)]
fn block_on(bencher: Bencher, shape: &str) {
    let array = input(shape);
    bencher.bench_local(|| TOKIO.block_on(write(array.clone())));
}

/// The write runs as a task on the runtime's workers.
#[divan::bench(args = SHAPES, sample_count = 10)]
fn spawned(bencher: Bencher, shape: &str) {
    let array = input(shape);
    bencher.bench_local(|| TOKIO.block_on(TOKIO.spawn(write(array.clone()))).unwrap());
}
