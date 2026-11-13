// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Sorted searches over compressed children: reads, through one-off and repeated probes, of a
//! Sparse array whose patch indices are BitPacked and of a RunEnd array whose ends are BitPacked,
//! and slices of those patches and of that RunEnd array. Each read or slice binary searches the compressed child, so
//! these measure how much of the child's preparation is shared across searches.

use std::ops::Range;
use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::session::AggregateFnSession;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::RunEnd;
use vortex_array::memory::MemorySession;
use vortex_array::optimizer::kernels::KernelSession;
use vortex_array::scalar::Scalar;
use vortex_array::session::ArraySession;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_fastlanes::bitpack_compress::bitpack_to_best_bit_width;
use vortex_session::VortexSession;
use vortex_sparse::Sparse;
use vortex_sparse::SparseArray;
use vortex_sparse::SparseExt;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty()
        .with::<ArraySession>()
        .with::<KernelSession>()
        .with::<AggregateFnSession>()
        .with::<MemorySession>();
    vortex_fastlanes::initialize(&session);
    vortex_sparse::initialize(&session);
    session
});

const LEN: usize = 1 << 20;
/// One patch, or one run end, every `STRIDE` rows.
const STRIDE: usize = 64;
// Sized to keep the CodSpeed simulation short per benchmark.
const ACCESSES: usize = 256;
const SLICES: usize = 64;
const SLICE_LEN: usize = 4_096;

/// Sorted positions every `STRIDE` rows, BitPacked.
fn bitpacked_positions(first: usize) -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    let positions = (first..=LEN)
        .step_by(STRIDE)
        .map(|i| u32::try_from(i).vortex_expect("positions fit u32"))
        .collect::<Buffer<u32>>();
    bitpack_to_best_bit_width(
        &PrimitiveArray::new(positions, Validity::NonNullable),
        &mut ctx,
    )
    .vortex_expect("bitpack positions")
    .into_array()
}

fn values(len: usize) -> ArrayRef {
    (0..len)
        .map(|i| i64::try_from(i).vortex_expect("values fit i64"))
        .collect::<Buffer<i64>>()
        .into_array()
}

fn sparse() -> SparseArray {
    let mut ctx = SESSION.create_execution_ctx();
    // Patch indices must stay below the array length.
    let indices = bitpacked_positions(STRIDE / 2);
    let values = values(indices.len());
    Sparse::try_new(indices, values, LEN, Scalar::from(-1i64), &mut ctx)
        .vortex_expect("sparse fixture")
}

fn run_end() -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    // Run ends are exclusive, so the last one is the array length.
    let ends = bitpacked_positions(STRIDE);
    let values = values(ends.len());
    RunEnd::try_new(ends, values, &mut ctx)
        .vortex_expect("run-end fixture")
        .into_array()
}

/// A fixed pseudo-random sequence below `bound`.
fn positions(count: usize, bound: usize) -> Vec<usize> {
    let mut seed = 42u64;
    (0..count)
        .map(|_| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            (seed >> 32) as usize % bound
        })
        .collect()
}

fn ranges() -> Vec<Range<usize>> {
    positions(SLICES, LEN - SLICE_LEN)
        .into_iter()
        .map(|start| start..start + SLICE_LEN)
        .collect()
}

#[divan::bench]
fn sparse_read_once(bencher: Bencher) {
    let array = sparse().into_array();
    let rows = positions(ACCESSES, LEN);
    bencher
        .with_inputs(|| SESSION.create_execution_ctx())
        .bench_refs(|ctx| {
            for &row in &rows {
                divan::black_box(array.execute_scalar(row, ctx).vortex_expect("read"));
            }
        });
}

#[divan::bench]
fn sparse_read_repeated(bencher: Bencher) {
    let array = sparse().into_array();
    let rows = positions(ACCESSES, LEN);
    bencher
        .with_inputs(|| (SESSION.create_execution_ctx(), array.repeated_probe()))
        .bench_refs(|(ctx, probe)| {
            for &row in &rows {
                divan::black_box(probe.execute_scalar(row, ctx).vortex_expect("read"));
            }
        });
}

#[divan::bench]
fn run_end_read_once(bencher: Bencher) {
    let array = run_end();
    let rows = positions(ACCESSES, LEN);
    bencher
        .with_inputs(|| SESSION.create_execution_ctx())
        .bench_refs(|ctx| {
            for &row in &rows {
                divan::black_box(array.execute_scalar(row, ctx).vortex_expect("read"));
            }
        });
}

#[divan::bench]
fn run_end_read_repeated(bencher: Bencher) {
    let array = run_end();
    let rows = positions(ACCESSES, LEN);
    bencher
        .with_inputs(|| (SESSION.create_execution_ctx(), array.repeated_probe()))
        .bench_refs(|(ctx, probe)| {
            for &row in &rows {
                divan::black_box(probe.execute_scalar(row, ctx).vortex_expect("read"));
            }
        });
}

#[divan::bench]
fn patches_slice(bencher: Bencher) {
    let patches = sparse().patches();
    let ranges = ranges();
    bencher
        .with_inputs(|| SESSION.create_execution_ctx())
        .bench_refs(|ctx| {
            for range in &ranges {
                divan::black_box(patches.slice(range.clone(), ctx).vortex_expect("slice"));
            }
        });
}

#[divan::bench]
fn run_end_slice(bencher: Bencher) {
    let array = run_end();
    let ranges = ranges();
    bencher
        .with_inputs(|| SESSION.create_execution_ctx())
        .bench_refs(|ctx| {
            for range in &ranges {
                divan::black_box(
                    array
                        .slice(range.clone())
                        .vortex_expect("slice")
                        .execute_until::<RunEnd>(ctx)
                        .vortex_expect("slice kernel"),
                );
            }
        });
}
