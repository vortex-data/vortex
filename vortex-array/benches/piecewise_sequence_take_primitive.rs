// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Primitive take benchmarks for [`PiecewiseSequenceArray`] length representations.

#![allow(clippy::cast_possible_truncation)]
#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use divan::counter::BytesCount;
use mimalloc::MiMalloc;
use rand::SeedableRng;
use rand::prelude::*;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PiecewiseSequenceArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::memory::MemorySession;
use vortex_array::optimizer::kernels::KernelSession;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    VortexSession::empty()
        .with::<KernelSession>()
        .with::<MemorySession>()
});

const SOURCE_LEN: usize = 512 * 1024;
// Sized to keep CodSpeed simulation under 1ms at run length 1.
const OUTPUT_LEN: usize = 4 * 1024;
const RUN_LENGTHS: &[usize] = &[1, 4, 16, 64, 256];

#[divan::bench(args = RUN_LENGTHS)]
fn optimized_constant_lengths(bencher: Bencher, run_length: usize) {
    let values = values();
    let indices = piecewise_indices(
        run_length,
        lengths_constant(run_length, run_count(run_length)),
    );

    bencher
        .counter(BytesCount::of_many::<i64>(OUTPUT_LEN))
        .with_inputs(|| (&values, indices.clone(), SESSION.create_execution_ctx()))
        .bench_refs(|(values, indices, ctx)| {
            divan::black_box(
                values
                    .clone()
                    .take(indices.clone())
                    .unwrap()
                    .execute::<PrimitiveArray>(ctx)
                    .unwrap(),
            );
        });
}

#[divan::bench(args = RUN_LENGTHS)]
fn prebuilt_dense_lengths(bencher: Bencher, run_length: usize) {
    let values = values();
    let indices = piecewise_indices(run_length, lengths_dense(run_length, run_count(run_length)));

    bencher
        .counter(BytesCount::of_many::<i64>(OUTPUT_LEN))
        .with_inputs(|| (&values, indices.clone(), SESSION.create_execution_ctx()))
        .bench_refs(|(values, indices, ctx)| {
            divan::black_box(
                values
                    .clone()
                    .take(indices.clone())
                    .unwrap()
                    .execute::<PrimitiveArray>(ctx)
                    .unwrap(),
            );
        });
}

fn values() -> ArrayRef {
    PrimitiveArray::from_iter((0..SOURCE_LEN).map(|idx| idx as i64)).into_array()
}

fn run_count(run_length: usize) -> usize {
    assert_eq!(OUTPUT_LEN % run_length, 0);
    OUTPUT_LEN / run_length
}

fn piecewise_indices(run_length: usize, lengths: ArrayRef) -> ArrayRef {
    let run_count = run_count(run_length);
    let starts = starts(run_count, run_length).into_array();
    let multipliers = ConstantArray::new(1u64, run_count).into_array();

    // SAFETY: starts, lengths, and unit multipliers have the same run count, are non-null unsigned
    // integer arrays, and expand to OUTPUT_LEN by construction.
    unsafe { PiecewiseSequenceArray::new_unchecked(starts, lengths, multipliers, OUTPUT_LEN) }
        .into_array()
}

fn lengths_constant(run_length: usize, run_count: usize) -> ArrayRef {
    ConstantArray::new(run_length as u64, run_count).into_array()
}

fn lengths_dense(run_length: usize, run_count: usize) -> ArrayRef {
    PrimitiveArray::from_iter((0..run_count).map(|_| run_length as u64)).into_array()
}

fn starts(run_count: usize, run_length: usize) -> PrimitiveArray {
    let mut rng = StdRng::seed_from_u64(0x51ced);
    let max_start = SOURCE_LEN - run_length;
    PrimitiveArray::from_iter((0..run_count).map(|_| rng.random_range(0..=max_start) as u64))
}
