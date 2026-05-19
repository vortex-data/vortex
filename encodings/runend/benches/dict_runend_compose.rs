// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Bench: compute `col > 5` on a `Dict(Primitive<i32>[D], RunEnd(ends, codes))` column.
//!
//! Two paths:
//! - `tree`   - apply the predicate to the encoded column; reduce rules push the
//!              comparison into the small dictionary, the result stays in `RunEnd<Bool>`
//!              until the final canonicalisation.
//! - `all_in_one` - canonicalise the column to `Primitive<i32>[N]` first, then compare.
//!
//! The dictionary has `D = 64` entries and the codes are run-length-encoded with
//! `R = 10` (i.e. average run length 10), so the column compresses ~10x.

#![expect(clippy::unwrap_used)]

use divan::Bencher;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_runend::RunEnd;

fn main() {
    divan::main();
}

const DICT_SIZE: usize = 64;
const RUN_LENGTH: usize = 10;

const N_VALUES: &[usize] = &[65_536, 200_000, 1_000_000];

/// Build `Dict(values=Primitive<i32>[64], codes=RunEnd(ends<u32>, values<u8>))` with N logical rows.
fn build_dict_runend(n: usize) -> vortex_array::ArrayRef {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();

    // RunEnd: each run of length RUN_LENGTH, the run value is i mod DICT_SIZE.
    let num_runs = n.div_ceil(RUN_LENGTH);
    let ends: Buffer<u32> = (1..=num_runs)
        .map(|i| std::cmp::min(i * RUN_LENGTH, n) as u32)
        .collect();
    let runend_values: Buffer<u8> = (0..num_runs).map(|i| (i % DICT_SIZE) as u8).collect();

    let ends_arr = PrimitiveArray::new(ends, Validity::NonNullable).into_array();
    let runend_values_arr =
        PrimitiveArray::new(runend_values, Validity::NonNullable).into_array();

    let codes = RunEnd::new(ends_arr, runend_values_arr, &mut ctx).into_array();

    // Dictionary values: 0, 3, 6, 9, ... (mix above and below the threshold 5).
    let dict_values: Buffer<i32> = (0..DICT_SIZE).map(|i| (i as i32) * 3).collect();
    let dict_values_arr = PrimitiveArray::new(dict_values, Validity::NonNullable).into_array();

    DictArray::new(codes, dict_values_arr).into_array()
}

#[divan::bench(args = N_VALUES, sample_count = 30)]
fn tree(bencher: Bencher, n: usize) {
    let col = build_dict_runend(n);
    bencher
        .with_inputs(|| (col.clone(), LEGACY_SESSION.create_execution_ctx()))
        .bench_values(|(col, mut ctx)| {
            let five = ConstantArray::new(5i32, n).into_array();
            let cmp = col.binary(five, Operator::Gt).unwrap();
            cmp.execute::<Canonical>(&mut ctx).unwrap()
        });
}

#[divan::bench(args = N_VALUES, sample_count = 30)]
fn all_in_one(bencher: Bencher, n: usize) {
    let col = build_dict_runend(n);
    bencher
        .with_inputs(|| (col.clone(), LEGACY_SESSION.create_execution_ctx()))
        .bench_values(|(col, mut ctx)| {
            // Pre-canonicalise the column to dense Primitive<i32>[N].
            let canonical = col.execute::<Canonical>(&mut ctx).unwrap().into_array();
            // Then compare on the dense array.
            let five = ConstantArray::new(5i32, n).into_array();
            let cmp = canonical.binary(five, Operator::Gt).unwrap();
            cmp.execute::<Canonical>(&mut ctx).unwrap()
        });
}

/// Decomposition: just canonicalise (decode dict + runend) - upper bound on what `all_in_one`
/// can ever beat.
#[divan::bench(args = N_VALUES, sample_count = 30)]
fn canonicalise_only(bencher: Bencher, n: usize) {
    let col = build_dict_runend(n);
    bencher
        .with_inputs(|| (col.clone(), LEGACY_SESSION.create_execution_ctx()))
        .bench_values(|(col, mut ctx)| col.execute::<Canonical>(&mut ctx).unwrap());
}

/// Decomposition: compute `col > 5` on an already-dense `Primitive<i32>[N]` - the fastest
/// possible dense compute step.
#[divan::bench(args = N_VALUES, sample_count = 30)]
fn dense_compute_only(bencher: Bencher, n: usize) {
    // Pre-build a dense Primitive[N] outside the bench loop.
    let mut setup_ctx = LEGACY_SESSION.create_execution_ctx();
    let canonical = build_dict_runend(n)
        .execute::<Canonical>(&mut setup_ctx)
        .unwrap()
        .into_array();
    bencher
        .with_inputs(|| (canonical.clone(), LEGACY_SESSION.create_execution_ctx()))
        .bench_values(|(canonical, mut ctx)| {
            let five = ConstantArray::new(5i32, n).into_array();
            let cmp = canonical.binary(five, Operator::Gt).unwrap();
            cmp.execute::<Canonical>(&mut ctx).unwrap()
        });
}

/// Decomposition: just the final `RunEnd<Bool>(N/R runs) → Bool[N]` decode step.
/// This isolates the cost of expanding a run-encoded boolean back to dense Bool.
#[divan::bench(args = N_VALUES, sample_count = 30)]
fn runend_bool_decode_only(bencher: Bencher, n: usize) {
    // Build a RunEnd<Bool> directly, with the same run structure as our dict-runend column.
    let mut setup_ctx = LEGACY_SESSION.create_execution_ctx();
    let num_runs = n.div_ceil(RUN_LENGTH);
    let ends: Buffer<u32> = (1..=num_runs)
        .map(|i| std::cmp::min(i * RUN_LENGTH, n) as u32)
        .collect();
    // Bool values per run: same shape as dict[code % 64] > 5 (mostly true, ~2/64 false).
    let bool_values: vortex_buffer::BitBufferMut = (0..num_runs)
        .map(|i| ((i % DICT_SIZE) as i32) * 3 > 5)
        .collect();
    let bool_values_arr = vortex_array::arrays::BoolArray::new(
        bool_values.freeze(),
        Validity::NonNullable,
    )
    .into_array();
    let ends_arr = PrimitiveArray::new(ends, Validity::NonNullable).into_array();
    let runend_bool =
        RunEnd::new(ends_arr, bool_values_arr, &mut setup_ctx).into_array();

    bencher
        .with_inputs(|| (runend_bool.clone(), LEGACY_SESSION.create_execution_ctx()))
        .bench_values(|(arr, mut ctx)| arr.execute::<Canonical>(&mut ctx).unwrap());
}
