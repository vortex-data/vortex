// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Prototype "compressed mask" experiment: filter a dense value column using a `RunEnd<Bool>`
//! mask, without going through the usual `Mask::from_buffer(BitBuffer)` materialisation.
//!
//! - `compressed_mask_filter` walks the run-encoded mask directly, copying contiguous blocks
//!    of values per true-run. O(num_runs + true_elements).
//! - `decode_then_filter` decodes the mask to dense Bool[N], builds a `Mask`, then runs the
//!    standard Vortex filter on the Primitive value column. The current `Mask` API forces
//!    this path - it has no compressed variant.
//!
//! This shows the ceiling for "decompression with filter": what the planner could buy if
//! `Mask` supported a `RunEnd` variant or if `Filter` could consume a compressed bool array
//! directly.
//!
//! Built with `target-cpu=native` for AVX-512.

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation)]

use divan::Bencher;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::bool::BoolArrayExt;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_mask::Mask;
use vortex_runend::RunEnd;
use vortex_runend::RunEndArray;
use vortex_runend::RunEndArrayExt;

fn main() {
    divan::main();
}

const N: usize = 1_000_000;

// Build the dense value column: Primitive<i64> with `i` at position `i`.
fn build_values(n: usize) -> ArrayRef {
    let buf: Buffer<i64> = (0..n as i64).collect();
    PrimitiveArray::new(buf, Validity::NonNullable).into_array()
}

// Build a `RunEnd<Bool>` mask of length `n` where each run is `r` elements long, and the
// bool value cycles `true, true, ..., false` such that the overall true selectivity is
// `selectivity_pct%`.
fn build_runend_bool_mask(n: usize, r: usize, selectivity_pct: usize) -> RunEndArray {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let num_runs = n.div_ceil(r);
    let ends_buf: Buffer<u32> = (1..=num_runs)
        .map(|i| std::cmp::min(i * r, n) as u32)
        .collect();
    // Run-bool pattern: for every 100 runs, mark `selectivity_pct` as true.
    let bool_vals: BitBuffer = BitBuffer::from_indices(
        num_runs,
        (0..num_runs).filter(|i| i % 100 < selectivity_pct),
    );
    let ends = PrimitiveArray::new(ends_buf, Validity::NonNullable).into_array();
    let bools = BoolArray::new(bool_vals, Validity::NonNullable).into_array();
    RunEnd::new(ends, bools, &mut ctx)
}

/// Walk the run-end mask directly, copying value blocks for each true run.
/// This is what a `Mask::RunEnd` variant + a fused Filter kernel would do.
fn filter_with_compressed_mask(
    values: &[i64],
    runend_mask: &RunEndArray,
) -> Buffer<i64> {
    let bool_vals = runend_mask
        .values()
        .clone()
        .execute::<BoolArray>(&mut LEGACY_SESSION.create_execution_ctx())
        .unwrap();
    let ends_arr = runend_mask
        .ends()
        .clone()
        .execute::<PrimitiveArray>(&mut LEGACY_SESSION.create_execution_ctx())
        .unwrap();
    let bool_buf = bool_vals.to_bit_buffer();
    let ends = ends_arr.as_slice::<u32>();

    // Upper bound: at most N elements survive.
    let mut out = BufferMut::<i64>::with_capacity(values.len());
    let mut prev_end = 0usize;
    for (i, &end) in ends.iter().enumerate() {
        let end = end as usize;
        if bool_buf.value(i) {
            out.extend_from_slice(&values[prev_end..end]);
        }
        prev_end = end;
    }
    out.freeze()
}

/// "Decode then filter" baseline: turn the RunEnd<Bool> into a dense Bool[N], wrap as Mask,
/// run the standard Vortex filter.
fn decode_then_filter(values: &ArrayRef, mask_arr: &ArrayRef) -> ArrayRef {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    // Canonical of RunEnd<Bool> is BoolArray[N].
    let canon = mask_arr
        .clone()
        .execute::<Canonical>(&mut ctx)
        .unwrap()
        .into_array();
    let bool_arr = canon.as_::<vortex_array::arrays::Bool>();
    let bit_buf = bool_arr.to_bit_buffer();
    let mask = Mask::from_buffer(bit_buf);
    let filtered = values.filter(mask).unwrap();
    filtered
        .execute::<Canonical>(&mut ctx)
        .unwrap()
        .into_array()
}

// (run_length, selectivity_pct) cases encoded as `r * 1000 + sel` so divan can use them as
// a single `args` parameter. e.g. 10_050 = R=10, 50% selectivity.
fn unpack(packed: usize) -> (usize, usize) {
    (packed / 1_000, packed % 1_000)
}

const CASES: &[usize] = &[
    10_050, 10_010, 10_001, 100_050, 100_010, 100_001,
];

#[divan::bench(args = CASES, sample_count = 30)]
fn compressed_mask_filter(bencher: Bencher, packed: usize) {
    let (r, sel) = unpack(packed);
    let values = build_values(N);
    let values_buf = values.as_::<vortex_array::arrays::Primitive>();
    let values_slice = values_buf.as_slice::<i64>();
    bencher
        .with_inputs(|| build_runend_bool_mask(N, r, sel))
        .bench_values(|mask| filter_with_compressed_mask(values_slice, &mask));
}

#[divan::bench(args = CASES, sample_count = 30)]
fn decode_then_filter_bench(bencher: Bencher, packed: usize) {
    let (r, sel) = unpack(packed);
    let values = build_values(N);
    bencher
        .with_inputs(|| build_runend_bool_mask(N, r, sel).into_array())
        .bench_values(|mask_arr| decode_then_filter(&values, &mask_arr));
}

/// Sanity test - run via `cargo test --release -p vortex-runend --bench runend_mask_filter`.
#[cfg(test)]
mod tests {
    use super::build_runend_bool_mask;
    use super::build_values;
    use super::decode_then_filter;
    use super::filter_with_compressed_mask;
    use super::N;
    use vortex_array::IntoArray;

    #[test]
    fn paths_agree() {
        let (r, sel) = (10, 50);
        let values = build_values(N);
        let values_buf = values.as_::<vortex_array::arrays::Primitive>();
        let values_slice = values_buf.as_slice::<i64>();
        let mask = build_runend_bool_mask(N, r, sel);
        let mask_arr = mask.clone().into_array();

        let tree_out = filter_with_compressed_mask(values_slice, &mask);
        let dense_out = decode_then_filter(&values, &mask_arr);
        let dense_buf = dense_out.as_::<vortex_array::arrays::Primitive>();
        let dense_slice = dense_buf.as_slice::<i64>();
        assert_eq!(tree_out.len(), dense_slice.len());
        assert_eq!(tree_out.as_ref(), dense_slice);
    }
}
