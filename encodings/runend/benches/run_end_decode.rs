// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used, clippy::cast_possible_truncation)]

use divan::Bencher;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::compute::warm_up_vtables;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::BufferMut;
use vortex_runend::decompress_bool::runend_decode_bools;

fn main() {
    warm_up_vtables();
    divan::main();
}

/// Distribution types for bool benchmarks
#[derive(Clone, Copy)]
enum BoolDistribution {
    /// Alternating true/false (50/50)
    Alternating,
    /// Mostly true (90% true runs)
    MostlyTrue,
    /// Mostly false (90% false runs)
    MostlyFalse,
    /// All true
    AllTrue,
    /// All false
    AllFalse,
}

/// Creates bool test data with configurable distribution
fn create_bool_test_data(
    total_length: usize,
    avg_run_length: usize,
    distribution: BoolDistribution,
) -> (PrimitiveArray, BoolArray) {
    let mut ends = BufferMut::<u32>::with_capacity(total_length / avg_run_length + 1);
    let mut values = Vec::with_capacity(total_length / avg_run_length + 1);

    let mut pos = 0usize;
    let mut run_index = 0usize;

    while pos < total_length {
        let run_len = avg_run_length.min(total_length - pos);
        pos += run_len;
        ends.push(pos as u32);

        let val = match distribution {
            BoolDistribution::Alternating => run_index % 2 == 0,
            BoolDistribution::MostlyTrue => run_index % 10 != 0, // 90% true
            BoolDistribution::MostlyFalse => run_index % 10 == 0, // 10% true (90% false)
            BoolDistribution::AllTrue => true,
            BoolDistribution::AllFalse => false,
        };
        values.push(val);
        run_index += 1;
    }

    (
        PrimitiveArray::new(ends.freeze(), Validity::NonNullable),
        BoolArray::from(BitBuffer::from(values)),
    )
}

// Medium size: 10k elements with various run lengths
const BOOL_ARGS: &[(usize, usize)] = &[
    (10_000, 2),    // Very short runs (5000 runs)
    (10_000, 10),   // Short runs (1000 runs)
    (10_000, 100),  // Medium runs (100 runs)
    (10_000, 1000), // Long runs (10 runs)
];

#[divan::bench(args = BOOL_ARGS)]
fn decode_bool_alternating(bencher: Bencher, (total_length, avg_run_length): (usize, usize)) {
    let (ends, values) =
        create_bool_test_data(total_length, avg_run_length, BoolDistribution::Alternating);
    bencher.bench(|| runend_decode_bools(ends.clone(), values.clone(), 0, total_length));
}

#[divan::bench(args = BOOL_ARGS)]
fn decode_bool_mostly_true(bencher: Bencher, (total_length, avg_run_length): (usize, usize)) {
    let (ends, values) =
        create_bool_test_data(total_length, avg_run_length, BoolDistribution::MostlyTrue);
    bencher.bench(|| runend_decode_bools(ends.clone(), values.clone(), 0, total_length));
}

#[divan::bench(args = BOOL_ARGS)]
fn decode_bool_mostly_false(bencher: Bencher, (total_length, avg_run_length): (usize, usize)) {
    let (ends, values) =
        create_bool_test_data(total_length, avg_run_length, BoolDistribution::MostlyFalse);
    bencher.bench(|| runend_decode_bools(ends.clone(), values.clone(), 0, total_length));
}

#[divan::bench(args = BOOL_ARGS)]
fn decode_bool_all_true(bencher: Bencher, (total_length, avg_run_length): (usize, usize)) {
    let (ends, values) =
        create_bool_test_data(total_length, avg_run_length, BoolDistribution::AllTrue);
    bencher.bench(|| runend_decode_bools(ends.clone(), values.clone(), 0, total_length));
}

#[divan::bench(args = BOOL_ARGS)]
fn decode_bool_all_false(bencher: Bencher, (total_length, avg_run_length): (usize, usize)) {
    let (ends, values) =
        create_bool_test_data(total_length, avg_run_length, BoolDistribution::AllFalse);
    bencher.bench(|| runend_decode_bools(ends.clone(), values.clone(), 0, total_length));
}
