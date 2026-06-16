// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks comparing strategies for computing a bitwise-op result *and* its `true_count`:
//!
//! * `*_two_pass`: current behaviour, materialise the result buffer then run the (vectorised)
//!   `count_ones` kernel as a separate pass (`&a & &b` followed by `.true_count()`).
//! * `*_fused`: single pass that combines the words and accumulates `count_ones` together
//!   (`BitBuffer::bitand_with_true_count`).

#![allow(clippy::unwrap_used, clippy::cast_possible_truncation)]

use divan::Bencher;
use divan::black_box;
use vortex_buffer::BitBuffer;

fn main() {
    divan::main();
}

// Lengths in bits, from L1-resident up to L2/L3.
const SIZES: &[usize] = &[
    1_024,   // 128 B
    8_192,   // 1 KB
    65_536,  // 8 KB
    524_288, // 64 KB
];

fn inputs(len: usize) -> (BitBuffer, BitBuffer) {
    let lhs =
        BitBuffer::from_iter((0..len).map(|i| (i.wrapping_mul(2_654_435_761) >> 13) & 1 == 0));
    let rhs = BitBuffer::from_iter((0..len).map(|i| (i.wrapping_mul(40_503) >> 7) & 1 == 0));
    (lhs, rhs)
}

#[divan::bench(args = SIZES)]
fn and_two_pass(bencher: Bencher, len: usize) {
    let (lhs, rhs) = inputs(len);
    bencher.bench_local(|| {
        let result = black_box(&lhs) & black_box(&rhs);
        let count = result.true_count();
        (result, count)
    });
}

#[divan::bench(args = SIZES)]
fn and_fused(bencher: Bencher, len: usize) {
    let (lhs, rhs) = inputs(len);
    bencher.bench_local(|| black_box(&lhs).bitand_with_true_count(black_box(&rhs)));
}

// Owned-LHS scenario: the current code can combine in-place (no allocation) then count
// separately; the fused path always allocates but does a single pass. A fresh owned, uniquely
// owned LHS is cloned per iteration so the in-place fast path is actually taken.
#[divan::bench(args = SIZES)]
fn and_owned_two_pass(bencher: Bencher, len: usize) {
    let (lhs, rhs) = inputs(len);
    bencher
        .with_inputs(|| lhs.clone())
        .bench_local_values(|lhs| {
            let result = lhs & black_box(&rhs);
            let count = result.true_count();
            (result, count)
        });
}

#[divan::bench(args = SIZES)]
fn and_owned_fused(bencher: Bencher, len: usize) {
    let (lhs, rhs) = inputs(len);
    bencher
        .with_inputs(|| lhs.clone())
        .bench_local_values(|lhs| lhs.bitand_with_true_count(black_box(&rhs)));
}
