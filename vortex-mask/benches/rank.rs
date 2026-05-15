// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks for `Mask::rank`.

#![allow(clippy::unwrap_used, clippy::cast_possible_truncation)]

use divan::Bencher;
use vortex_buffer::BitBuffer;
use vortex_mask::Mask;

fn main() {
    divan::main();
}

const BENCH_SIZES: &[(usize, f64)] = &[
    (1_024, 0.1),
    (1_024, 0.9),
    (16_384, 0.1),
    (16_384, 0.9),
    (65_536, 0.1),
    (65_536, 0.9),
];

fn create_mask(len: usize, density: f64) -> Mask {
    Mask::from_buffer(BitBuffer::from_iter((0..len).map(|i| {
        let threshold = (density * 1000.0) as usize;
        (i * 7 + 13) % 1000 < threshold
    })))
}

/// Single rank lookup at the midpoint.
#[divan::bench(args = BENCH_SIZES)]
fn rank_single(bencher: Bencher, (len, density): (usize, f64)) {
    let mask = create_mask(len, density);
    let mid = mask.true_count() / 2;
    bencher
        .with_inputs(|| (&mask, mid))
        .bench_refs(|(mask, mid)| mask.rank(*mid));
}
