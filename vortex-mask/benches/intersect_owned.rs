// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compares repeated binary mask intersection with the one-pass owned collection path.

use std::ops::BitAnd;

use divan::Bencher;
use vortex_buffer::BitBuffer;
use vortex_mask::Mask;

fn main() {
    divan::main();
}

const ARGS: &[(usize, usize)] = &[(131_072, 3), (131_072, 8), (131_072, 14), (131_077, 8)];

fn masks(len: usize, count: usize) -> Vec<Mask> {
    (0..count)
        .map(|mask_index| {
            let buffer = BitBuffer::from_iter((0..len).map(|index| {
                index
                    .wrapping_mul(17 + mask_index * 2)
                    .wrapping_add(mask_index * 13)
                    % 101
                    < 73
            }));
            Mask::from_buffer(buffer)
        })
        .collect()
}

#[divan::bench(args = ARGS)]
fn repeated_binary(bencher: Bencher, (len, count): (usize, usize)) {
    bencher
        .with_inputs(|| masks(len, count))
        .bench_values(|masks| {
            masks
                .into_iter()
                .reduce(|left, right| left.bitand(&right))
                .unwrap_or_else(|| Mask::new_true(len))
        });
}

#[divan::bench(args = ARGS)]
fn owned_buffers_one_count(bencher: Bencher, (len, count): (usize, usize)) {
    bencher
        .with_inputs(|| masks(len, count))
        .bench_values(Mask::intersect_owned);
}
