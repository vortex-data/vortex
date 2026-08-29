// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Density matrix for coverage-domain to dense-rank mask compression.

use divan::Bencher;
use vortex_buffer::BitBuffer;
use vortex_mask::Mask;

fn main() {
    divan::main();
}

const ARGS: &[(usize, usize, usize)] = &[
    (100_000, 2, 4),
    (100_000, 10, 50),
    (100_000, 100, 500),
    (1_000_000, 2, 4),
];

fn fixture(len: usize, selector_stride: usize, refined_stride: usize) -> (Mask, Mask) {
    let selector = Mask::from_buffer(BitBuffer::from_iter(
        (0..len).map(|idx| idx.is_multiple_of(selector_stride)),
    ));
    let refined = Mask::from_buffer(BitBuffer::from_iter(
        (0..len).map(|idx| idx.is_multiple_of(refined_stride)),
    ));
    (selector, refined)
}

#[divan::bench(args = ARGS)]
fn compress_by_mask(
    bencher: Bencher,
    (len, selector_stride, refined_stride): (usize, usize, usize),
) {
    let (selector, refined) = fixture(len, selector_stride, refined_stride);
    bencher
        .with_inputs(|| (&selector, &refined))
        .bench_refs(|(selector, refined)| selector.compress_by_mask(refined));
}
