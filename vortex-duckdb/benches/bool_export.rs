// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Microbenchmark for the bit -> byte-bool unpack performed by `BoolExporter::export`.
//!
//! DuckDB stores booleans as one byte per value (`&mut [bool]`), while Vortex stores them
//! bit-packed in a [`BitBuffer`]. On every boolean column export we unpack a slice of the
//! bit buffer into the destination byte slice.
//!
//! This bench isolates that pure unpacking logic so it can run without a live DuckDB
//! vector: a reused `Vec<bool>` stands in for the DuckDB byte-bool destination. It compares
//! the previous implementation (`.iter().collect::<Vec<bool>>()` followed by
//! `copy_from_slice`) against the new allocation-free direct zip-write.

use divan::Bencher;
use vortex::buffer::BitBuffer;

fn main() {
    divan::main();
}

const SIZES: &[usize] = &[1024, 16_384];

/// Bit densities (true ratio), expressed as `(numerator, denominator)`.
const DENSITIES: &[(usize, usize)] = &[(1, 2), (1, 10), (9, 10)];

fn make_buffer(len: usize, density: (usize, usize)) -> BitBuffer {
    let (num, den) = density;
    // `+ 8` so we can slice off a non-byte-aligned offset, matching real exports.
    BitBuffer::from_iter((0..len + 8).map(|i| (i % den) < num))
}

/// Previous implementation: allocate a throwaway `Vec<bool>` then copy into the destination.
#[divan::bench(args = SIZES, consts = [0usize, 1usize, 2usize])]
fn old_collect_copy<const DENSITY_IDX: usize>(bencher: Bencher, len: usize) {
    let buffer = make_buffer(len, DENSITIES[DENSITY_IDX]);
    bencher
        .with_inputs(|| vec![false; len])
        .bench_refs(|dst: &mut Vec<bool>| {
            // Offset by 1 to exercise a non-byte-aligned slice like the export hot path.
            dst.copy_from_slice(&buffer.slice(1..(1 + len)).iter().collect::<Vec<bool>>());
            divan::black_box(&dst);
        });
}

/// New implementation: zip the sliced bit iterator directly into the destination slice.
#[divan::bench(args = SIZES, consts = [0usize, 1usize, 2usize])]
fn new_zip_write<const DENSITY_IDX: usize>(bencher: Bencher, len: usize) {
    let buffer = make_buffer(len, DENSITIES[DENSITY_IDX]);
    bencher
        .with_inputs(|| vec![false; len])
        .bench_refs(|dst: &mut Vec<bool>| {
            for (slot, bit) in dst.iter_mut().zip(buffer.slice(1..(1 + len)).iter()) {
                *slot = bit;
            }
            divan::black_box(&dst);
        });
}
