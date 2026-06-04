// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Strategy comparison for "process the valid elements of a mask".
//!
//! The kernel is `sum of values[i] for every set bit i` — a stand-in for any
//! validity-gated loop. We compare:
//!
//! * `per_element_value`  — `for i in 0..n { if buf.value(i) {..} }` (what callers
//!   typically write after materializing validity into a mask)
//! * `bit_iterator`       — arrow's `BitIterator` (tracks a word internally)
//! * `word_trailing_zeros`— iterate `u64` words, fast-path all-set/all-unset, and
//!   walk set bits with `trailing_zeros` / `w &= w - 1`
//! * `set_slices`         — iterate contiguous true runs (`set_slices`)
//! * `set_indices`        — iterate set positions (`set_indices`)
//!
//! Run across densities to expose the dense-vs-sparse crossover.

#![allow(clippy::unwrap_used, clippy::cast_possible_truncation)]

use divan::Bencher;
use vortex_buffer::BitBuffer;

fn main() {
    divan::main();
}

const ARGS: &[(usize, f64)] = &[
    (16_384, 0.01),
    (16_384, 0.10),
    (16_384, 0.50),
    (16_384, 0.90),
    (16_384, 0.99),
];

fn make(len: usize, density: f64) -> (BitBuffer, Vec<u64>) {
    let threshold = (density * 1000.0) as usize;
    let buf = BitBuffer::from_iter((0..len).map(|i| (i * 7 + 13) % 1000 < threshold));
    let values = (0..len as u64).collect();
    (buf, values)
}

#[divan::bench(args = ARGS)]
fn per_element_value(bencher: Bencher, (len, density): (usize, f64)) {
    let (buf, values) = make(len, density);
    bencher
        .with_inputs(|| (&buf, &values))
        .bench_refs(|(buf, values)| {
            let mut acc = 0u64;
            for i in 0..len {
                if buf.value(i) {
                    acc = acc.wrapping_add(values[i]);
                }
            }
            acc
        });
}

#[divan::bench(args = ARGS)]
fn bit_iterator(bencher: Bencher, (len, density): (usize, f64)) {
    let (buf, values) = make(len, density);
    bencher
        .with_inputs(|| (&buf, &values))
        .bench_refs(|(buf, values)| {
            let mut acc = 0u64;
            for (i, set) in buf.iter().enumerate() {
                if set {
                    acc = acc.wrapping_add(values[i]);
                }
            }
            acc
        });
}

#[divan::bench(args = ARGS)]
fn word_trailing_zeros(bencher: Bencher, (len, density): (usize, f64)) {
    let (buf, values) = make(len, density);
    bencher
        .with_inputs(|| (&buf, &values))
        .bench_refs(|(buf, values)| {
            let mut acc = 0u64;
            let chunks = buf.chunks();
            let mut base = 0usize;
            for word in chunks.iter() {
                if word == u64::MAX {
                    for k in 0..64 {
                        acc = acc.wrapping_add(values[base + k]);
                    }
                } else {
                    let mut w = word;
                    while w != 0 {
                        let b = w.trailing_zeros() as usize;
                        acc = acc.wrapping_add(values[base + b]);
                        w &= w - 1;
                    }
                }
                base += 64;
            }
            let mut w = chunks.remainder_bits();
            let rem = buf.len() - base;
            while w != 0 {
                let b = w.trailing_zeros() as usize;
                if b >= rem {
                    break;
                }
                acc = acc.wrapping_add(values[base + b]);
                w &= w - 1;
            }
            acc
        });
}

#[divan::bench(args = ARGS)]
fn for_each_set_index(bencher: Bencher, (len, density): (usize, f64)) {
    let (buf, values) = make(len, density);
    bencher
        .with_inputs(|| (&buf, &values))
        .bench_refs(|(buf, values)| {
            let mut acc = 0u64;
            buf.for_each_set_index(|i| acc = acc.wrapping_add(values[i]));
            acc
        });
}

#[divan::bench(args = ARGS)]
fn set_slices(bencher: Bencher, (len, density): (usize, f64)) {
    let (buf, values) = make(len, density);
    bencher
        .with_inputs(|| (&buf, &values))
        .bench_refs(|(buf, values)| {
            let mut acc = 0u64;
            for (start, end) in buf.set_slices() {
                for i in start..end {
                    acc = acc.wrapping_add(values[i]);
                }
            }
            acc
        });
}

#[divan::bench(args = ARGS)]
fn set_indices(bencher: Bencher, (len, density): (usize, f64)) {
    let (buf, values) = make(len, density);
    bencher
        .with_inputs(|| (&buf, &values))
        .bench_refs(|(buf, values)| {
            let mut acc = 0u64;
            for i in buf.set_indices() {
                acc = acc.wrapping_add(values[i]);
            }
            acc
        });
}
