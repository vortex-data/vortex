// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Three pipelines: apply a predicate to masked rows, producing an N-length result mask.
//!
//! Path A — dense compare: compare all N values, then AND with the base mask.
//! Path B — scattered compare: for each true position, evaluate predicate, set bit in N-length mask.
//! Path C — filter+compare+expand: filter N→true_count, dense compare over true_count,
//!          intersect_by_rank to expand back to N-length mask.
//! Path C is also broken into its individual steps for profiling.

#![allow(
    clippy::unwrap_used,
    clippy::cast_possible_truncation,
    clippy::needless_range_loop
)]

use divan::Bencher;
use vortex_buffer::BitBuffer;
use vortex_buffer::BufferMut;
use vortex_mask::Mask;

fn main() {
    divan::main();
}

const N: usize = 1_000_000;
const THRESHOLD: u64 = 500_000;

fn create_random_mask(len: usize, selectivity: f64) -> Mask {
    Mask::from_buffer(BitBuffer::from_iter((0..len).map(|i| {
        let threshold = (selectivity * 1000.0) as usize;
        (i * 7 + 13) % 1000 < threshold
    })))
}

fn create_true_indices(len: usize, selectivity: f64) -> Vec<usize> {
    (0..len)
        .filter(|i| {
            let threshold = (selectivity * 1000.0) as usize;
            (i * 7 + 13) % 1000 < threshold
        })
        .collect()
}

const DENSITIES: &[(f64, &str)] = &[
    (0.01, "1pct"),
    (0.05, "5pct"),
    (0.10, "10pct"),
    (0.15, "15pct"),
    (0.20, "20pct"),
    (0.25, "25pct"),
    (0.30, "30pct"),
    (0.35, "35pct"),
    (0.40, "40pct"),
    (0.50, "50pct"),
    (0.90, "90pct"),
];

// ── Path A: dense compare all N values, then AND with base mask ─────────────

#[divan::bench(args = DENSITIES)]
fn path_a_dense_compare(bencher: Bencher, (density, _name): (f64, &str)) {
    let base_mask = create_random_mask(N, density);
    let values: Vec<u64> = (0..N as u64).collect();
    let num_chunks = N.div_ceil(64);

    bencher.bench_local(|| {
        // Step 1: compare all N values → N-length bitmask
        let mut result_chunks: BufferMut<u64> = BufferMut::with_capacity(num_chunks);
        for chunk_i in 0..num_chunks {
            let base = chunk_i * 64;
            let mut bits = 0u64;
            let end = (base + 64).min(N);
            for i in base..end {
                if unsafe { *values.get_unchecked(i) } > THRESHOLD {
                    bits |= 1u64 << (i - base);
                }
            }
            unsafe { result_chunks.push_unchecked(bits) };
        }
        let compare_mask =
            Mask::from_buffer(BitBuffer::new(result_chunks.freeze().into_byte_buffer(), N));

        // Step 2: AND with base mask
        let result = &base_mask & &compare_mask;
        divan::black_box(result);
    });
}

// ── Path B: scattered compare at true positions → N-length mask ─────────────

#[divan::bench(args = DENSITIES)]
fn path_b_scattered_compare(bencher: Bencher, (density, _name): (f64, &str)) {
    let indices = create_true_indices(N, density);
    let values: Vec<u64> = (0..N as u64).collect();
    let num_chunks = N.div_ceil(64);
    let mut result_chunks: BufferMut<u64> = BufferMut::zeroed(num_chunks);

    bencher.bench_local(|| {
        let chunks = result_chunks.as_mut_slice();
        for chunk in chunks.iter_mut() {
            *chunk = 0;
        }
        for &idx in &indices {
            let pass = unsafe { *values.get_unchecked(idx) > THRESHOLD };
            if pass {
                unsafe {
                    *chunks.get_unchecked_mut(idx / 64) |= 1u64 << (idx % 64);
                }
            }
        }
        divan::black_box(&result_chunks);
    });
}

// ── Path C: dense compare over pre-filtered true_count + intersect_by_rank ──
//    (assumes data is already stored compacted — no filter cost)

#[divan::bench(args = DENSITIES)]
fn path_c_compare_expand(bencher: Bencher, (density, _name): (f64, &str)) {
    let base_mask = create_random_mask(N, density);
    let true_count = base_mask.true_count();
    // Pre-compacted values (as if stored in columnar format already filtered)
    let filtered: Vec<u64> = (0..true_count as u64).collect();

    bencher.bench_local(|| {
        // Step 1: dense compare over true_count → true_count-length bitmask
        let predicate_mask = Mask::from_buffer(BitBuffer::from_iter(
            filtered.iter().map(|&v| v > THRESHOLD),
        ));
        // Step 2: intersect_by_rank — expand true_count mask back to N positions
        let result = base_mask.intersect_by_rank(&predicate_mask);
        divan::black_box(result);
    });
}
