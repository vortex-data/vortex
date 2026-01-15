// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks for BitBuffer::select operation.
//!
//! Tests performance of the optimized select implementation with:
//! - Block8: Process 8 chunks at a time for instruction-level parallelism
//! - Bidirectional: Search from end if target > 50% for up to 60x speedup
//! - UnalignedBitChunk: Aligned loads for unaligned data

#![allow(clippy::unwrap_used)]

use divan::Bencher;
use vortex_buffer::BitBuffer;

fn main() {
    divan::main();
}

const SIZES: &[usize] = &[100_000];
const PERCENTILES: &[usize] = &[2, 10, 25, 50, 75, 90, 98];

// =============================================================================
// Test data generators
// =============================================================================

fn make_aligned_buf(len: usize) -> BitBuffer {
    BitBuffer::from_iter((0..len).map(|i| i % 10 == 0)) // 10% density
}

fn make_unaligned_buf(len: usize) -> BitBuffer {
    let buf = BitBuffer::from_iter((0..len + 1).map(|i| i % 10 == 0));
    buf.slice(1..len + 1)
}

// =============================================================================
// Benchmarks
// =============================================================================

#[divan::bench(args = PERCENTILES)]
fn aligned_select(bencher: Bencher, pct: usize) {
    let buf = make_aligned_buf(SIZES[0]);
    let true_count = buf.true_count();
    let target = true_count * pct / 100;
    bencher
        .with_inputs(|| (&buf, target, true_count))
        .bench_refs(|(b, t, tc)| b.select_with_true_count(*t, *tc));
}

#[divan::bench(args = PERCENTILES)]
fn unaligned_select(bencher: Bencher, pct: usize) {
    let buf = make_unaligned_buf(SIZES[0]);
    let true_count = buf.true_count();
    let target = true_count * pct / 100;
    bencher
        .with_inputs(|| (&buf, target, true_count))
        .bench_refs(|(b, t, tc)| b.select_with_true_count(*t, *tc));
}

/// Benchmark select without pre-computed true_count (includes popcount overhead)
#[divan::bench(args = PERCENTILES)]
fn aligned_select_no_true_count(bencher: Bencher, pct: usize) {
    let buf = make_aligned_buf(SIZES[0]);
    let true_count = buf.true_count();
    let target = true_count * pct / 100;
    bencher
        .with_inputs(|| (&buf, target))
        .bench_refs(|(b, t)| b.select(*t));
}
