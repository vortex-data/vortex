// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks comparing different select-in-word implementations.

#![allow(clippy::unwrap_used)]

use divan::Bencher;
use vortex_buffer::BitBuffer;

fn main() {
    divan::main();
}

// =============================================================================
// Different select_in_word implementations to compare
// =============================================================================

/// Original loop implementation: O(n) where n is the rank within the word.
#[inline]
fn select_in_word_loop(mut word: u64, mut n: usize) -> usize {
    loop {
        let tz = word.trailing_zeros() as usize;
        if n == 0 {
            return tz;
        }
        word &= word - 1;
        n -= 1;
    }
}

/// Binary search + lookup table: O(log 64) = max 3 comparisons + table lookup.
#[allow(clippy::cast_possible_truncation)]
#[inline]
fn select_in_word_binary_search(word: u64, mut n: usize) -> usize {
    let mut word = word;
    let mut pos = 0usize;

    // Check lower 32 bits
    let lower_count = (word as u32).count_ones() as usize;
    if n >= lower_count {
        n -= lower_count;
        word >>= 32;
        pos += 32;
    }

    // Check lower 16 bits
    let lower_count = ((word as u32) as u16).count_ones() as usize;
    if n >= lower_count {
        n -= lower_count;
        word >>= 16;
        pos += 16;
    }

    // Check lower 8 bits
    let lower_count = (word as u8).count_ones() as usize;
    if n >= lower_count {
        n -= lower_count;
        word >>= 8;
        pos += 8;
    }

    // Final 8 bits - use lookup table
    pos + SELECT_IN_BYTE_TABLE[(word as u8) as usize][n] as usize
}

/// Hybrid: loop for n <= 3, binary search for n > 3.
#[inline]
fn select_in_word_hybrid(word: u64, n: usize) -> usize {
    if n <= 3 {
        select_in_word_loop(word, n)
    } else {
        select_in_word_binary_search(word, n)
    }
}

/// BMI2 pdep implementation: O(1) - single instruction on supported hardware.
#[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
#[inline]
fn select_in_word_pdep(word: u64, n: usize) -> usize {
    use std::arch::x86_64::_pdep_u64;
    unsafe { _pdep_u64(1u64 << n, word).trailing_zeros() as usize }
}

/// Lookup table for select within a byte.
#[allow(clippy::cast_possible_truncation)]
static SELECT_IN_BYTE_TABLE: [[u8; 8]; 256] = {
    let mut table = [[8u8; 8]; 256];
    let mut byte = 0usize;
    while byte < 256 {
        let mut bit_pos = 0usize;
        let mut rank = 0usize;
        while bit_pos < 8 {
            if (byte >> bit_pos) & 1 == 1 {
                // bit_pos is always < 8, so fits in u8
                table[byte][rank] = bit_pos as u8;
                rank += 1;
            }
            bit_pos += 1;
        }
        byte += 1;
    }
    table
};

// =============================================================================
// Micro-benchmarks: select within a single u64 word
// =============================================================================

/// Test words with different densities
fn test_words() -> Vec<(&'static str, u64, usize)> {
    vec![
        ("sparse_1bit", 0x8000_0000_0000_0000, 0), // 1 bit set, find it
        ("sparse_4bits", 0x8000_0001_0000_0001, 2), // 4 bits, find 3rd
        ("medium_16bits", 0x5555_5555_5555_5555, 8), // 32 bits (alternating), find 9th
        ("dense_32bits", 0xFFFF_FFFF_0000_0000, 16), // 32 bits in upper half, find 17th
        ("very_dense_48bits", 0xFFFF_FFFF_FFFF_0000, 24), // 48 bits, find 25th
        ("all_set", u64::MAX, 32),                 // 64 bits, find 33rd
    ]
}

#[divan::bench(args = [0, 1, 2, 3, 4, 5])]
fn select_in_word_loop_bench(bencher: Bencher, word_idx: usize) {
    let words = test_words();
    let (_, word, n) = words[word_idx];

    bencher
        .with_inputs(|| (word, n))
        .bench_refs(|(word, n)| select_in_word_loop(*word, *n));
}

#[divan::bench(args = [0, 1, 2, 3, 4, 5])]
fn select_in_word_binsearch_bench(bencher: Bencher, word_idx: usize) {
    let words = test_words();
    let (_, word, n) = words[word_idx];

    bencher
        .with_inputs(|| (word, n))
        .bench_refs(|(word, n)| select_in_word_binary_search(*word, *n));
}

#[divan::bench(args = [0, 1, 2, 3, 4, 5])]
fn select_in_word_hybrid_bench(bencher: Bencher, word_idx: usize) {
    let words = test_words();
    let (_, word, n) = words[word_idx];

    bencher
        .with_inputs(|| (word, n))
        .bench_refs(|(word, n)| select_in_word_hybrid(*word, *n));
}

#[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
#[divan::bench(args = [0, 1, 2, 3, 4, 5])]
fn select_in_word_pdep_bench(bencher: Bencher, word_idx: usize) {
    let words = test_words();
    let (_, word, n) = words[word_idx];

    bencher
        .with_inputs(|| (word, n))
        .bench_refs(|(word, n)| select_in_word_pdep(*word, *n));
}

// =============================================================================
// Full BitBuffer::select benchmarks
// =============================================================================

const BUFFER_SIZES: &[usize] = &[1_000, 10_000, 100_000, 1_000_000];

/// Benchmark BitBuffer::select (uses the optimized implementation).
#[divan::bench(args = BUFFER_SIZES)]
fn bitbuffer_select_middle(bencher: Bencher, len: usize) {
    let buf = BitBuffer::from_iter((0..len).map(|i| i % 10 == 0));
    let target = buf.true_count() / 2;

    bencher
        .with_inputs(|| (&buf, target))
        .bench_refs(|(buf, target)| buf.select(*target));
}

/// Benchmark selecting near the start (best case for loop, equal for others).
#[divan::bench(args = BUFFER_SIZES)]
fn bitbuffer_select_start(bencher: Bencher, len: usize) {
    let buf = BitBuffer::from_iter((0..len).map(|i| i % 10 == 0));
    let target = 0; // First set bit

    bencher
        .with_inputs(|| (&buf, target))
        .bench_refs(|(buf, target)| buf.select(*target));
}

/// Benchmark selecting near the end (worst case for loop).
#[divan::bench(args = BUFFER_SIZES)]
fn bitbuffer_select_end(bencher: Bencher, len: usize) {
    let buf = BitBuffer::from_iter((0..len).map(|i| i % 10 == 0));
    let target = buf.true_count() - 1; // Last set bit

    bencher
        .with_inputs(|| (&buf, target))
        .bench_refs(|(buf, target)| buf.select(*target));
}

/// Benchmark with very sparse data (1% density).
#[divan::bench(args = BUFFER_SIZES)]
fn bitbuffer_select_sparse(bencher: Bencher, len: usize) {
    let buf = BitBuffer::from_iter((0..len).map(|i| i % 100 == 0));
    let target = buf.true_count() / 2;

    bencher
        .with_inputs(|| (&buf, target))
        .bench_refs(|(buf, target)| buf.select(*target));
}

/// Benchmark with very dense data (90% density).
#[divan::bench(args = BUFFER_SIZES)]
fn bitbuffer_select_dense(bencher: Bencher, len: usize) {
    let buf = BitBuffer::from_iter((0..len).map(|i| i % 10 != 0)); // 90% set
    let target = buf.true_count() / 2;

    bencher
        .with_inputs(|| (&buf, target))
        .bench_refs(|(buf, target)| buf.select(*target));
}
