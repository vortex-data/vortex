// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Microbenchmark comparing three bit-packing strategies for `dfa_scan_to_bitbuf`:
//!
//! 1. **manual_word** — current: pack directly into u64 words one bit at a time
//! 2. **collect_bool** — use `BitBuffer::collect_bool` with a closure
//! 3. **bool_buf_64** — write results into `[bool; 64]` stack buffer, then compress
//!
//! Uses a trivial matcher (single byte comparison) so that the packing
//! overhead dominates rather than per-string work.

#![allow(clippy::unwrap_used, clippy::cast_possible_truncation)]

use divan::Bencher;
use vortex_buffer::BitBuffer;
use vortex_buffer::BufferMut;

fn main() {
    divan::main();
}

// ---------------------------------------------------------------------------
// Test data: precomputed bool results + offsets (to keep the offset-reading
// overhead consistent across strategies while making the matcher trivial)
// ---------------------------------------------------------------------------

const N: usize = 100_000;

struct TestData {
    /// Precomputed match results for each of the N "strings".
    results: Vec<bool>,
    /// Fake offsets array (N+1 entries) so the offset-reading overhead is
    /// included, matching the real `dfa_scan_to_bitbuf` pattern.
    offsets: Vec<u32>,
    /// Single-byte "strings" — just used so the matcher reads *something*.
    bytes: Vec<u8>,
}

impl TestData {
    fn new() -> Self {
        let mut rng_state: u64 = 0xDEAD_BEEF;
        let mut next = || -> u64 {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            rng_state >> 33
        };

        // Each "string" is exactly 1 byte so the matcher is essentially free.
        let bytes: Vec<u8> = (0..N).map(|_| (next() % 256) as u8).collect();
        let offsets: Vec<u32> = (0..=N).map(|i| i as u32).collect();
        let results: Vec<bool> = bytes.iter().map(|&b| b >= 128).collect();

        Self {
            results,
            offsets,
            bytes,
        }
    }
}

// Trivial matcher: single byte check.  The real work being benchmarked is
// the bit-packing loop, not this function.
#[inline(always)]
fn matcher(data: &[u8]) -> bool {
    // SAFETY: benchmark guarantees non-empty slices
    unsafe { *data.get_unchecked(0) >= 128 }
}

// ---------------------------------------------------------------------------
// Strategy 1: manual word packing (current implementation)
// ---------------------------------------------------------------------------

#[inline(never)]
fn scan_manual_word(offsets: &[u32], all_bytes: &[u8], n: usize, negated: bool) -> BitBuffer {
    let n_words = n / 64;
    let remainder = n % 64;
    let mut words: BufferMut<u64> = BufferMut::with_capacity(n.div_ceil(64));

    for chunk in 0..n_words {
        let base = chunk * 64;
        let mut word = 0u64;
        let mut start = offsets[base] as usize;
        for bit in 0..64 {
            let end = offsets[base + bit + 1] as usize;
            word |= ((matcher(&all_bytes[start..end]) != negated) as u64) << bit;
            start = end;
        }
        unsafe { words.push_unchecked(word) };
    }

    if remainder != 0 {
        let base = n_words * 64;
        let mut word = 0u64;
        let mut start = offsets[base] as usize;
        for bit in 0..remainder {
            let end = offsets[base + bit + 1] as usize;
            word |= ((matcher(&all_bytes[start..end]) != negated) as u64) << bit;
            start = end;
        }
        unsafe { words.push_unchecked(word) };
    }

    BitBuffer::new(words.into_byte_buffer().freeze(), n)
}

// ---------------------------------------------------------------------------
// Strategy 2: BitBuffer::collect_bool
// ---------------------------------------------------------------------------

#[inline(never)]
fn scan_collect_bool(offsets: &[u32], all_bytes: &[u8], n: usize, negated: bool) -> BitBuffer {
    let mut start = offsets[0] as usize;
    BitBuffer::collect_bool(n, |i| {
        let end = offsets[i + 1] as usize;
        let result = matcher(&all_bytes[start..end]) != negated;
        start = end;
        result
    })
}

// ---------------------------------------------------------------------------
// Strategy 3: [bool; 64] stack buffer then compress
// ---------------------------------------------------------------------------

#[inline(never)]
fn scan_bool_buf(offsets: &[u32], all_bytes: &[u8], n: usize, negated: bool) -> BitBuffer {
    let n_words = n / 64;
    let remainder = n % 64;
    let mut words: BufferMut<u64> = BufferMut::with_capacity(n.div_ceil(64));

    for chunk in 0..n_words {
        let base = chunk * 64;
        let mut bools = [false; 64];
        let mut start = offsets[base] as usize;
        for bit in 0..64 {
            let end = offsets[base + bit + 1] as usize;
            bools[bit] = matcher(&all_bytes[start..end]) != negated;
            start = end;
        }
        let mut word = 0u64;
        for bit in 0..64 {
            word |= (bools[bit] as u64) << bit;
        }
        unsafe { words.push_unchecked(word) };
    }

    if remainder != 0 {
        let base = n_words * 64;
        let mut bools = [false; 64];
        let mut start = offsets[base] as usize;
        for bit in 0..remainder {
            let end = offsets[base + bit + 1] as usize;
            bools[bit] = matcher(&all_bytes[start..end]) != negated;
            start = end;
        }
        let mut word = 0u64;
        for bit in 0..remainder {
            word |= (bools[bit] as u64) << bit;
        }
        unsafe { words.push_unchecked(word) };
    }

    BitBuffer::new(words.into_byte_buffer().freeze(), n)
}

// ---------------------------------------------------------------------------
// Strategy 4: precomputed bools (pure packing, no matcher at all)
// Isolates *just* the bool→bitbuffer packing cost.
// ---------------------------------------------------------------------------

#[inline(never)]
fn pack_from_slice_manual(results: &[bool], n: usize) -> BitBuffer {
    let n_words = n / 64;
    let remainder = n % 64;
    let mut words: BufferMut<u64> = BufferMut::with_capacity(n.div_ceil(64));

    for chunk in 0..n_words {
        let base = chunk * 64;
        let mut word = 0u64;
        for bit in 0..64 {
            word |= (results[base + bit] as u64) << bit;
        }
        unsafe { words.push_unchecked(word) };
    }

    if remainder != 0 {
        let base = n_words * 64;
        let mut word = 0u64;
        for bit in 0..remainder {
            word |= (results[base + bit] as u64) << bit;
        }
        unsafe { words.push_unchecked(word) };
    }

    BitBuffer::new(words.into_byte_buffer().freeze(), n)
}

#[inline(never)]
fn pack_from_slice_collect_bool(results: &[bool], n: usize) -> BitBuffer {
    BitBuffer::collect_bool(n, |i| results[i])
}

#[inline(never)]
fn pack_from_slice_bool_buf(results: &[bool], n: usize) -> BitBuffer {
    let n_words = n / 64;
    let remainder = n % 64;
    let mut words: BufferMut<u64> = BufferMut::with_capacity(n.div_ceil(64));

    for chunk in 0..n_words {
        let base = chunk * 64;
        let mut bools = [false; 64];
        bools.copy_from_slice(&results[base..base + 64]);
        let mut word = 0u64;
        for bit in 0..64 {
            word |= (bools[bit] as u64) << bit;
        }
        unsafe { words.push_unchecked(word) };
    }

    if remainder != 0 {
        let base = n_words * 64;
        let mut bools = [false; 64];
        bools[..remainder].copy_from_slice(&results[base..base + remainder]);
        let mut word = 0u64;
        for bit in 0..remainder {
            word |= (bools[bit] as u64) << bit;
        }
        unsafe { words.push_unchecked(word) };
    }

    BitBuffer::new(words.into_byte_buffer().freeze(), n)
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

static TEST_DATA: std::sync::LazyLock<TestData> = std::sync::LazyLock::new(TestData::new);

// --- Group 1: with offset reading + trivial matcher (dfa_scan_to_bitbuf shape) ---

#[divan::bench]
fn with_offsets_manual_word(bencher: Bencher) {
    let data = &*TEST_DATA;
    bencher.bench_local(|| scan_manual_word(&data.offsets, &data.bytes, N, false));
}

#[divan::bench]
fn with_offsets_collect_bool(bencher: Bencher) {
    let data = &*TEST_DATA;
    bencher.bench_local(|| scan_collect_bool(&data.offsets, &data.bytes, N, false));
}

#[divan::bench]
fn with_offsets_bool_buf_64(bencher: Bencher) {
    let data = &*TEST_DATA;
    bencher.bench_local(|| scan_bool_buf(&data.offsets, &data.bytes, N, false));
}

// --- Group 2: pure packing from precomputed bools (isolates packing cost) ---

#[divan::bench]
fn pure_pack_manual_word(bencher: Bencher) {
    let data = &*TEST_DATA;
    bencher.bench_local(|| pack_from_slice_manual(&data.results, N));
}

#[divan::bench]
fn pure_pack_collect_bool(bencher: Bencher) {
    let data = &*TEST_DATA;
    bencher.bench_local(|| pack_from_slice_collect_bool(&data.results, N));
}

#[divan::bench]
fn pure_pack_bool_buf_64(bencher: Bencher) {
    let data = &*TEST_DATA;
    bencher.bench_local(|| pack_from_slice_bool_buf(&data.results, N));
}
