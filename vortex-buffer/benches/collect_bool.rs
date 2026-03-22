// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used, clippy::cast_possible_truncation)]

use divan::Bencher;
use vortex_buffer::BitBufferMut;

fn main() {
    #[cfg(target_arch = "x86_64")]
    {
        let _ = is_x86_feature_detected!("avx2");
        let _ = is_x86_feature_detected!("avx512f");
    }

    divan::main();
}

const SIZES: &[usize] = &[10_000, 100_000];

// --- Cheap closure: a few ALU instructions ---
#[inline(always)]
fn cheap(i: usize) -> bool {
    (i ^ (i >> 1)) & 1 == 0
}

// --- Medium closure: ALU + a load from a slice ---
#[inline(always)]
fn with_load(data: &[u8], i: usize) -> bool {
    unsafe { *data.get_unchecked(i) > 127 }
}

// --- Expensive closure: ~50 instructions of work ---
#[inline(always)]
fn expensive(i: usize) -> bool {
    let mut x = i as u64;
    x = x.wrapping_mul(6364136223846793005);
    x ^= x >> 17;
    x = x.wrapping_add(0xdeadbeef);
    x ^= x << 7;
    x = x.wrapping_mul(2862933555777941757);
    x ^= x >> 13;
    x = x.wrapping_add(0xcafebabe);
    x ^= x << 11;
    x = x.wrapping_mul(6364136223846793005);
    x ^= x >> 15;
    x = x.wrapping_add(0xfeedface);
    x ^= x << 5;
    x = x.wrapping_mul(2862933555777941757);
    x ^= x >> 19;
    x = x.wrapping_add(0x12345678);
    x ^= x << 9;
    x = x.wrapping_mul(6364136223846793005);
    x ^= x >> 11;
    x = x.wrapping_add(0x87654321);
    x ^= x << 3;
    x = x.wrapping_mul(2862933555777941757);
    x ^= x >> 23;
    x = x.wrapping_add(0xabcdef01);
    x ^= x << 13;
    x & 1 == 0
}

// ============ Old baseline (u64-at-a-time) ============

#[divan::bench(args = SIZES)]
fn collect_bool_u64_cheap(bencher: Bencher, n: usize) {
    bencher.bench(|| BitBufferMut::collect_bool_u64(n, cheap));
}

#[divan::bench(args = SIZES)]
fn collect_bool_u64_with_load(bencher: Bencher, n: usize) {
    let data: Vec<u8> = (0..n).map(|i| (i & 0xFF) as u8).collect();
    bencher.bench(|| BitBufferMut::collect_bool_u64(n, |i| with_load(&data, i)));
}

#[divan::bench(args = SIZES)]
fn collect_bool_u64_expensive(bencher: Bencher, n: usize) {
    bencher.bench(|| BitBufferMut::collect_bool_u64(n, expensive));
}

// ============ New default (u8 byte packing) ============

#[divan::bench(args = SIZES)]
fn collect_bool_cheap(bencher: Bencher, n: usize) {
    bencher.bench(|| BitBufferMut::collect_bool(n, cheap));
}

#[divan::bench(args = SIZES)]
fn collect_bool_with_load(bencher: Bencher, n: usize) {
    let data: Vec<u8> = (0..n).map(|i| (i & 0xFF) as u8).collect();
    bencher.bench(|| BitBufferMut::collect_bool(n, |i| with_load(&data, i)));
}

#[divan::bench(args = SIZES)]
fn collect_bool_expensive(bencher: Bencher, n: usize) {
    bencher.bench(|| BitBufferMut::collect_bool(n, expensive));
}

// ============ SIMD: temp buffer + pack ============

#[divan::bench(args = SIZES)]
fn collect_bool_simd_cheap(bencher: Bencher, n: usize) {
    bencher.bench(|| BitBufferMut::collect_bool_simd(n, cheap));
}

#[divan::bench(args = SIZES)]
fn collect_bool_simd_with_load(bencher: Bencher, n: usize) {
    let data: Vec<u8> = (0..n).map(|i| (i & 0xFF) as u8).collect();
    bencher.bench(|| BitBufferMut::collect_bool_simd(n, |i| with_load(&data, i)));
}

#[divan::bench(args = SIZES)]
fn collect_bool_simd_expensive(bencher: Bencher, n: usize) {
    bencher.bench(|| BitBufferMut::collect_bool_simd(n, expensive));
}
