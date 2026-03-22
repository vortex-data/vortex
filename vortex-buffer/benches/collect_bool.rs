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

// --- Very large closure: ~1000 instructions of work ---
#[inline(always)]
fn very_expensive(i: usize) -> bool {
    let mut x = i as u64;
    // Repeat the expensive block many times to reach ~1k instructions.
    macro_rules! round {
        ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr) => {
            x = x.wrapping_mul($a);
            x ^= x >> $d;
            x = x.wrapping_add($b);
            x ^= x << $e;
            x = x.wrapping_mul($c);
            x ^= x >> 17;
        };
    }
    // ~12 instructions per round, ~80 rounds ≈ ~960 instructions
    round!(6364136223846793005, 0xdeadbeef, 2862933555777941757, 13, 7);
    round!(5765228890318722077, 0xcafebabe, 1442695040888963407, 15, 11);
    round!(6364136223846793005, 0xfeedface, 2862933555777941757, 11, 5);
    round!(5765228890318722077, 0x12345678, 1442695040888963407, 19, 9);
    round!(6364136223846793005, 0x87654321, 2862933555777941757, 23, 3);
    round!(5765228890318722077, 0xabcdef01, 1442695040888963407, 13, 13);
    round!(6364136223846793005, 0x11111111, 2862933555777941757, 15, 7);
    round!(5765228890318722077, 0x22222222, 1442695040888963407, 11, 11);
    round!(6364136223846793005, 0x33333333, 2862933555777941757, 19, 5);
    round!(5765228890318722077, 0x44444444, 1442695040888963407, 23, 9);
    round!(6364136223846793005, 0x55555555, 2862933555777941757, 13, 3);
    round!(5765228890318722077, 0x66666666, 1442695040888963407, 15, 13);
    round!(6364136223846793005, 0x77777777, 2862933555777941757, 11, 7);
    round!(5765228890318722077, 0x88888888, 1442695040888963407, 19, 11);
    round!(6364136223846793005, 0x99999999, 2862933555777941757, 23, 5);
    round!(5765228890318722077, 0xaaaaaaaa, 1442695040888963407, 13, 9);
    round!(6364136223846793005, 0xbbbbbbbb, 2862933555777941757, 15, 3);
    round!(5765228890318722077, 0xcccccccc, 1442695040888963407, 11, 13);
    round!(6364136223846793005, 0xdddddddd, 2862933555777941757, 19, 7);
    round!(5765228890318722077, 0xeeeeeeee, 1442695040888963407, 23, 11);
    // 20 more rounds
    round!(6364136223846793005, 0x01010101, 2862933555777941757, 13, 5);
    round!(5765228890318722077, 0x02020202, 1442695040888963407, 15, 9);
    round!(6364136223846793005, 0x03030303, 2862933555777941757, 11, 3);
    round!(5765228890318722077, 0x04040404, 1442695040888963407, 19, 13);
    round!(6364136223846793005, 0x05050505, 2862933555777941757, 23, 7);
    round!(5765228890318722077, 0x06060606, 1442695040888963407, 13, 11);
    round!(6364136223846793005, 0x07070707, 2862933555777941757, 15, 5);
    round!(5765228890318722077, 0x08080808, 1442695040888963407, 11, 9);
    round!(6364136223846793005, 0x09090909, 2862933555777941757, 19, 3);
    round!(5765228890318722077, 0x0a0a0a0a, 1442695040888963407, 23, 13);
    round!(6364136223846793005, 0x0b0b0b0b, 2862933555777941757, 13, 7);
    round!(5765228890318722077, 0x0c0c0c0c, 1442695040888963407, 15, 11);
    round!(6364136223846793005, 0x0d0d0d0d, 2862933555777941757, 11, 5);
    round!(5765228890318722077, 0x0e0e0e0e, 1442695040888963407, 19, 9);
    round!(6364136223846793005, 0x0f0f0f0f, 2862933555777941757, 23, 3);
    round!(5765228890318722077, 0x10101010, 1442695040888963407, 13, 13);
    round!(6364136223846793005, 0xa1a1a1a1, 2862933555777941757, 15, 7);
    round!(5765228890318722077, 0xb2b2b2b2, 1442695040888963407, 11, 11);
    round!(6364136223846793005, 0xc3c3c3c3, 2862933555777941757, 19, 5);
    round!(5765228890318722077, 0xd4d4d4d4, 1442695040888963407, 23, 9);
    // 20 more rounds
    round!(6364136223846793005, 0xe5e5e5e5, 2862933555777941757, 13, 3);
    round!(5765228890318722077, 0xf6f6f6f6, 1442695040888963407, 15, 13);
    round!(6364136223846793005, 0xa7a7a7a7, 2862933555777941757, 11, 7);
    round!(5765228890318722077, 0xb8b8b8b8, 1442695040888963407, 19, 11);
    round!(6364136223846793005, 0xc9c9c9c9, 2862933555777941757, 23, 5);
    round!(5765228890318722077, 0xdadadada, 1442695040888963407, 13, 9);
    round!(6364136223846793005, 0xebebebeb, 2862933555777941757, 15, 3);
    round!(5765228890318722077, 0xfcfcfcfc, 1442695040888963407, 11, 13);
    round!(6364136223846793005, 0x1d1d1d1d, 2862933555777941757, 19, 7);
    round!(5765228890318722077, 0x2e2e2e2e, 1442695040888963407, 23, 11);
    round!(6364136223846793005, 0x3f3f3f3f, 2862933555777941757, 13, 5);
    round!(5765228890318722077, 0x40404040, 1442695040888963407, 15, 9);
    round!(6364136223846793005, 0x51515151, 2862933555777941757, 11, 3);
    round!(5765228890318722077, 0x62626262, 1442695040888963407, 19, 13);
    round!(6364136223846793005, 0x73737373, 2862933555777941757, 23, 7);
    round!(5765228890318722077, 0x84848484, 1442695040888963407, 13, 11);
    round!(6364136223846793005, 0x95959595, 2862933555777941757, 15, 5);
    round!(5765228890318722077, 0xa6a6a6a6, 1442695040888963407, 11, 9);
    round!(6364136223846793005, 0xb7b7b7b7, 2862933555777941757, 19, 3);
    round!(5765228890318722077, 0xc8c8c8c8, 1442695040888963407, 23, 13);
    // 20 more rounds to get closer to 80 total
    round!(6364136223846793005, 0xd9d9d9d9, 2862933555777941757, 13, 7);
    round!(5765228890318722077, 0xeaeaeaea, 1442695040888963407, 15, 11);
    round!(6364136223846793005, 0xfbfbfbfb, 2862933555777941757, 11, 5);
    round!(5765228890318722077, 0x1c1c1c1c, 1442695040888963407, 19, 9);
    round!(6364136223846793005, 0x2d2d2d2d, 2862933555777941757, 23, 3);
    round!(5765228890318722077, 0x3e3e3e3e, 1442695040888963407, 13, 13);
    round!(6364136223846793005, 0x4f4f4f4f, 2862933555777941757, 15, 7);
    round!(5765228890318722077, 0x50505050, 1442695040888963407, 11, 11);
    round!(6364136223846793005, 0x61616161, 2862933555777941757, 19, 5);
    round!(5765228890318722077, 0x72727272, 1442695040888963407, 23, 9);
    round!(6364136223846793005, 0x83838383, 2862933555777941757, 13, 3);
    round!(5765228890318722077, 0x94949494, 1442695040888963407, 15, 13);
    round!(6364136223846793005, 0xa5a5a5a5, 2862933555777941757, 11, 7);
    round!(5765228890318722077, 0xb6b6b6b6, 1442695040888963407, 19, 11);
    round!(6364136223846793005, 0xc7c7c7c7, 2862933555777941757, 23, 5);
    round!(5765228890318722077, 0xd8d8d8d8, 1442695040888963407, 13, 9);
    round!(6364136223846793005, 0xe9e9e9e9, 2862933555777941757, 15, 3);
    round!(5765228890318722077, 0xfafafafa, 1442695040888963407, 11, 13);
    round!(6364136223846793005, 0x0b0b0b0b, 2862933555777941757, 19, 7);
    round!(5765228890318722077, 0x1c1c1c1c, 1442695040888963407, 23, 11);
    x & 1 == 0
}

// --- Non-inlined function call ---
#[inline(never)]
fn non_inlined(i: usize) -> bool {
    let mut x = i as u64;
    x = x.wrapping_mul(6364136223846793005);
    x ^= x >> 17;
    x = x.wrapping_add(0xdeadbeef);
    x ^= x << 7;
    x & 1 == 0
}

// ============ Benchmarks ============

#[divan::bench(args = SIZES)]
fn bench_cheap(bencher: Bencher, n: usize) {
    bencher.bench(|| BitBufferMut::collect_bool(n, cheap));
}

#[divan::bench(args = SIZES)]
fn bench_with_load(bencher: Bencher, n: usize) {
    let data: Vec<u8> = (0..n).map(|i| (i & 0xFF) as u8).collect();
    bencher.bench(|| BitBufferMut::collect_bool(n, |i| with_load(&data, i)));
}

#[divan::bench(args = SIZES)]
fn bench_expensive(bencher: Bencher, n: usize) {
    bencher.bench(|| BitBufferMut::collect_bool(n, expensive));
}

#[divan::bench(args = SIZES)]
fn bench_very_expensive(bencher: Bencher, n: usize) {
    bencher.bench(|| BitBufferMut::collect_bool(n, very_expensive));
}

#[divan::bench(args = SIZES)]
fn bench_non_inlined(bencher: Bencher, n: usize) {
    bencher.bench(|| BitBufferMut::collect_bool(n, non_inlined));
}
