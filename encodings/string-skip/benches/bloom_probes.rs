// SPDX-License-Identifier: Apache-2.0
//! Microbenchmarks for the hot paths.
//!
//! These measure per-op cost in nanoseconds, so they show whether our
//! hand-rolled bloom is in the ballpark of `fastbloom` (which we
//! decided not to depend on).

use criterion::BatchSize;
use criterion::Criterion;
use criterion::black_box;
use criterion::criterion_group;
use criterion::criterion_main;
use string_skip::Bloom;
use string_skip::DictPresence;
use string_skip::UbiquitousBigrams;
use string_skip::hash::splitmix32;

fn bench_bloom_insert(c: &mut Criterion) {
    let mut g = c.benchmark_group("bloom_insert");
    for &num_bits in &[1024usize, 16_384, 131_072] {
        g.bench_function(format!("bits={num_bits}_k=3"), |b| {
            b.iter_batched_ref(
                || (Bloom::new(num_bits, 3), 0u32),
                |(bloom, i)| {
                    let h1 = splitmix32(*i);
                    let h2 = splitmix32(*i ^ 0x27d4_eb2f);
                    bloom.insert(black_box(h1), black_box(h2));
                    *i = i.wrapping_add(1);
                },
                BatchSize::SmallInput,
            );
        });
    }
    g.finish();
}

fn bench_bloom_contains(c: &mut Criterion) {
    let mut g = c.benchmark_group("bloom_contains");
    for &num_bits in &[1024usize, 16_384, 131_072] {
        // Pre-fill the bloom to roughly 50% load (theoretical optimal).
        let mut bloom = Bloom::new(num_bits, 3);
        let fill = (num_bits / 3) as u32;
        for i in 0..fill {
            let h1 = splitmix32(i);
            let h2 = splitmix32(i ^ 0x27d4_eb2f);
            bloom.insert(h1, h2);
        }
        g.bench_function(format!("bits={num_bits}_k=3"), |b| {
            let mut i = fill;
            b.iter(|| {
                let h1 = splitmix32(i);
                let h2 = splitmix32(i ^ 0x27d4_eb2f);
                let r = bloom.contains(black_box(h1), black_box(h2));
                i = i.wrapping_add(1);
                r
            });
        });
    }
    g.finish();
}

fn bench_bloom_contains_k(c: &mut Criterion) {
    let mut g = c.benchmark_group("bloom_contains_k_variable");
    let mut bloom = Bloom::new(16_384, 3);
    for i in 0..5000u32 {
        let h1 = splitmix32(i);
        let h2 = splitmix32(i ^ 0x27d4_eb2f);
        bloom.insert(h1, h2);
    }
    for &k in &[1u32, 2, 3, 4, 5] {
        g.bench_function(format!("k={k}"), |b| {
            let mut i = 5000u32;
            b.iter(|| {
                let h1 = splitmix32(i);
                let h2 = splitmix32(i ^ 0x27d4_eb2f);
                let r = bloom.contains_k(black_box(h1), black_box(h2), k);
                i = i.wrapping_add(1);
                r
            });
        });
    }
    g.finish();
}

fn bench_dict_presence_lookup(c: &mut Criterion) {
    let codes: Vec<u16> = (0..4096u16).chain(0..4096u16).collect();
    let presence = DictPresence::build(&codes, 4096);
    c.bench_function("dict_presence_is_set", |b| {
        let mut i = 0u16;
        b.iter(|| {
            let r = presence.is_set(black_box(i as usize % 4096));
            i = i.wrapping_add(1);
            r
        });
    });
}

fn bench_ubiq_contains(c: &mut Criterion) {
    // Simulate 10K chunks where bigrams (0..1000, 0..1000) appear often.
    let n_chunks = 100usize;
    let mut codes = Vec::new();
    let mut offsets = vec![0u32];
    for _ in 0..n_chunks {
        for j in 0..50u16 {
            codes.push(j);
            codes.push(j + 1);
        }
        offsets.push(codes.len() as u32);
    }
    let ubiq = UbiquitousBigrams::build(&codes, &offsets, 1, 50);
    c.bench_function("ubiq_contains", |b| {
        let mut i = 0u16;
        b.iter(|| {
            let r = ubiq.contains(black_box(i), black_box(i + 1));
            i = i.wrapping_add(1);
            r
        });
    });
}

criterion_group!(
    benches,
    bench_bloom_insert,
    bench_bloom_contains,
    bench_bloom_contains_k,
    bench_dict_presence_lookup,
    bench_ubiq_contains,
);
criterion_main!(benches);
