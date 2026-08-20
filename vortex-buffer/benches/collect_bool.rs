// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks for `collect_bool` bit packing.
//!
//! Three groups, each with a scalar baseline that replicates the previous bit-at-a-time
//! implementation:
//!
//! - `pack_words_*`: the portable SWAR kernel vs the scalar loop, word by word. The SIMD
//!   kernels are covered by `words_gather_*` instead, where they can inline.
//! - `words_gather_*`: each multiversioned word loop with a bounds-check-free bool gather,
//!   measuring the fused fill + pack pipeline per SIMD level.
//! - `collect_bool_*` / `from_bool_slice`: the public entry points end to end, with a
//!   boolean-gather predicate and a `u32` comparison predicate.
//!
//! `words_gather_dispatch` and `words_gather_scalar` carry `#[cpu_features]`, so they are
//! measured on every walltime CPU-feature leg rather than in simulation. Both are written
//! once and compiled differently per leg: the shipped entry point picks its pack kernel
//! through `cfg(target_feature)`, and how well the scalar loop auto-vectorizes depends on
//! the build. Comparing them across legs is the point.
//!
//! The hand-written per-kernel benchmarks are not tagged. Each one needs an instruction set
//! extension the other legs do not build for, so they stay out of CodSpeed entirely and
//! remain local A/B tools, as do the historical bit-at-a-time baselines.
//!
//! A plain `cargo bench` ignores all of it and runs everything on the host.

use divan::Bencher;
use vortex_buffer::BitBuffer;
use vortex_buffer::collect_bool_word_scalar;
#[cfg(not(codspeed))]
use vortex_buffer::pack_bool_word_swar;

fn main() {
    // Pre-warm CPUID feature detection so the one-time probe cost is never
    // included in any benchmark iteration.
    #[cfg(target_arch = "x86_64")]
    {
        let _ = is_x86_feature_detected!("avx2");
        let _ = is_x86_feature_detected!("avx512f");
        let _ = is_x86_feature_detected!("avx512bw");
    }

    divan::main();
}

const INPUT_SIZE: &[usize] = &[1024, 65_536];

/// Deterministic pseudo-random words (LCG), the source for all benchmark inputs.
fn make_words(len: usize) -> impl Iterator<Item = u64> {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    (0..len).map(move |_| {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state
    })
}

fn make_bools(len: usize) -> Vec<bool> {
    make_words(len).map(|w| (w >> 33) & 1 == 1).collect()
}

fn make_u32s(len: usize) -> Vec<u32> {
    make_words(len).map(|w| (w >> 32) as u32).collect()
}

/// Benchmark a per-word packing kernel over `len` pre-materialized bools.
#[cfg(not(codspeed))]
fn bench_pack_words(bencher: Bencher, len: usize, pack: impl Fn(&[bool; 64]) -> u64 + Sync) {
    let bools = make_bools(len);
    bencher
        .with_inputs(|| vec![0u64; len / 64])
        .bench_refs(|out| {
            let (chunks, _) = bools.as_chunks::<64>();
            for (word, chunk) in out.iter_mut().zip(chunks) {
                *word = pack(chunk);
            }
        });
}

#[cfg(not(codspeed))]
#[divan::bench(args = INPUT_SIZE)]
fn pack_words_scalar(bencher: Bencher, len: usize) {
    bench_pack_words(bencher, len, |chunk| {
        collect_bool_word_scalar(64, |i| chunk[i])
    });
}

#[cfg(not(codspeed))]
#[divan::bench(args = INPUT_SIZE)]
fn pack_words_swar(bencher: Bencher, len: usize) {
    bench_pack_words(bencher, len, pack_bool_word_swar);
}

/// Benchmark a full multiversioned word loop with a bounds-check-free bool gather, measuring
/// the packing pipeline itself (fill + pack fully inlined) rather than predicate evaluation.
fn bench_words_gather(
    bencher: Bencher,
    len: usize,
    collect: impl Fn(&mut [u64], usize, &[bool]) + Sync,
) {
    let bools = make_bools(len);
    bencher
        .with_inputs(|| vec![0u64; len.div_ceil(64)])
        .bench_refs(|words| collect(words, len, &bools));
}

#[vortex_bench_support::cpu_features]
#[divan::bench(args = INPUT_SIZE)]
fn words_gather_dispatch(bencher: Bencher, len: usize) {
    bench_words_gather(bencher, len, |words, len, bools| {
        // SAFETY: `collect_bool_words` invokes the predicate with indices `0..len` only.
        vortex_buffer::collect_bool_words(words, len, |i| unsafe { *bools.get_unchecked(i) })
    });
}

#[vortex_bench_support::cpu_features]
#[divan::bench(args = INPUT_SIZE)]
fn words_gather_scalar(bencher: Bencher, len: usize) {
    bench_words_gather(bencher, len, |words, len, bools| {
        // SAFETY: `collect_bool_words_old` invokes the predicate with indices `0..len` only.
        collect_bool_words_old(words, len, |i| unsafe { *bools.get_unchecked(i) })
    });
}

#[cfg(target_arch = "x86_64")]
#[cfg(not(codspeed))]
#[divan::bench(args = INPUT_SIZE)]
fn words_gather_sse2(bencher: Bencher, len: usize) {
    bench_words_gather(bencher, len, |words, len, bools| {
        // SAFETY: SSE2 is part of the x86-64 baseline; indices passed are `0..len`.
        unsafe { vortex_buffer::collect_bool_words_sse2(words, len, |i| *bools.get_unchecked(i)) }
    });
}

#[cfg(target_arch = "x86_64")]
#[cfg(not(codspeed))]
#[divan::bench(args = INPUT_SIZE)]
fn words_gather_avx2(bencher: Bencher, len: usize) {
    if !is_x86_feature_detected!("avx2") {
        return;
    }
    bench_words_gather(bencher, len, |words, len, bools| {
        // SAFETY: runtime detection guarantees AVX2; indices passed are `0..len`.
        unsafe { vortex_buffer::collect_bool_words_avx2(words, len, |i| *bools.get_unchecked(i)) }
    });
}

#[cfg(target_arch = "x86_64")]
#[cfg(not(codspeed))]
#[divan::bench(args = INPUT_SIZE)]
fn words_gather_avx512(bencher: Bencher, len: usize) {
    if !(is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw")) {
        return;
    }
    bench_words_gather(bencher, len, |words, len, bools| {
        // SAFETY: runtime detection guarantees AVX-512F/BW; indices passed are `0..len`.
        unsafe { vortex_buffer::collect_bool_words_avx512(words, len, |i| *bools.get_unchecked(i)) }
    });
}

#[cfg(target_arch = "aarch64")]
#[cfg(not(codspeed))]
#[divan::bench(args = INPUT_SIZE)]
fn words_gather_neon(bencher: Bencher, len: usize) {
    bench_words_gather(bencher, len, |words, len, bools| {
        // SAFETY: NEON is part of the aarch64 baseline; indices passed are `0..len`.
        unsafe { vortex_buffer::collect_bool_words_neon(words, len, |i| *bools.get_unchecked(i)) }
    });
}

/// Faithful copy of the previous scalar-only `collect_bool_words` word loop, used as the
/// baseline for the end-to-end comparison.
fn collect_bool_words_old(words: &mut [u64], len: usize, mut f: impl FnMut(usize) -> bool) {
    let full = len / 64;
    let remainder = len % 64;
    for word_idx in 0..full {
        let offset = word_idx * 64;
        words[word_idx] = collect_bool_word_scalar(64, |bit_idx| f(offset + bit_idx));
    }
    if remainder != 0 {
        let offset = full * 64;
        words[full] = collect_bool_word_scalar(remainder, |bit_idx| f(offset + bit_idx));
    }
}

#[cfg(not(codspeed))]
#[divan::bench(args = INPUT_SIZE)]
fn from_bool_slice_old_scalar(bencher: Bencher, len: usize) {
    let bools = make_bools(len);
    bencher
        .with_inputs(|| vec![0u64; len.div_ceil(64)])
        .bench_refs(|words| collect_bool_words_old(words, len, |i| bools[i]));
}

#[divan::bench(args = INPUT_SIZE)]
fn from_bool_slice(bencher: Bencher, len: usize) {
    let bools = make_bools(len);
    bencher.bench(|| vortex_buffer::BitBufferMut::from(bools.as_slice()));
}

#[cfg(not(codspeed))]
#[divan::bench(args = INPUT_SIZE)]
fn collect_bool_u32_gt_old_scalar(bencher: Bencher, len: usize) {
    let values = make_u32s(len);
    bencher
        .with_inputs(|| vec![0u64; len.div_ceil(64)])
        .bench_refs(|words| collect_bool_words_old(words, len, |i| values[i] > u32::MAX / 2));
}

#[divan::bench(args = INPUT_SIZE)]
fn collect_bool_u32_gt(bencher: Bencher, len: usize) {
    let values = make_u32s(len);
    bencher.bench(|| BitBuffer::collect_bool(len, |i| values[i] > u32::MAX / 2));
}
