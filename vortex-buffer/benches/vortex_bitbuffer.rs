// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used)]

use std::iter::Iterator;

use arrow_buffer::BooleanBuffer;
use arrow_buffer::BooleanBufferBuilder;
use divan::Bencher;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_buffer::ScalarBitIndexIterator;
use vortex_buffer::collect_set_indices;
use vortex_buffer::collect_set_indices_scalar;
use vortex_buffer::collect_set_indices_with_count;

fn main() {
    // Pre-warm CPUID feature detection so the one-time probe cost is never
    // included in any benchmark iteration.
    #[cfg(target_arch = "x86_64")]
    {
        let _ = is_x86_feature_detected!("avx2");
        let _ = is_x86_feature_detected!("avx512f");
        let _ = is_x86_feature_detected!("avx512vpopcntdq");
        let _ = is_x86_feature_detected!("bmi2");
    }

    divan::main();
}

/// Wraps an arrow buffer so Divan can provide a nice name
pub struct Arrow<T>(T);

impl FromIterator<bool> for Arrow<BooleanBuffer> {
    fn from_iter<I: IntoIterator<Item = bool>>(iter: I) -> Self {
        Self(BooleanBuffer::from_iter(iter))
    }
}

const INPUT_SIZE: &[usize] = &[128, 1024, 2048, 16_384, 65_536];

#[inline]
fn true_count_pattern(i: usize) -> bool {
    (i.is_multiple_of(3)) ^ (i.is_multiple_of(11))
}

#[cfg(not(codspeed))]
#[divan::bench(args = INPUT_SIZE)]
fn from_iter_arrow(n: usize) {
    Arrow::<BooleanBuffer>::from_iter((0..n).map(|i| i % 2 == 0));
}

#[divan::bench(args = INPUT_SIZE)]
fn from_iter_bit_buffer(n: usize) {
    BitBuffer::from_iter((0..n).map(|i| i % 2 == 0));
}

#[divan::bench(args = INPUT_SIZE)]
fn append_vortex_buffer(bencher: Bencher, length: usize) {
    bencher
        .with_inputs(|| (BitBufferMut::with_capacity(length), length))
        .bench_refs(|(buffer, length)| {
            for idx in 0..*length {
                buffer.append(idx % 2 == 0);
            }
        });
}

#[divan::bench(args = INPUT_SIZE)]
fn append_arrow_buffer(bencher: Bencher, length: usize) {
    bencher
        .with_inputs(|| (Arrow(BooleanBufferBuilder::new(length)), length))
        .bench_refs(|(buffer, length)| {
            for idx in 0..*length {
                buffer.0.append(idx % 2 == 0);
            }
        });
}

#[divan::bench(args = INPUT_SIZE)]
fn append_n_vortex_buffer(bencher: Bencher, length: usize) {
    bencher
        .with_inputs(|| (BitBufferMut::with_capacity(length), length, true))
        .bench_refs(|(buffer, length, boolean)| {
            for _ in 0..100 {
                buffer.append_n(*boolean, *length / 100);
            }
        });
}

#[divan::bench(args = INPUT_SIZE)]
fn append_n_arrow_buffer(bencher: Bencher, length: usize) {
    bencher
        .with_inputs(|| (Arrow(BooleanBufferBuilder::new(length)), length, true))
        .bench_refs(|(buffer, length, boolean)| {
            for _ in 0..100 {
                buffer.0.append_n(*length / 100, *boolean);
            }
        });
}

#[divan::bench(args = INPUT_SIZE)]
fn append_buffer_vortex_buffer(bencher: Bencher, length: usize) {
    bencher
        .with_inputs(|| {
            let source = BitBuffer::from_iter((0..length / 100).map(|i| i % 2 == 0));
            let dest = BitBufferMut::with_capacity(length);
            (source, dest)
        })
        .bench_refs(|(source, dest)| {
            for _ in 0..100 {
                dest.append_buffer(source);
            }
        });
}

#[divan::bench(args = INPUT_SIZE)]
fn append_buffer_arrow_buffer(bencher: Bencher, length: usize) {
    bencher
        .with_inputs(|| {
            let source = Arrow(BooleanBuffer::from_iter(
                (0..length / 100).map(|i| i % 2 == 0),
            ));
            let dest = Arrow(BooleanBufferBuilder::new(length));
            (source, dest)
        })
        .bench_refs(|(source, dest)| {
            for _ in 0..100 {
                for value in source.0.iter() {
                    dest.0.append(value);
                }
            }
        });
}

#[divan::bench(args = INPUT_SIZE)]
fn value_vortex_buffer(bencher: Bencher, length: usize) {
    let buffer = BitBuffer::from_iter((0..length).map(|i| i % 2 == 0));
    bencher.with_inputs(|| &buffer).bench_refs(|buffer| {
        for idx in 0..length {
            divan::black_box(buffer.value(idx));
        }
    });
}

#[divan::bench(args = INPUT_SIZE)]
fn value_arrow_buffer(bencher: Bencher, length: usize) {
    let buffer = Arrow(BooleanBuffer::from_iter((0..length).map(|i| i % 2 == 0)));
    bencher.with_inputs(|| &buffer).bench_refs(|buffer| {
        for idx in 0..length {
            divan::black_box(buffer.0.value(idx));
        }
    });
}

#[divan::bench(args = INPUT_SIZE)]
fn slice_vortex_buffer(bencher: Bencher, length: usize) {
    let buffer = BitBuffer::from_iter((0..length).map(|i| i % 2 == 0));
    bencher
        .with_inputs(|| (&buffer, length / 2))
        .bench_refs(|(buffer, mid)| {
            let mid = *mid;
            buffer.slice(mid / 2..mid + mid / 2)
        });
}

#[cfg(not(codspeed))]
#[divan::bench(args = INPUT_SIZE)]
fn slice_arrow_buffer(bencher: Bencher, length: usize) {
    let buffer = Arrow(BooleanBuffer::from_iter((0..length).map(|i| i % 2 == 0)));
    bencher
        .with_inputs(|| (&buffer, length / 2))
        .bench_refs(|(buffer, mid)| {
            let mid = *mid;
            buffer.0.slice(mid / 2, mid / 2)
        });
}

#[divan::bench(args = INPUT_SIZE)]
fn true_count_vortex_buffer(bencher: Bencher, length: usize) {
    let buffer = BitBuffer::from_iter((0..length).map(true_count_pattern));

    bencher
        .with_inputs(|| &buffer)
        .bench_refs(|buffer| buffer.true_count())
}

#[divan::bench(args = INPUT_SIZE)]
fn true_count_arrow_buffer(bencher: Bencher, length: usize) {
    let buffer = Arrow(BooleanBuffer::from_iter(
        (0..length).map(true_count_pattern),
    ));

    buffer.0.count_set_bits();

    bencher
        .with_inputs(|| &buffer)
        .bench_refs(|buffer| buffer.0.count_set_bits());
}

#[divan::bench(args = INPUT_SIZE)]
fn bitwise_and_vortex_buffer(bencher: Bencher, length: usize) {
    let a = BitBuffer::from_iter((0..length).map(|i| i % 2 == 0));
    let b = BitBuffer::from_iter((0..length).map(|i| i % 3 == 0));
    bencher
        .with_inputs(|| (&a, &b))
        .bench_values(|(a, b)| a & b);
}

#[divan::bench(args = INPUT_SIZE)]
fn bitwise_and_arrow_buffer(bencher: Bencher, length: usize) {
    let a = Arrow(BooleanBuffer::from_iter((0..length).map(|i| i % 2 == 0)));
    let b = Arrow(BooleanBuffer::from_iter((0..length).map(|i| i % 3 == 0)));
    bencher
        .with_inputs(|| (&a, &b))
        .bench_refs(|(a, b)| &a.0 & &b.0);
}

#[divan::bench(args = INPUT_SIZE)]
fn bitwise_or_vortex_buffer(bencher: Bencher, length: usize) {
    let a = BitBuffer::from_iter((0..length).map(|i| i % 2 == 0));
    let b = BitBuffer::from_iter((0..length).map(|i| i % 3 == 0));
    bencher
        .with_inputs(|| (&a, &b))
        .bench_values(|(a, b)| a | b);
}

#[divan::bench(args = INPUT_SIZE)]
fn bitwise_or_arrow_buffer(bencher: Bencher, length: usize) {
    let a = Arrow(BooleanBuffer::from_iter((0..length).map(|i| i % 2 == 0)));
    let b = Arrow(BooleanBuffer::from_iter((0..length).map(|i| i % 3 == 0)));
    bencher
        .with_inputs(|| (&a, &b))
        .bench_refs(|(a, b)| &a.0 | &b.0);
}

#[divan::bench(args = INPUT_SIZE)]
fn bitwise_not_vortex_buffer(bencher: Bencher, length: usize) {
    bencher
        .with_inputs(|| BitBuffer::from_iter((0..length).map(|i| i % 2 == 0)))
        .bench_values(|buffer| !&buffer);
}

#[divan::bench(args = INPUT_SIZE)]
fn bitwise_not_vortex_buffer_mut(bencher: Bencher, length: usize) {
    bencher
        .with_inputs(|| BitBufferMut::from_iter((0..length).map(|i| i % 2 == 0)))
        .bench_values(|buffer| !buffer);
}

#[divan::bench(args = INPUT_SIZE)]
fn bitwise_not_arrow_buffer(bencher: Bencher, length: usize) {
    bencher
        .with_inputs(|| Arrow(BooleanBuffer::from_iter((0..length).map(|i| i % 2 == 0))))
        .bench_values(|buffer| !&buffer.0);
}

#[divan::bench(args = INPUT_SIZE)]
fn iter_vortex_buffer(bencher: Bencher, length: usize) {
    let buffer = BitBuffer::from_iter((0..length).map(|i| i % 2 == 0));
    bencher.with_inputs(|| &buffer).bench_refs(|buffer| {
        for value in buffer.iter() {
            divan::black_box(value);
        }
    });
}

#[divan::bench(args = INPUT_SIZE)]
fn iter_arrow_buffer(bencher: Bencher, length: usize) {
    let buffer = Arrow(BooleanBuffer::from_iter((0..length).map(|i| i % 2 == 0)));
    bencher.with_inputs(|| &buffer).bench_refs(|buffer| {
        for value in buffer.0.iter() {
            divan::black_box(value);
        }
    });
}

#[divan::bench(args = INPUT_SIZE)]
fn set_indices_vortex_buffer(bencher: Bencher, length: usize) {
    let buffer = BitBuffer::from_iter((0..length).map(|i| i % 2 == 0));
    bencher.with_inputs(|| &buffer).bench_refs(|buffer| {
        for idx in buffer.set_indices() {
            divan::black_box(idx);
        }
    });
}

#[divan::bench(args = INPUT_SIZE)]
fn set_indices_arrow_buffer(bencher: Bencher, length: usize) {
    let buffer = Arrow(BooleanBuffer::from_iter((0..length).map(|i| i % 2 == 0)));
    bencher.with_inputs(|| &buffer).bench_refs(|buffer| {
        for idx in buffer.0.set_indices() {
            divan::black_box(idx);
        }
    });
}

#[divan::bench(args = INPUT_SIZE)]
fn set_indices_scalar_optimized(bencher: Bencher, length: usize) {
    let buffer = BitBuffer::from_iter((0..length).map(|i| i % 2 == 0));
    bencher.with_inputs(|| &buffer).bench_refs(|buffer| {
        for idx in
            ScalarBitIndexIterator::new(buffer.inner().as_slice(), buffer.offset(), buffer.len())
        {
            divan::black_box(idx);
        }
    });
}

#[divan::bench(args = INPUT_SIZE)]
fn collect_set_indices_scalar_bench(bencher: Bencher, length: usize) {
    let buffer = BitBuffer::from_iter((0..length).map(|i| i % 2 == 0));
    bencher.with_inputs(|| &buffer).bench_refs(|buffer| {
        divan::black_box(collect_set_indices_scalar(
            buffer.inner().as_slice(),
            buffer.offset(),
            buffer.len(),
        ));
    });
}

#[divan::bench(args = INPUT_SIZE)]
fn collect_set_indices_simd_bench(bencher: Bencher, length: usize) {
    let buffer = BitBuffer::from_iter((0..length).map(|i| i % 2 == 0));
    bencher.with_inputs(|| &buffer).bench_refs(|buffer| {
        divan::black_box(collect_set_indices(
            buffer.inner().as_slice(),
            buffer.offset(),
            buffer.len(),
        ));
    });
}

// ---------------------------------------------------------------------------
// Density-varied benchmarks: 1M bits at different set-bit densities
// and distributions.
//
// Distributions tested:
//   - "uniform":  every Nth bit (perfectly regular)
//   - "clustered": set bits arrive in bursts/clusters
//   - "random":   pseudo-random (deterministic hash)
// ---------------------------------------------------------------------------

const LARGE_N: usize = 1_000_000;

/// Simple deterministic hash for pseudo-random patterns.
#[inline]
fn splitmix(i: usize) -> u64 {
    let mut z = (i as u64).wrapping_add(0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

// --- Buffer constructors ---

fn make_uniform(density_pct: usize) -> BitBuffer {
    let period = 100 / density_pct;
    BitBuffer::from_iter((0..LARGE_N).map(|i| i % period == 0))
}

fn make_clustered(density_pct: usize) -> BitBuffer {
    // Clusters of 8 set bits, then gaps.
    // Cluster spacing chosen to achieve target density.
    let cluster_size = 8usize;
    let total_per_group = (cluster_size * 100) / density_pct;
    BitBuffer::from_iter((0..LARGE_N).map(|i| (i % total_per_group) < cluster_size))
}

fn make_random(density_pct: usize) -> BitBuffer {
    // Pseudo-random: bit is set if splitmix(i) mod 100 < density_pct
    BitBuffer::from_iter((0..LARGE_N).map(|i| (splitmix(i) % 100) < density_pct as u64))
}

fn make_uniform_arrow(density_pct: usize) -> Arrow<BooleanBuffer> {
    let period = 100 / density_pct;
    Arrow(BooleanBuffer::from_iter(
        (0..LARGE_N).map(|i| i % period == 0),
    ))
}

fn make_random_arrow(density_pct: usize) -> Arrow<BooleanBuffer> {
    Arrow(BooleanBuffer::from_iter(
        (0..LARGE_N).map(|i| (splitmix(i) % 100) < density_pct as u64),
    ))
}

// =========================================================================
// Macro to generate all benchmark variants for a given density + distribution
// =========================================================================
macro_rules! bench_density {
    ($density:literal, $dist:ident, $make_fn:ident, $make_arrow_fn:ident) => {
        ::paste::paste! {
            // Arrow: iterate set_indices (no alloc)
            #[divan::bench]
            fn [< d $density pct_ $dist _arrow >](bencher: Bencher) {
                let buffer = $make_arrow_fn($density);
                bencher.with_inputs(|| &buffer).bench_refs(|buffer| {
                    for idx in buffer.0.set_indices() {
                        divan::black_box(idx);
                    }
                });
            }

            // Arrow: collect into Vec<usize> (allocates)
            #[divan::bench]
            fn [< d $density pct_ $dist _arrow_collect >](bencher: Bencher) {
                let buffer = $make_arrow_fn($density);
                bencher.with_inputs(|| &buffer).bench_refs(|buffer| {
                    divan::black_box(buffer.0.set_indices().collect::<Vec<usize>>());
                });
            }

            // Vortex scalar iterator (no alloc)
            #[divan::bench]
            fn [< d $density pct_ $dist _vortex >](bencher: Bencher) {
                let buffer = $make_fn($density);
                bencher.with_inputs(|| &buffer).bench_refs(|buffer| {
                    for idx in buffer.set_indices() {
                        divan::black_box(idx);
                    }
                });
            }

            // Vortex SIMD bulk collect with pre-known count (allocates)
            #[divan::bench]
            fn [< d $density pct_ $dist _collect_precount >](bencher: Bencher) {
                let buffer = $make_fn($density);
                let true_count = buffer.true_count();
                bencher.with_inputs(|| &buffer).bench_refs(|buffer| {
                    divan::black_box(collect_set_indices_with_count(
                        buffer.inner().as_slice(),
                        buffer.offset(),
                        buffer.len(),
                        Some(true_count),
                    ));
                });
            }
        }
    };
}

// 0.01% density (1 in 10,000 — almost all 256-bit groups are zero)
fn make_very_sparse(period: usize) -> BitBuffer {
    BitBuffer::from_iter((0..LARGE_N).map(|i| i % period == 0))
}

fn make_very_sparse_arrow(period: usize) -> Arrow<BooleanBuffer> {
    Arrow(BooleanBuffer::from_iter(
        (0..LARGE_N).map(|i| i % period == 0),
    ))
}

fn make_very_sparse_random() -> BitBuffer {
    // ~0.01%: 1 in 10,000
    BitBuffer::from_iter((0..LARGE_N).map(|i| splitmix(i).is_multiple_of(10_000)))
}

fn make_very_sparse_random_arrow() -> Arrow<BooleanBuffer> {
    Arrow(BooleanBuffer::from_iter(
        (0..LARGE_N).map(|i| splitmix(i).is_multiple_of(10_000)),
    ))
}

#[divan::bench]
fn d001pct_uniform_arrow(bencher: Bencher) {
    let buffer = make_very_sparse_arrow(10_000);
    bencher.with_inputs(|| &buffer).bench_refs(|buffer| {
        for idx in buffer.0.set_indices() {
            divan::black_box(idx);
        }
    });
}

#[divan::bench]
fn d001pct_uniform_arrow_collect(bencher: Bencher) {
    let buffer = make_very_sparse_arrow(10_000);
    bencher.with_inputs(|| &buffer).bench_refs(|buffer| {
        divan::black_box(buffer.0.set_indices().collect::<Vec<usize>>());
    });
}

#[divan::bench]
fn d001pct_uniform_vortex(bencher: Bencher) {
    let buffer = make_very_sparse(10_000);
    bencher.with_inputs(|| &buffer).bench_refs(|buffer| {
        for idx in buffer.set_indices() {
            divan::black_box(idx);
        }
    });
}

#[divan::bench]
fn d001pct_uniform_collect_precount(bencher: Bencher) {
    let buffer = make_very_sparse(10_000);
    let true_count = buffer.true_count();
    bencher.with_inputs(|| &buffer).bench_refs(|buffer| {
        divan::black_box(collect_set_indices_with_count(
            buffer.inner().as_slice(),
            buffer.offset(),
            buffer.len(),
            Some(true_count),
        ));
    });
}

#[divan::bench]
fn d001pct_random_arrow(bencher: Bencher) {
    let buffer = make_very_sparse_random_arrow();
    bencher.with_inputs(|| &buffer).bench_refs(|buffer| {
        for idx in buffer.0.set_indices() {
            divan::black_box(idx);
        }
    });
}

#[divan::bench]
fn d001pct_random_arrow_collect(bencher: Bencher) {
    let buffer = make_very_sparse_random_arrow();
    bencher.with_inputs(|| &buffer).bench_refs(|buffer| {
        divan::black_box(buffer.0.set_indices().collect::<Vec<usize>>());
    });
}

#[divan::bench]
fn d001pct_random_vortex(bencher: Bencher) {
    let buffer = make_very_sparse_random();
    bencher.with_inputs(|| &buffer).bench_refs(|buffer| {
        for idx in buffer.set_indices() {
            divan::black_box(idx);
        }
    });
}

#[divan::bench]
fn d001pct_random_collect_precount(bencher: Bencher) {
    let buffer = make_very_sparse_random();
    let true_count = buffer.true_count();
    bencher.with_inputs(|| &buffer).bench_refs(|buffer| {
        divan::black_box(collect_set_indices_with_count(
            buffer.inner().as_slice(),
            buffer.offset(),
            buffer.len(),
            Some(true_count),
        ));
    });
}

// 1% density
bench_density!(1, uniform, make_uniform, make_uniform_arrow);
bench_density!(1, random, make_random, make_random_arrow);

// 2% density
bench_density!(2, uniform, make_uniform, make_uniform_arrow);
bench_density!(2, random, make_random, make_random_arrow);

// 3% density — near the sparse/medium boundary
bench_density!(3, uniform, make_uniform, make_uniform_arrow);
bench_density!(3, random, make_random, make_random_arrow);

// 4% density
bench_density!(4, uniform, make_uniform, make_uniform_arrow);
bench_density!(4, random, make_random, make_random_arrow);

// 5% density
bench_density!(5, uniform, make_uniform, make_uniform_arrow);
bench_density!(5, random, make_random, make_random_arrow);
bench_density!(5, clustered, make_clustered, make_uniform_arrow);

// 6% density
bench_density!(6, uniform, make_uniform, make_uniform_arrow);
bench_density!(6, random, make_random, make_random_arrow);

// 7% density
bench_density!(7, uniform, make_uniform, make_uniform_arrow);
bench_density!(7, random, make_random, make_random_arrow);

// 8% density
bench_density!(8, uniform, make_uniform, make_uniform_arrow);
bench_density!(8, random, make_random, make_random_arrow);

// 10% density
bench_density!(10, uniform, make_uniform, make_uniform_arrow);
bench_density!(10, random, make_random, make_random_arrow);
bench_density!(10, clustered, make_clustered, make_uniform_arrow);

// 20% density
bench_density!(20, uniform, make_uniform, make_uniform_arrow);
bench_density!(20, random, make_random, make_random_arrow);
bench_density!(20, clustered, make_clustered, make_uniform_arrow);

// 50% density (for reference)
bench_density!(50, uniform, make_uniform, make_uniform_arrow);

// =========================================================================
// Memory bandwidth baselines — measure the floor
// =========================================================================

/// Baseline: read 125KB bitmap + popcount (no output writes).
/// Measures the pure read + compute cost without any output bandwidth.
#[divan::bench]
fn baseline_read_popcount_1m(bencher: Bencher) {
    let buffer = make_random(5);
    bencher.with_inputs(|| &buffer).bench_refs(|buffer| {
        divan::black_box(buffer.true_count());
    });
}

/// Baseline: write N u32 values sequentially (no read, no computation).
/// Measures the pure output write bandwidth floor.
macro_rules! bench_write_baseline {
    ($count:literal) => {
        ::paste::paste! {
            #[divan::bench]
            #[allow(clippy::uninit_vec)]
            fn [< baseline_write_ $count _u32 >](bencher: Bencher) {
                bencher
                    .with_inputs(|| {
                        let mut v: Vec<u32> = Vec::with_capacity($count);
                        // SAFETY: we immediately overwrite all elements in the benchmark body.
                        unsafe { v.set_len($count) };
                        v
                    })
                    .bench_refs(|v| {
                        let ptr = v.as_mut_ptr();
                        for i in 0..$count as u32 {
                            unsafe { ptr.add(i as usize).write(i) };
                        }
                        divan::black_box(&v);
                    });
            }
        }
    };
}

// Write baselines matching typical set-bit counts at various densities:
// 1% of 1M = 10K, 5% = 50K, 10% = 100K, 20% = 200K, 50% = 500K
bench_write_baseline!(10_000);
bench_write_baseline!(50_000);
bench_write_baseline!(100_000);
bench_write_baseline!(200_000);
bench_write_baseline!(500_000);

/// Baseline: read 125KB bitmap sequentially (memcpy-like scan).
/// Measures the pure input read bandwidth.
#[divan::bench]
fn baseline_read_bitmap_125kb(bencher: Bencher) {
    let buffer = make_random(5);
    bencher.with_inputs(|| &buffer).bench_refs(|buffer| {
        let bytes = buffer.inner().as_slice();
        let mut acc = 0u64;
        let ptr = bytes.as_ptr() as *const u64;
        let n = bytes.len() / 8;
        for i in 0..n {
            acc ^= unsafe { *ptr.add(i) };
        }
        divan::black_box(acc);
    });
}

/// Baseline: read 125KB + write 50K u32 (combined bandwidth, no real computation).
/// This is the absolute floor for 5% density: just touch all the memory.
#[divan::bench]
#[allow(clippy::uninit_vec)]
fn baseline_read125kb_write50k(bencher: Bencher) {
    let buffer = make_random(5);
    bencher
        .with_inputs(|| {
            let mut v: Vec<u32> = Vec::with_capacity(50_000);
            unsafe { v.set_len(50_000) };
            (&buffer, v)
        })
        .bench_refs(|(buffer, v)| {
            let bytes = buffer.inner().as_slice();
            let src = bytes.as_ptr() as *const u64;
            let dst = v.as_mut_ptr();
            let n_words = bytes.len() / 8;
            let mut idx = 0u32;
            for i in 0..n_words {
                let w = unsafe { *src.add(i) };
                let pc = w.count_ones();
                for j in 0..pc {
                    unsafe { dst.add(idx as usize).write(idx + j) };
                }
                idx += pc;
            }
            divan::black_box(&v);
        });
}
