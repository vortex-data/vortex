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
// Density-varied benchmarks: 100k bits at different set-bit densities
// ---------------------------------------------------------------------------

const LARGE_N: usize = 100_000;

/// 1% density (sparse)
fn make_sparse() -> BitBuffer {
    BitBuffer::from_iter((0..LARGE_N).map(|i| i % 100 == 0))
}

/// 50% density (dense)
fn make_dense() -> BitBuffer {
    BitBuffer::from_iter((0..LARGE_N).map(|i| i % 2 == 0))
}

/// 99% density (nearly all set)
fn make_nearly_full() -> BitBuffer {
    BitBuffer::from_iter((0..LARGE_N).map(|i| i % 100 != 0))
}

fn make_sparse_arrow() -> Arrow<BooleanBuffer> {
    Arrow(BooleanBuffer::from_iter((0..LARGE_N).map(|i| i % 100 == 0)))
}

fn make_dense_arrow() -> Arrow<BooleanBuffer> {
    Arrow(BooleanBuffer::from_iter((0..LARGE_N).map(|i| i % 2 == 0)))
}

fn make_nearly_full_arrow() -> Arrow<BooleanBuffer> {
    Arrow(BooleanBuffer::from_iter((0..LARGE_N).map(|i| i % 100 != 0)))
}

// --- Arrow baseline at different densities ---

#[divan::bench]
fn density_1pct_arrow(bencher: Bencher) {
    let buffer = make_sparse_arrow();
    bencher.with_inputs(|| &buffer).bench_refs(|buffer| {
        for idx in buffer.0.set_indices() {
            divan::black_box(idx);
        }
    });
}

#[divan::bench]
fn density_50pct_arrow(bencher: Bencher) {
    let buffer = make_dense_arrow();
    bencher.with_inputs(|| &buffer).bench_refs(|buffer| {
        for idx in buffer.0.set_indices() {
            divan::black_box(idx);
        }
    });
}

#[divan::bench]
fn density_99pct_arrow(bencher: Bencher) {
    let buffer = make_nearly_full_arrow();
    bencher.with_inputs(|| &buffer).bench_refs(|buffer| {
        for idx in buffer.0.set_indices() {
            divan::black_box(idx);
        }
    });
}

// --- Current vortex (delegates to Arrow) ---

#[divan::bench]
fn density_1pct_vortex_current(bencher: Bencher) {
    let buffer = make_sparse();
    bencher.with_inputs(|| &buffer).bench_refs(|buffer| {
        for idx in buffer.set_indices() {
            divan::black_box(idx);
        }
    });
}

#[divan::bench]
fn density_50pct_vortex_current(bencher: Bencher) {
    let buffer = make_dense();
    bencher.with_inputs(|| &buffer).bench_refs(|buffer| {
        for idx in buffer.set_indices() {
            divan::black_box(idx);
        }
    });
}

#[divan::bench]
fn density_99pct_vortex_current(bencher: Bencher) {
    let buffer = make_nearly_full();
    bencher.with_inputs(|| &buffer).bench_refs(|buffer| {
        for idx in buffer.set_indices() {
            divan::black_box(idx);
        }
    });
}

// --- New scalar iterator ---

#[divan::bench]
fn density_1pct_scalar_iter(bencher: Bencher) {
    let buffer = make_sparse();
    bencher.with_inputs(|| &buffer).bench_refs(|buffer| {
        for idx in
            ScalarBitIndexIterator::new(buffer.inner().as_slice(), buffer.offset(), buffer.len())
        {
            divan::black_box(idx);
        }
    });
}

#[divan::bench]
fn density_50pct_scalar_iter(bencher: Bencher) {
    let buffer = make_dense();
    bencher.with_inputs(|| &buffer).bench_refs(|buffer| {
        for idx in
            ScalarBitIndexIterator::new(buffer.inner().as_slice(), buffer.offset(), buffer.len())
        {
            divan::black_box(idx);
        }
    });
}

#[divan::bench]
fn density_99pct_scalar_iter(bencher: Bencher) {
    let buffer = make_nearly_full();
    bencher.with_inputs(|| &buffer).bench_refs(|buffer| {
        for idx in
            ScalarBitIndexIterator::new(buffer.inner().as_slice(), buffer.offset(), buffer.len())
        {
            divan::black_box(idx);
        }
    });
}

// --- Bulk scalar collect ---

#[divan::bench]
fn density_1pct_collect_scalar(bencher: Bencher) {
    let buffer = make_sparse();
    bencher.with_inputs(|| &buffer).bench_refs(|buffer| {
        divan::black_box(collect_set_indices_scalar(
            buffer.inner().as_slice(),
            buffer.offset(),
            buffer.len(),
        ));
    });
}

#[divan::bench]
fn density_50pct_collect_scalar(bencher: Bencher) {
    let buffer = make_dense();
    bencher.with_inputs(|| &buffer).bench_refs(|buffer| {
        divan::black_box(collect_set_indices_scalar(
            buffer.inner().as_slice(),
            buffer.offset(),
            buffer.len(),
        ));
    });
}

#[divan::bench]
fn density_99pct_collect_scalar(bencher: Bencher) {
    let buffer = make_nearly_full();
    bencher.with_inputs(|| &buffer).bench_refs(|buffer| {
        divan::black_box(collect_set_indices_scalar(
            buffer.inner().as_slice(),
            buffer.offset(),
            buffer.len(),
        ));
    });
}

// --- Bulk SIMD/BMI2 collect ---

#[divan::bench]
fn density_1pct_collect_simd(bencher: Bencher) {
    let buffer = make_sparse();
    bencher.with_inputs(|| &buffer).bench_refs(|buffer| {
        divan::black_box(collect_set_indices(
            buffer.inner().as_slice(),
            buffer.offset(),
            buffer.len(),
        ));
    });
}

#[divan::bench]
fn density_50pct_collect_simd(bencher: Bencher) {
    let buffer = make_dense();
    bencher.with_inputs(|| &buffer).bench_refs(|buffer| {
        divan::black_box(collect_set_indices(
            buffer.inner().as_slice(),
            buffer.offset(),
            buffer.len(),
        ));
    });
}

#[divan::bench]
fn density_99pct_collect_simd(bencher: Bencher) {
    let buffer = make_nearly_full();
    bencher.with_inputs(|| &buffer).bench_refs(|buffer| {
        divan::black_box(collect_set_indices(
            buffer.inner().as_slice(),
            buffer.offset(),
            buffer.len(),
        ));
    });
}
