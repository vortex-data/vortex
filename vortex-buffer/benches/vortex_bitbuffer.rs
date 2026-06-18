// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::iter::Iterator;

use arrow_buffer::BooleanBuffer;
use arrow_buffer::BooleanBufferBuilder;
use divan::Bencher;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;

fn main() {
    // Pre-warm CPUID feature detection so the one-time probe cost is never
    // included in any benchmark iteration.
    #[cfg(target_arch = "x86_64")]
    {
        let _ = is_x86_feature_detected!("avx2");
        let _ = is_x86_feature_detected!("avx512f");
        let _ = is_x86_feature_detected!("avx512vpopcntdq");
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

/// Owned-LHS AND: the left operand is a fresh, uniquely-owned `BitBuffer` each iteration, so
/// `bitwise_binary_op_lhs_owned` takes the in-place (zero-allocation) fast path. Compare against
/// `bitwise_and_vortex_buffer` (reference-LHS, which always allocates a result buffer).
#[divan::bench(args = INPUT_SIZE)]
fn bitand_owned_lhs_vortex_buffer(bencher: Bencher, length: usize) {
    let b = BitBuffer::from_iter((0..length).map(|i| i % 3 == 0));
    bencher
        .with_inputs(|| BitBuffer::from_iter((0..length).map(|i| i % 2 == 0)))
        .bench_values(|a| a & &b);
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
