// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used)]

use std::iter::Iterator;

use arrow_buffer::{BooleanBuffer, BooleanBufferBuilder};
use divan::Bencher;
use vortex_buffer::{BitBuffer, BitBufferMut};

fn main() {
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

#[divan::bench(
    types = [Arrow<BooleanBuffer>, BitBuffer],
    args = INPUT_SIZE,
)]
fn from_iter<B: FromIterator<bool>>(n: usize) {
    B::from_iter((0..n).map(|i| i % 2 == 0));
}

#[divan::bench(args = INPUT_SIZE)]
fn append_vortex_buffer(bencher: Bencher, length: usize) {
    bencher
        .with_inputs(|| BitBufferMut::with_capacity(length))
        .bench_refs(|buffer| {
            for idx in 0..length {
                buffer.append(divan::black_box(idx % 2 == 0));
            }
        });
}

#[divan::bench(args = INPUT_SIZE)]
fn append_arrow_buffer(bencher: Bencher, length: usize) {
    bencher
        .with_inputs(|| Arrow(BooleanBufferBuilder::new(length)))
        .bench_refs(|buffer| {
            for idx in 0..length {
                buffer.0.append(divan::black_box(idx % 2 == 0));
            }
        });
}

#[divan::bench(args = INPUT_SIZE)]
fn append_n_vortex_buffer(bencher: Bencher, length: usize) {
    bencher
        .with_inputs(|| BitBufferMut::with_capacity(length))
        .bench_refs(|buffer| {
            for _ in 0..100 {
                buffer.append_n(divan::black_box(true), divan::black_box(length / 100));
            }
        });
}

#[divan::bench(args = INPUT_SIZE)]
fn append_n_arrow_buffer(bencher: Bencher, length: usize) {
    bencher
        .with_inputs(|| Arrow(BooleanBufferBuilder::new(length)))
        .bench_refs(|buffer| {
            for _ in 0..100 {
                buffer
                    .0
                    .append_n(divan::black_box(length / 100), divan::black_box(true));
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
                dest.append_buffer(divan::black_box(source));
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
                for value in divan::black_box(&source.0).iter() {
                    dest.0.append(value);
                }
            }
        });
}

#[divan::bench(args = INPUT_SIZE)]
fn value_vortex_buffer(bencher: Bencher, length: usize) {
    let buffer = BitBuffer::from_iter((0..length).map(|i| i % 2 == 0));
    bencher.bench_local(|| {
        for idx in 0..length {
            divan::black_box(buffer.value(idx));
        }
    });
}

#[divan::bench(args = INPUT_SIZE)]
fn value_arrow_buffer(bencher: Bencher, length: usize) {
    let buffer = Arrow(BooleanBuffer::from_iter((0..length).map(|i| i % 2 == 0)));
    bencher.bench_local(|| {
        for idx in 0..length {
            divan::black_box(buffer.0.value(idx));
        }
    });
}

#[divan::bench(args = INPUT_SIZE)]
fn slice_vortex_buffer(bencher: Bencher, length: usize) {
    bencher
        .with_inputs(|| BitBuffer::from_iter((0..length).map(|i| i % 2 == 0)))
        .bench_values(|buffer| {
            let mid = length / 2;
            divan::black_box(buffer.slice(mid / 2..mid + mid / 2));
        });
}

#[divan::bench(args = INPUT_SIZE)]
fn slice_arrow_buffer(bencher: Bencher, length: usize) {
    bencher
        .with_inputs(|| Arrow(BooleanBuffer::from_iter((0..length).map(|i| i % 2 == 0))))
        .bench_values(|buffer| {
            let mid = length / 2;
            divan::black_box(buffer.0.slice(mid / 2, mid / 2));
        });
}

#[divan::bench(args = INPUT_SIZE)]
fn true_count_vortex_buffer(bencher: Bencher, length: usize) {
    let buffer = BitBuffer::from_iter((0..length).map(|i| i % 2 == 0));
    bencher.bench_local(|| {
        divan::black_box(buffer.true_count());
    })
}

#[divan::bench(args = INPUT_SIZE)]
fn true_count_arrow_buffer(bencher: Bencher, length: usize) {
    let buffer = Arrow(BooleanBuffer::from_iter((0..length).map(|i| i % 2 == 0)));
    bencher.bench_local(|| {
        divan::black_box(buffer.0.count_set_bits());
    });
}

#[divan::bench(args = INPUT_SIZE)]
fn bitwise_and_vortex_buffer(bencher: Bencher, length: usize) {
    bencher
        .with_inputs(|| {
            let a = BitBuffer::from_iter((0..length).map(|i| i % 2 == 0));
            let b = BitBuffer::from_iter((0..length).map(|i| i % 3 == 0));
            (a, b)
        })
        .bench_values(|(a, b)| {
            divan::black_box(&a & &b);
        });
}

#[divan::bench(args = INPUT_SIZE)]
fn bitwise_and_arrow_buffer(bencher: Bencher, length: usize) {
    bencher
        .with_inputs(|| {
            let a = Arrow(BooleanBuffer::from_iter((0..length).map(|i| i % 2 == 0)));
            let b = Arrow(BooleanBuffer::from_iter((0..length).map(|i| i % 3 == 0)));
            (a, b)
        })
        .bench_values(|(a, b)| {
            divan::black_box(&a.0 & &b.0);
        });
}

#[divan::bench(args = INPUT_SIZE)]
fn bitwise_or_vortex_buffer(bencher: Bencher, length: usize) {
    bencher
        .with_inputs(|| {
            let a = BitBuffer::from_iter((0..length).map(|i| i % 2 == 0));
            let b = BitBuffer::from_iter((0..length).map(|i| i % 3 == 0));
            (a, b)
        })
        .bench_values(|(a, b)| {
            divan::black_box(&a | &b);
        });
}

#[divan::bench(args = INPUT_SIZE)]
fn bitwise_or_arrow_buffer(bencher: Bencher, length: usize) {
    bencher
        .with_inputs(|| {
            let a = Arrow(BooleanBuffer::from_iter((0..length).map(|i| i % 2 == 0)));
            let b = Arrow(BooleanBuffer::from_iter((0..length).map(|i| i % 3 == 0)));
            (a, b)
        })
        .bench_values(|(a, b)| {
            divan::black_box(&a.0 | &b.0);
        });
}

#[divan::bench(args = INPUT_SIZE)]
fn bitwise_not_vortex_buffer(bencher: Bencher, length: usize) {
    bencher
        .with_inputs(|| BitBuffer::from_iter((0..length).map(|i| i % 2 == 0)))
        .bench_values(|buffer| {
            divan::black_box(!&buffer);
        });
}

#[divan::bench(args = INPUT_SIZE)]
fn bitwise_not_arrow_buffer(bencher: Bencher, length: usize) {
    bencher
        .with_inputs(|| Arrow(BooleanBuffer::from_iter((0..length).map(|i| i % 2 == 0))))
        .bench_values(|buffer| {
            divan::black_box(!&buffer.0);
        });
}

#[divan::bench(args = INPUT_SIZE)]
fn iter_vortex_buffer(bencher: Bencher, length: usize) {
    let buffer = BitBuffer::from_iter((0..length).map(|i| i % 2 == 0));
    bencher.bench_local(|| {
        for value in buffer.iter() {
            divan::black_box(value);
        }
    });
}

#[divan::bench(args = INPUT_SIZE)]
fn iter_arrow_buffer(bencher: Bencher, length: usize) {
    let buffer = Arrow(BooleanBuffer::from_iter((0..length).map(|i| i % 2 == 0)));
    bencher.bench_local(|| {
        for value in buffer.0.iter() {
            divan::black_box(value);
        }
    });
}

#[divan::bench(args = INPUT_SIZE)]
fn set_indices_vortex_buffer(bencher: Bencher, length: usize) {
    let buffer = BitBuffer::from_iter((0..length).map(|i| i % 2 == 0));
    bencher.bench_local(|| {
        for idx in buffer.set_indices() {
            divan::black_box(idx);
        }
    });
}

#[divan::bench(args = INPUT_SIZE)]
fn set_indices_arrow_buffer(bencher: Bencher, length: usize) {
    let buffer = Arrow(BooleanBuffer::from_iter((0..length).map(|i| i % 2 == 0)));
    bencher.bench_local(|| {
        for idx in buffer.0.set_indices() {
            divan::black_box(idx);
        }
    });
}
