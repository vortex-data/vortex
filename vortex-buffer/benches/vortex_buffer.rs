// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used)]

use std::iter::Iterator;

use arrow_buffer::ArrowNativeType;
use arrow_buffer::MutableBuffer;
use arrow_buffer::ScalarBuffer;
use divan::Bencher;
use num_traits::PrimInt;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::vortex_err;

fn main() {
    divan::main();
}

/// Wraps an arrow buffer so Divan can provide a nice name
pub struct Arrow<T>(T);

impl<T: ArrowNativeType> FromIterator<T> for Arrow<ScalarBuffer<T>> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self(ScalarBuffer::from_iter(iter))
    }
}

impl<T: ArrowNativeType> FromIterator<T> for Arrow<MutableBuffer> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self(MutableBuffer::from_iter(iter))
    }
}

const INPUT_SIZE: &[i32] = &[128, 1024, 2048, 16_384, 65_536];
const INPUT_SIZE_USIZE: &[usize] = &[128, 1024, 2048, 16_384, 65_536];

#[divan::bench(
    types = [Arrow<ScalarBuffer<i32>>,Buffer<i32>],
    args = INPUT_SIZE,
)]
fn from_iter<B: FromIterator<i32>>(n: i32) {
    B::from_iter((0..n).map(|i| i % i32::MAX));
}

trait MapEach<T, R> {
    type Output;

    fn map_each<F>(self, f: F) -> Self::Output
    where
        F: FnMut(T) -> R;
}

impl<T: ArrowNativeType, R: ArrowNativeType> MapEach<T, R> for Arrow<ScalarBuffer<T>> {
    type Output = Arrow<ScalarBuffer<R>>;

    fn map_each<F>(self, f: F) -> Self::Output
    where
        F: FnMut(T) -> R,
    {
        Arrow(ScalarBuffer::from(
            self.0
                .into_inner()
                .into_vec::<T>()
                .map_err(|_| vortex_err!("Failed to convert Arrow buffer into a mut vec"))
                .vortex_expect("Failed to convert Arrow buffer into a mut vec")
                .into_iter()
                .map(f)
                .collect::<Vec<R>>(),
        ))
    }
}

impl<T: Copy, R> MapEach<T, R> for Buffer<T> {
    type Output = BufferMut<R>;

    fn map_each<F>(self, f: F) -> Self::Output
    where
        F: FnMut(T) -> R,
    {
        Buffer::<T>::map_each_in_place(self, f)
    }
}

impl<T: Copy, R> MapEach<T, R> for BufferMut<T> {
    type Output = BufferMut<R>;

    fn map_each<F>(self, f: F) -> Self::Output
    where
        F: FnMut(T) -> R,
    {
        BufferMut::<T>::map_each_in_place(self, f)
    }
}

#[divan::bench(
    types = [Arrow<ScalarBuffer<i32>>, Buffer<i32>, BufferMut<i32>],
    args = INPUT_SIZE,
)]
fn map_each<B: MapEach<i32, u32> + FromIterator<i32>>(bencher: Bencher, n: i32) {
    bencher
        .with_inputs(|| B::from_iter((0..n).map(|i| i % i32::MAX)))
        .bench_values(|buffer| B::map_each(buffer, |i| (i as u32) + 1));
}

#[divan::bench(args = INPUT_SIZE)]
fn push_vortex_buffer(bencher: Bencher, length: i32) {
    bencher
        .with_inputs(|| BufferMut::<i32>::with_capacity(length as usize))
        .bench_refs(|buffer| {
            for idx in 0..length {
                buffer.push(divan::black_box(idx));
            }
        });
}

#[divan::bench(args = INPUT_SIZE)]
fn push_arrow_buffer(bencher: Bencher, length: i32) {
    bencher
        .with_inputs(|| {
            Arrow(MutableBuffer::with_capacity(
                length as usize * size_of::<i32>(),
            ))
        })
        .bench_refs(|buffer| {
            for idx in 0..length {
                buffer.0.push(divan::black_box(idx));
            }
        });
}

#[divan::bench(types = [u8, u16, u32, u64], args = INPUT_SIZE_USIZE)]
fn push_n_vortex_buffer<T: PrimInt>(bencher: Bencher, length: usize) {
    bencher
        .with_inputs(|| (BufferMut::<T>::with_capacity(length), length, T::one()))
        .bench_refs(|(buffer, length, one)| {
            for _ in 0..100 {
                unsafe { buffer.push_n_unchecked(*one, *length / 100) };
            }
        });
}

#[divan::bench(args = INPUT_SIZE)]
fn map_new_output(bencher: Bencher, n: i32) {
    bencher
        .with_inputs(|| {
            (
                Buffer::from_iter((0..n).map(|i| i % i32::MAX)),
                BufferMut::with_capacity(n as usize),
            )
        })
        .bench_refs(|(buffer, out)| {
            buffer
                .iter()
                .for_each(|&i| unsafe { out.push_unchecked((i as u32) + 1) })
        });
}
