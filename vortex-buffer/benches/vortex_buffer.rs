// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::iter::Iterator;

use divan::Bencher;
use num_traits::PrimInt;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;

fn main() {
    divan::main();
}

const INPUT_SIZE: &[i32] = &[128, 1024, 2048, 16_384, 65_536];
const INPUT_SIZE_USIZE: &[usize] = &[128, 1024, 2048, 16_384, 65_536];

#[divan::bench(
    types = [Buffer<i32>],
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
    types = [Buffer<i32>, BufferMut<i32>],
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
