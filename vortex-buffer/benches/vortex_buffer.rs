#![allow(clippy::unwrap_used)]

use std::iter::Iterator;

use arrow_buffer::{ArrowNativeType, MutableBuffer, ScalarBuffer};
use divan::Bencher;
use vortex_buffer::{Buffer, BufferMut};
use vortex_error::{VortexExpect, vortex_err};

fn main() {
    divan::main();
}

// We wrap the Arrow Buffer so the Divan output has a nice name!!
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

#[divan::bench(
    types = [Arrow<ScalarBuffer<i32>>,Buffer<i32>],
    args = [1, 100, 1_000, 100_000, 10_000_000],
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

impl<T: Copy, R> MapEach<T, R> for BufferMut<T> {
    type Output = BufferMut<R>;

    fn map_each<F>(self, f: F) -> Self::Output
    where
        F: FnMut(T) -> R,
    {
        BufferMut::<T>::map_each(self, f)
    }
}

#[divan::bench(
    types = [Arrow<ScalarBuffer<i32>>, BufferMut<i32>],
    args = [1, 100, 1_000, 100_000, 10_000_000],
)]
fn map_each<B: MapEach<i32, u32> + FromIterator<i32>>(bencher: Bencher, n: i32) {
    bencher
        .with_inputs(|| B::from_iter((0..n).map(|i| i % i32::MAX)))
        .bench_values(|buffer| B::map_each(buffer, |i| (i as u32) + 1));
}

#[divan::bench(args = [100, 1_000, 10_000, 100_000, 1_000_000])]
fn push_vortex_buffer(bencher: Bencher, length: i32) {
    bencher
        .with_inputs(|| BufferMut::<i32>::with_capacity(length as usize))
        .bench_refs(|buffer| {
            for idx in 0..length {
                buffer.push(divan::black_box(idx));
            }
        });
}

#[divan::bench(args = [100, 1_000, 10_000, 100_000, 1_000_000])]
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
