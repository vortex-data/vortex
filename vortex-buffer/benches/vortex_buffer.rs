#![allow(clippy::unwrap_used)]

use std::hint::black_box;
use std::iter::Iterator;

use arrow_buffer::{ArrowNativeType, ScalarBuffer};
use divan::Bencher;
use vortex_buffer::{Buffer, BufferMut};

fn main() {
    divan::main();
}

// We wrap the Arrow Buffer so divan output distinguishes the type name.
pub struct ArrowBuffer<T: ArrowNativeType>(pub ScalarBuffer<T>);

impl<T: ArrowNativeType> FromIterator<T> for ArrowBuffer<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self(ScalarBuffer::from_iter(iter))
    }
}

#[divan::bench(
    types = [
        ArrowBuffer<i32>,
        Buffer<i32>,
    ],
    args = [1, 100, 10_00, 100_000, 10_000_000],
)]
fn from_iter<B: FromIterator<i32>>(n: i32) {
    B::from_iter((0..n).map(|i| i % i32::MAX));
}

#[divan::bench()]
fn map_each_arrow(bencher: Bencher) {
    bencher
        .with_inputs(|| ScalarBuffer::<i32>::from_iter((0..1_000_000i32).map(|i| i % i32::MAX)))
        .bench_local_values(|buffer| {
            black_box(
                buffer
                    .into_inner()
                    .into_vec::<i32>()
                    .expect("Failed to convert Arrow buffer into a mut vec")
                    .into_iter()
                    .map(|i| (i as u32) + 1)
                    .collect::<Vec<u32>>(),
            );
        });
}

#[divan::bench()]
fn map_each_vortex(bencher: Bencher) {
    let buffer = BufferMut::<i32>::from_iter((0..1_000_000i32).map(|i| i % i32::MAX));

    bencher.bench_local(move || {
        let buffer = buffer.clone();
        buffer.map_each(|i| (*i as u32) + 1)
    });
}
