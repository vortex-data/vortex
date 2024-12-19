#![allow(clippy::unwrap_used)]

use std::iter::Iterator;

use arrow_buffer::ArrowNativeType;
use vortex_buffer::Buffer;

fn main() {
    divan::main();
}

// We wrap the Arrow Buffer so divan output distinguishes the type name.
pub struct ArrowBuffer<T: ArrowNativeType>(pub arrow_buffer::Buffer<T>);

impl<T: ArrowNativeType> FromIterator<T> for ArrowBuffer<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self(arrow_buffer::Buffer::from_iter(iter))
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
    B::from_iter((0..n).map(|i| i % i32::max_value()));
}
