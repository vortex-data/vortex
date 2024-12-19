#![allow(clippy::unwrap_used)]

use std::iter::Iterator;

use arrow_buffer::ArrowNativeType;
use vortex_buffer::ScalarBuffer;

fn main() {
    divan::main();
}

// We wrap the Arrow ScalarBuffer so divan output distinguishes the type name.
pub struct ArrowScalarBuffer<T: ArrowNativeType>(pub arrow_buffer::ScalarBuffer<T>);

impl<T: ArrowNativeType> FromIterator<T> for ArrowScalarBuffer<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self(arrow_buffer::ScalarBuffer::from_iter(iter))
    }
}

#[divan::bench(
    types = [
        ArrowScalarBuffer<i32>,
        ScalarBuffer<i32>,
    ],
    args = [1, 100, 10_00, 100_000, 10_000_000],
)]
fn from_iter<B: FromIterator<i32>>(n: i32) {
    B::from_iter((0..n).map(|i| i % i32::max_value()));
}
