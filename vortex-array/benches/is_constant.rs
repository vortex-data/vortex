#![allow(clippy::unwrap_used)]

use divan::Bencher;
use num_traits::Bounded;
use rand::distr::uniform::SampleUniform;
use rand::prelude::StdRng;
use rand::{Rng, SeedableRng};
use vortex_array::compute::IsConstantOpts;
use vortex_array::{Array, IntoArray};
use vortex_buffer::Buffer;
use vortex_dtype::NativePType;

fn main() {
    divan::main();
}

const ARRAY_SIZES: &[usize] = &[10_000, 262144, 10_000_000];

// f16 is omitted since it doesn't implement SampleUniform
#[divan::bench(types = [u8, u16, u32, u64, f32, f64], args = ARRAY_SIZES)]
fn primitive_is_constant<T: SampleUniform + PartialOrd + NativePType + Bounded>(
    bencher: Bencher,
    size: usize,
) {
    let mut rng = StdRng::seed_from_u64(0);
    let value = rng.random_range(T::zero()..T::max_value());

    let arr = Buffer::full(value, size).into_array();

    bencher.with_inputs(|| &arr).bench_refs(|arr| {
        arr.vtable()
            .is_constant_fn()
            .unwrap()
            .is_constant(*arr, &IsConstantOpts::default())
            .unwrap()
    });
}
