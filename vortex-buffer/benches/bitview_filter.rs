// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used)]

use std::hint::black_box;
use std::iter::Iterator;

use divan::Bencher;
use rand::prelude::StdRng;
use rand::{Rng, SeedableRng};
use vortex_buffer::bench::bench_filter_in_place_scalar;
use vortex_buffer::{buffer_mut, BitBuffer};

fn main() {
    divan::main();
}

// Focus on benchmarking for our known vector length.
const N: usize = 1024;
type BitView<'a> = vortex_buffer::BitView<'a, 128>;
const MASK_DENSITY: &[f64] = &[
    0.0, 0.01, 0.05, 0.1, 0.25, // 0.3,
    // 0.4,
    0.5,  // 0.6,
    0.75, // 0.85,
    0.9,  // 0.95,
    0.99, 1.00,
];

#[divan::bench(
    types = [u8, u16, u32, u64, u128],
    args = MASK_DENSITY,
)]
fn filter_scalar_in_place<T: Default + Copy>(bencher: Bencher, mask_density: f64) {
    let mut buffer = buffer_mut![T::default(); N];

    let mut rng = StdRng::seed_from_u64(0);
    let mask = (0..N)
        .map(|_| rng.random_bool(mask_density))
        .collect::<BitBuffer>();

    bencher.bench_local(|| {
        let view = BitView::new(mask.inner().as_ref().try_into().unwrap());
        bench_filter_in_place_scalar(&view, &mut buffer);
        black_box(&mut buffer);
    });
}
