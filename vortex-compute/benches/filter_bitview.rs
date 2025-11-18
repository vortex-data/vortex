// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used)]

use std::hint::black_box;
use std::iter::Iterator;

use divan::Bencher;
use rand::prelude::StdRng;
use rand::{Rng, SeedableRng};
use vortex_buffer::{buffer_mut, BitBuffer};
use vortex_compute::bench;
use vortex_compute::filter::Filter;

fn main() {
    divan::main();
}

// Focus on benchmarking for our known vector length.
const N: usize = 1024;
type BitView<'a> = vortex_buffer::BitView<'a, 128>;

trait FilterImpl {
    fn filter<'a, T: Copy>(bitview: &BitView, slice: &'a mut [T]) -> &'a mut [T];
}

/// The main entry point for the filter function that performs all the dispatch.
struct ActualFilter;
impl FilterImpl for ActualFilter {
    fn filter<'a, T: Copy>(bitview: &BitView, slice: &'a mut [T]) -> &'a mut [T] {
        slice.filter(bitview)
    }
}

struct ScalarFilter;
impl FilterImpl for ScalarFilter {
    fn filter<'a, T: Copy>(bitview: &BitView, slice: &'a mut [T]) -> &'a mut [T] {
        bench::bench_filter_scalar::<_, T>(bitview, slice)
    }
}

struct NeonFilter;
impl FilterImpl for NeonFilter {
    fn filter<'a, T: Copy>(bitview: &BitView, slice: &'a mut [T]) -> &'a mut [T] {
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                return bench::bench_filter_neon::<_, T>(bitview, slice);
            }
        }

        // Otherwise, do nothing.
        let _ = bitview;
        slice
    }
}

const MASK_DENSITY: &[f64] = &[
    0.0, 0.01, 0.05, 0.1, 0.25, // 0.3,
    // 0.4,
    0.5,  // 0.6,
    0.75, // 0.85,
    0.9,  // 0.95,
    0.99, 1.00,
];

#[divan::bench(types = [ScalarFilter, NeonFilter, ActualFilter], args = MASK_DENSITY)]
fn filter_u8<F: FilterImpl>(bencher: Bencher, mask_density: f64) {
    bench_filter_fn::<F, u8>(bencher, mask_density)
}

#[divan::bench(types = [ScalarFilter, NeonFilter, ActualFilter], args = MASK_DENSITY)]
fn filter_u16<F: FilterImpl>(bencher: Bencher, mask_density: f64) {
    bench_filter_fn::<F, u16>(bencher, mask_density)
}

#[divan::bench(types = [ScalarFilter, NeonFilter, ActualFilter], args = MASK_DENSITY)]
fn filter_u32<F: FilterImpl>(bencher: Bencher, mask_density: f64) {
    bench_filter_fn::<F, u32>(bencher, mask_density)
}

#[divan::bench(types = [ScalarFilter, NeonFilter, ActualFilter], args = MASK_DENSITY)]
fn filter_u64<F: FilterImpl>(bencher: Bencher, mask_density: f64) {
    bench_filter_fn::<F, u64>(bencher, mask_density)
}

#[divan::bench(types = [ScalarFilter, NeonFilter, ActualFilter], args = MASK_DENSITY)]
fn filter_u128<F: FilterImpl>(bencher: Bencher, mask_density: f64) {
    bench_filter_fn::<F, u128>(bencher, mask_density)
}

fn bench_filter_fn<F: FilterImpl, T: Default + Copy>(bencher: Bencher, mask_density: f64) {
    let mut buffer = buffer_mut![T::default(); N];

    let mut rng = StdRng::seed_from_u64(0);
    let mask = (0..N)
        .map(|_| rng.random_bool(mask_density))
        .collect::<BitBuffer>();

    bencher.bench_local(|| {
        let view = BitView::new(mask.inner().as_ref().try_into().unwrap());
        black_box(F::filter(&view, &mut buffer));
    });
}
