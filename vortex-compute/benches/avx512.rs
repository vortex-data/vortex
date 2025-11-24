// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::cast_possible_truncation)]

use itertools::Itertools;
use rand::Rng;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use vortex_compute::filter::slice::in_place::avx512::filter_in_place_avx512;
use vortex_compute::filter::slice::in_place::filter_in_place_scalar;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use vortex_compute::filter::slice::out::avx512::filter_into_avx512;
use vortex_compute::filter::slice::out::filter_into_scalar;

fn main() {
    divan::main();
}

// Create a random mask where each bit has `probability` chance of being set.
fn create_random_mask(size: usize, probability: f64) -> Vec<u8> {
    let mut rng = rand::rng();
    let num_bytes = size.div_ceil(8);
    let mut mask = Vec::with_capacity(num_bytes);

    for _ in 0..num_bytes {
        let mut byte = 0u8;
        for bit in 0..8 {
            if rng.random::<f64>() < probability {
                byte |= 1 << bit;
            }
        }
        mask.push(byte);
    }

    mask
}

/// Benchmark different data sizes.
const SIZES: &[usize] = &[1 << 10, 1 << 11, 1 << 14, 1 << 17];

/// Different probability values to benchmark.
const PROBABILITIES: &[f64] = &[0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0];

/// The number of samples per benchmark.
const SAMPLE_SIZE: u32 = 64;

#[divan::bench(sample_size = SAMPLE_SIZE, args = SIZES.iter().copied().cartesian_product(PROBABILITIES.iter().copied()))]
fn in_place_scalar(bencher: divan::Bencher, (size, probability): (usize, f64)) {
    let mask = divan::black_box(create_random_mask(size, probability));
    let data = (0..size as i32).collect::<Vec<_>>();
    bencher
        .with_inputs(|| (data.clone(), &mask))
        .bench_refs(|(data, mask)| filter_in_place_scalar(data, mask))
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), not(codspeed)))]
#[divan::bench(sample_size = SAMPLE_SIZE, args = SIZES.iter().copied().cartesian_product(PROBABILITIES.iter().copied()))]
fn in_place_avx512(bencher: divan::Bencher, (size, probability): (usize, f64)) {
    let mask = divan::black_box(create_random_mask(size, probability));
    let data = (0..size as i32).collect::<Vec<_>>();
    bencher
        .with_inputs(|| (data.clone(), &mask))
        .bench_refs(|(data, mask)| unsafe { filter_in_place_avx512(data, mask) })
}

#[divan::bench(sample_size = SAMPLE_SIZE, args = SIZES.iter().copied().cartesian_product(PROBABILITIES.iter().copied()))]
fn out_scalar(bencher: divan::Bencher, (size, probability): (usize, f64)) {
    let mask = divan::black_box(create_random_mask(size, probability));
    let src = (0..size as i32).collect::<Vec<_>>();
    bencher
        .with_inputs(|| {
            let dest = vec![0i32; size];
            (src.clone(), dest, &mask)
        })
        .bench_refs(|(src, dest, mask)| filter_into_scalar(src, dest, mask))
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), not(codspeed)))]
#[divan::bench(sample_size = SAMPLE_SIZE, args = SIZES.iter().copied().cartesian_product(PROBABILITIES.iter().copied()))]
fn out_avx512(bencher: divan::Bencher, (size, probability): (usize, f64)) {
    let mask = divan::black_box(create_random_mask(size, probability));
    let src = (0..size as i32).collect::<Vec<_>>();
    bencher
        .with_inputs(|| {
            let dest = vec![0i32; size];
            (src.clone(), dest, &mask)
        })
        .bench_refs(|(src, dest, mask)| unsafe { filter_into_avx512(src, dest, mask) })
}
