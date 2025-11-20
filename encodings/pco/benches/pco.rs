// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used)]

use divan::Bencher;
use mimalloc::MiMalloc;
use rand::prelude::StdRng;
use rand::{Rng, SeedableRng};
use vortex_array::compute::{filter, warm_up_vtables};
use vortex_array::{IntoArray, ToCanonical};
use vortex_buffer::{BitBuffer, BufferMut};
use vortex_mask::Mask;
use vortex_pco::PcoArray;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

pub fn main() {
    warm_up_vtables();
    divan::main();
}

#[divan::bench(args = [
    (10_000, 0.1),
    (10_000, 0.5),
    (10_000, 0.9),
    (10_000, 1.0),
    (50_000, 0.1),
    (50_000, 0.5),
    (50_000, 0.9),
    (50_000, 1.0),
    (100_000, 0.1),
    (100_000, 0.5),
    (100_000, 0.9),
    (100_000, 1.0)]
)]
pub fn pco_pipeline(bencher: Bencher, (size, selectivity): (usize, f64)) {
    let mut rng = StdRng::seed_from_u64(42);
    #[allow(clippy::cast_possible_truncation)]
    let values = (0..size)
        .map(|i| (i % 10000) as i32)
        .collect::<BufferMut<i32>>()
        .into_array()
        .to_primitive();

    let pco_array = PcoArray::from_primitive(&values, 3, 0).unwrap();
    let mask = (0..size)
        .map(|_| rng.random_bool(selectivity))
        .collect::<BitBuffer>();

    bencher
        .with_inputs(|| (Mask::from_buffer(mask.clone()), pco_array.clone()))
        .bench_refs(|(mask, pco_array)| pco_array.execute_with_selection(&mask).unwrap());
}

#[divan::bench(args = [
    (10_000, 0.1),
    (10_000, 0.5),
    (10_000, 0.9),
    (10_000, 1.0),
    (50_000, 0.1),
    (50_000, 0.5),
    (50_000, 0.9),
    (50_000, 1.0),
    (100_000, 0.1),
    (100_000, 0.5),
    (100_000, 0.9),
    (100_000, 1.0)]
)]
pub fn pco_canonical(bencher: Bencher, (size, selectivity): (usize, f64)) {
    let mut rng = StdRng::seed_from_u64(42);
    #[allow(clippy::cast_possible_truncation)]
    let values = (0..size)
        .map(|i| (i % 10000) as i32)
        .collect::<BufferMut<i32>>()
        .into_array()
        .to_primitive();

    let pco_array = PcoArray::from_primitive(&values, 3, 0).unwrap();
    let mask = (0..size)
        .map(|_| rng.random_bool(selectivity))
        .collect::<BitBuffer>();

    bencher
        .with_inputs(|| (Mask::from_buffer(mask.clone()), pco_array.clone()))
        .bench_refs(|(mask, pco_array)| filter(pco_array.to_canonical().as_ref(), &mask).unwrap());
}
