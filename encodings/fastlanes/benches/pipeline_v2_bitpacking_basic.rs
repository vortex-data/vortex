// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used)]
#![allow(unexpected_cfgs)]

use divan::Bencher;
use mimalloc::MiMalloc;
use rand::prelude::StdRng;
use rand::{Rng, SeedableRng};
use vortex_array::arrays::PrimitiveArray;
use vortex_fastlanes::BitPackedArray;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

pub fn main() {
    divan::main();
}

// Cross product of NUM_ELEMENTS and VALIDITY_PCT.
const BENCH_PARAMS: &[(usize, f64)] = &[
    (1_000, 0.5),
    (1_000, 1.0),
    (10_000, 0.5),
    (10_000, 1.0),
    (100_000, 0.5),
    (100_000, 1.0),
];

#[divan::bench(args = BENCH_PARAMS)]
fn bitpack_pipeline_unpack(bencher: Bencher, (num_elements, validity_pct): (usize, f64)) {
    let mut rng = StdRng::seed_from_u64(42);

    // Create array with randomized validity.
    // Keep values small enough to fit in the bit width (0-1023 for 10 bits).
    let values = (0..num_elements).map(|_| {
        let is_valid = rng.random_bool(validity_pct);
        is_valid.then(|| rng.random_range(0u32..1024))
    });

    let primitive = PrimitiveArray::from_option_iter(values).to_array();

    // Encode with 10-bit width (supports values up to 1023).
    let bitpacked = BitPackedArray::encode(&primitive, 10).unwrap();

    bencher
        .with_inputs(|| bitpacked.to_array())
        .bench_refs(|array| array.execute().unwrap());
}

#[divan::bench(args = BENCH_PARAMS)]
fn bitpack_canonical_unpack(bencher: Bencher, (num_elements, validity_pct): (usize, f64)) {
    let mut rng = StdRng::seed_from_u64(42);

    // Create array with randomized validity.
    // Keep values small enough to fit in the bit width (0-1023 for 10 bits).
    let values = (0..num_elements).map(|_| {
        let is_valid = rng.random_bool(validity_pct);
        is_valid.then(|| rng.random_range(0u32..1024))
    });

    let primitive = PrimitiveArray::from_option_iter(values).to_array();

    // Encode with 10-bit width (supports values up to 1023).
    let bitpacked = BitPackedArray::encode(&primitive, 10).unwrap();
    let array = bitpacked.to_array();

    bencher
        .with_inputs(|| &array)
        .bench_refs(|array| array.to_canonical());
}
