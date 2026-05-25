// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks comparing [`PVector`] take vs [`DictArray`] canonicalization.
//!
//! Both are tracked by number of indices/codes for fair comparison.

#![expect(clippy::cast_possible_truncation)]
#![expect(clippy::unwrap_used)]

use divan::Bencher;
use rand::distr::Uniform;
use rand::prelude::*;
use rand_distr::Zipf;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::PrimitiveArray;

fn main() {
    divan::main();
}

/// Number of indices to take.
const NUM_INDICES: &[usize] = &[1_000, 10_000, 100_000];

// --- Direct take by value width ---
//
// The AVX2 take kernel only accelerates 32- and 64-bit value types; 8- and 16-bit value
// columns fall back to a scalar bounds-checked gather. These benches measure take across value
// widths (u8/u16/u32/u64) with a fixed-size value array and many u32 indices.

/// Sizes of the value array that indices address into, spanning in-cache (4K) to
/// out-of-cache (4M elements ~ 16MB for u32, exceeding typical L3).
const VALUE_LENS: &[usize] = &[4_096, 262_144, 4_194_304];

macro_rules! take_by_width {
    ($name:ident, $ty:ty) => {
        #[divan::bench(args = VALUE_LENS, sample_count = 2_000)]
        fn $name(bencher: Bencher, value_len: usize) {
            const NUM_INDICES: usize = 100_000;
            #[allow(clippy::cast_possible_truncation)]
            let values =
                PrimitiveArray::from_iter((0..value_len as u64).map(|i| i as $ty)).into_array();

            let rng = StdRng::seed_from_u64(0);
            let range = Uniform::new(0u32, value_len as u32).unwrap();
            let indices =
                PrimitiveArray::from_iter(rng.sample_iter(range).take(NUM_INDICES)).into_array();

            bencher
                .with_inputs(|| (values.clone(), indices.clone()))
                .bench_values(|(values, indices)| {
                    values
                        .take(indices)
                        .unwrap()
                        .execute::<Canonical>(&mut LEGACY_SESSION.create_execution_ctx())
                });
        }
    };
}

take_by_width!(take_u8_values, u8);
take_by_width!(take_u16_values, u16);
take_by_width!(take_u32_values, u32);
take_by_width!(take_u64_values, u64);

/// Size of the source vector / dictionary values.
const VECTOR_SIZE: &[usize] = &[16, 256, 2048, 8192];

// --- DictArray canonicalization benchmarks ---

#[divan::bench(args = NUM_INDICES, consts = VECTOR_SIZE, sample_count = 100_000)]
fn dict_canonicalize_uniform<const NUM_VALUES: usize>(bencher: Bencher, num_indices: usize) {
    let values = PrimitiveArray::from_iter(0..NUM_VALUES as u32);

    let rng = StdRng::seed_from_u64(0);
    let range = Uniform::new(0u32, NUM_VALUES as u32).unwrap();
    let codes = PrimitiveArray::from_iter(rng.sample_iter(range).take(num_indices));

    let dict = DictArray::try_new(codes.into_array(), values.into_array()).unwrap();

    bencher.with_inputs(|| &dict).bench_refs(|dict| {
        (*dict)
            .clone()
            .into_array()
            .execute::<Canonical>(&mut LEGACY_SESSION.create_execution_ctx())
    });
}

#[divan::bench(args = NUM_INDICES, consts = VECTOR_SIZE, sample_count = 100_000)]
fn dict_canonicalize_zipfian<const NUM_VALUES: usize>(bencher: Bencher, num_indices: usize) {
    let values = PrimitiveArray::from_iter(0..NUM_VALUES as u32);

    let rng = StdRng::seed_from_u64(0);
    let zipf = Zipf::new(NUM_VALUES as f64, 1.0).unwrap();
    let codes = PrimitiveArray::from_iter(
        rng.sample_iter(&zipf)
            .take(num_indices)
            .map(|i: f64| (i as u32 - 1).min(NUM_VALUES as u32 - 1)),
    );

    let dict = DictArray::try_new(codes.into_array(), values.into_array()).unwrap();

    bencher.with_inputs(|| &dict).bench_refs(|dict| {
        (*dict)
            .clone()
            .into_array()
            .execute::<Canonical>(&mut LEGACY_SESSION.create_execution_ctx())
    });
}
