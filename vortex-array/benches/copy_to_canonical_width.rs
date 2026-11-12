// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Copies of unscaled decimal values into the canonical width of a target dtype.
//!
//! Each widening case copies values of one native type into a wider canonical width, which
//! [`copy_to_canonical_width`] does without a check. The `same_width_i64` control is a plain copy.
//! Widening never reads the validity, so every case has no nulls.
//!
//! Every case carries `#[cpu_features]`, so it is measured on every walltime CPU-feature leg
//! rather than in simulation: the widening loop is written once, and how the compiler vectorizes
//! it depends on the build flags.

#![expect(clippy::expect_used)]

use divan::Bencher;
use mimalloc::MiMalloc;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::arrays::decimal::copy_to_canonical_width;
use vortex_array::dtype::BigCast;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::NativeDecimalType;
use vortex_mask::Mask;

// The copy allocates its output inside the timed region, so the allocator is fixed to keep the
// system allocator out of the measurement.
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    divan::main();
}

const LEN: usize = 16_384;

/// Copies [`LEN`] random values of `F`, each within `source_precision` digits, into the canonical
/// width of `target`.
fn bench_copy<F: NativeDecimalType>(bencher: Bencher, source_precision: u32, target: DecimalDType) {
    let mut rng = StdRng::seed_from_u64(42);
    let max = 10i64.pow(source_precision) - 1;
    let values: Vec<F> = (0..LEN)
        .map(|_| {
            <F as BigCast>::from(rng.random_range(-max..=max))
                .expect("the source precision fits the source type")
        })
        .collect();
    let validity = Mask::new_true(LEN);

    bencher.bench_local(|| {
        copy_to_canonical_width(&values, &validity, target).expect("every valid value fits target")
    });
}

#[vortex_bench_support::cpu_features]
#[divan::bench]
fn i32_to_i64(bencher: Bencher) {
    bench_copy::<i32>(bencher, 9, DecimalDType::new(18, 2));
}

#[vortex_bench_support::cpu_features]
#[divan::bench]
fn i64_to_i128(bencher: Bencher) {
    bench_copy::<i64>(bencher, 18, DecimalDType::new(38, 2));
}

#[vortex_bench_support::cpu_features]
#[divan::bench]
fn i128_to_i256(bencher: Bencher) {
    bench_copy::<i128>(bencher, 18, DecimalDType::new(76, 2));
}

#[vortex_bench_support::cpu_features]
#[divan::bench]
fn same_width_i64(bencher: Bencher) {
    bench_copy::<i64>(bencher, 18, DecimalDType::new(18, 2));
}
