// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use divan::Bencher;
use rand::prelude::*;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::expr::stats::Stat;

fn main() {
    divan::main();
}

const N: usize = 100_000;

#[divan::bench]
fn cast_u16_to_u32(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(42);
    #[expect(clippy::cast_possible_truncation)]
    let arr = PrimitiveArray::from_option_iter((0..N).map(|i| {
        if rng.random_bool(0.5) {
            None
        } else {
            Some(i as u16)
        }
    }))
    .into_array();
    // Pre-compute min/max so values_fit_in is a cache hit during the benchmark.
    arr.statistics()
        .compute_all(
            &[Stat::Min, Stat::Max],
            &mut LEGACY_SESSION.create_execution_ctx(),
        )
        .ok();
    bencher.with_inputs(|| arr.clone()).bench_refs(|a| {
        #[expect(clippy::unwrap_used)]
        a.cast(DType::Primitive(PType::U32, Nullability::Nullable))
            .unwrap()
            .execute::<Canonical>(&mut LEGACY_SESSION.create_execution_ctx())
    });
}

/// Narrowing cast on nullable data with *no* precomputed stats: exercises the fused fallible
/// kernel (validity-aware min/max + cast in one pass).
#[divan::bench]
fn cast_u32_to_u8_nullable(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(42);
    let arr = PrimitiveArray::from_option_iter((0..N).map(|_| {
        if rng.random_bool(0.5) {
            None
        } else {
            Some(rng.random_range(0u32..=255))
        }
    }))
    .into_array();
    bencher.with_inputs(|| arr.clone()).bench_refs(|a| {
        #[expect(clippy::unwrap_used)]
        a.cast(DType::Primitive(PType::U8, Nullability::Nullable))
            .unwrap()
            .execute::<Canonical>(&mut LEGACY_SESSION.create_execution_ctx())
    });
}

/// Float-to-int narrowing on nullable data with no precomputed stats.
#[divan::bench]
fn cast_f64_to_i32_nullable(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(42);
    let arr = PrimitiveArray::from_option_iter((0..N).map(|_| {
        if rng.random_bool(0.5) {
            None
        } else {
            Some(rng.random_range(-1000.0f64..=1000.0))
        }
    }))
    .into_array();
    bencher.with_inputs(|| arr.clone()).bench_refs(|a| {
        #[expect(clippy::unwrap_used)]
        a.cast(DType::Primitive(PType::I32, Nullability::Nullable))
            .unwrap()
            .execute::<Canonical>(&mut LEGACY_SESSION.create_execution_ctx())
    });
}

/// Float-to-int narrowing on dense (all-valid) data with no precomputed stats.
#[divan::bench]
fn cast_f64_to_i32_dense(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(42);
    let arr = PrimitiveArray::from_iter((0..N).map(|_| rng.random_range(-1000.0f64..=1000.0)))
        .into_array();
    bencher.with_inputs(|| arr.clone()).bench_refs(|a| {
        #[expect(clippy::unwrap_used)]
        a.cast(DType::Primitive(PType::I32, Nullability::Nullable))
            .unwrap()
            .execute::<Canonical>(&mut LEGACY_SESSION.create_execution_ctx())
    });
}
