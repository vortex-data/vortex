// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks for primitive take and [`DictArray`] canonicalization.

#![expect(clippy::cast_possible_truncation)]
#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use divan::counter::ItemsCount;
use rand::distr::Uniform;
use rand::prelude::*;
use rand_distr::Zipf;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_session::VortexSession;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

/// Number of indices to take. The top tier is sized to keep CodSpeed simulation under 1ms.
const NUM_INDICES: &[usize] = &[1_000, 10_000, 25_000];

/// Large enough to measure both cache-resident and streaming dictionary decoding.
const GT_NUM_INDICES: &[usize] = &[1_000_000, 16_000_000];

/// Size of the source vector / dictionary values.
const VECTOR_SIZE: &[usize] = &[16, 256, 2048, 8192];

static SESSION: LazyLock<VortexSession> = LazyLock::new(array_session);

#[divan::bench(sample_count = 200)]
fn primitive_take_u32(bencher: Bencher) {
    // Sized to keep CodSpeed simulation under 1ms.
    const NUM_INDICES: usize = 50_000;

    let values = PrimitiveArray::from_iter(0u32..256).into_array();
    let indices =
        PrimitiveArray::from_iter((0..NUM_INDICES).map(|index| index as u32 & 255)).into_array();

    bencher
        .counter(ItemsCount::new(NUM_INDICES))
        .with_inputs(|| (indices.clone(), SESSION.create_execution_ctx()))
        .bench_values(|(indices, mut ctx)| {
            values
                .take(indices)
                .unwrap()
                .execute::<PrimitiveArray>(&mut ctx)
                .unwrap()
        });
}

// --- DictArray canonicalization benchmarks ---

#[divan::bench(args = NUM_INDICES, consts = VECTOR_SIZE, sample_count = 100_000)]
fn dict_canonicalize_uniform<const NUM_VALUES: usize>(bencher: Bencher, num_indices: usize) {
    let values = PrimitiveArray::from_iter(0..NUM_VALUES as u32);

    let rng = StdRng::seed_from_u64(0);
    let range = Uniform::new(0u32, NUM_VALUES as u32).unwrap();
    let codes = PrimitiveArray::from_iter(rng.sample_iter(range).take(num_indices));

    let dict = DictArray::try_new(codes.into_array(), values.into_array()).unwrap();

    bencher
        .with_inputs(|| (&dict, SESSION.create_execution_ctx()))
        .bench_refs(|(dict, ctx)| (*dict).clone().into_array().execute::<Canonical>(ctx));
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

    bencher
        .with_inputs(|| (&dict, SESSION.create_execution_ctx()))
        .bench_refs(|(dict, ctx)| (*dict).clone().into_array().execute::<Canonical>(ctx));
}

/// StatPopGen GT distribution: 75.21% 0, 3.85% 1, 0.45% 2, and 20.49% null.
#[vortex_bench_support::cpu_features]
#[divan::bench(args = GT_NUM_INDICES)]
fn dict_canonicalize_gt_u8(bencher: Bencher, num_indices: usize) {
    let values = PrimitiveArray::from_option_iter([Some(0u8), Some(1), Some(2), None]);
    let range = Uniform::new(0u16, 10_000).unwrap();
    let codes = PrimitiveArray::from_iter(
        StdRng::seed_from_u64(0)
            .sample_iter(range)
            .take(num_indices)
            .map(|sample| match sample {
                0..7521 => 0u8,
                7521..7906 => 1,
                7906..7951 => 2,
                _ => 3,
            }),
    );
    let dict = DictArray::try_new(codes.into_array(), values.into_array()).unwrap();

    bencher
        .counter(ItemsCount::new(num_indices))
        .with_inputs(|| (&dict, SESSION.create_execution_ctx()))
        .bench_refs(|(dict, ctx)| (*dict).clone().into_array().execute::<Canonical>(ctx));
}
