// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use rand::RngExt;
use rand::SeedableRng as _;
use rand::rngs::StdRng;
use vortex_alp::ALPFloat;
use vortex_alp::ALPRDFloat;
use vortex_alp::RDEncoder;
use vortex_alp::alp_encode;
use vortex_alp::decompress_into_array;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::NativePType;
use vortex_array::session::ArraySession;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::buffer;
use vortex_session::VortexSession;

fn main() {
    divan::main();
}

const BENCH_ARGS: &[(usize, f64, f64)] = &[
    // length, fraction_patch, fraction_valid
    (1_000, 0.0, 0.25),
    (1_000, 0.01, 0.25),
    (1_000, 0.1, 0.25),
    (1_000, 0.0, 0.95),
    (1_000, 0.01, 0.95),
    (1_000, 0.1, 0.95),
    (1_000, 0.0, 1.0),
    (1_000, 0.01, 1.0),
    (1_000, 0.1, 1.0),
    (10_000, 0.0, 0.25),
    (10_000, 0.01, 0.25),
    (10_000, 0.1, 0.25),
    (10_000, 0.0, 0.95),
    (10_000, 0.01, 0.95),
    (10_000, 0.1, 0.95),
    (10_000, 0.0, 1.0),
    (10_000, 0.01, 1.0),
    (10_000, 0.1, 1.0),
];

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

#[divan::bench(types = [f32, f64], args = BENCH_ARGS)]
fn compress_alp<T: ALPFloat + NativePType>(bencher: Bencher, args: (usize, f64, f64)) {
    let (n, fraction_patch, fraction_valid) = args;
    let mut rng = StdRng::seed_from_u64(0);
    let mut values = buffer![T::from(1.234).unwrap(); n].into_mut();
    if fraction_patch > 0.0 {
        for index in 0..values.len() {
            if rng.random_bool(fraction_patch) {
                values[index] = T::from(1000.0).unwrap()
            }
        }
    }
    let validity = if fraction_valid < 1.0 {
        Validity::from_iter((0..values.len()).map(|_| rng.random_bool(fraction_valid)))
    } else {
        Validity::NonNullable
    };
    let values = values.freeze();
    let array = PrimitiveArray::new(values, validity);

    bencher
        .with_inputs(|| (&array, SESSION.create_execution_ctx()))
        .bench_values(|(array, mut ctx)| alp_encode(array.as_view(), None, &mut ctx).unwrap())
}

#[divan::bench(types = [f32, f64], args = BENCH_ARGS)]
fn decompress_alp<T: ALPFloat + NativePType>(bencher: Bencher, args: (usize, f64, f64)) {
    let (n, fraction_patch, fraction_valid) = args;
    let mut rng = StdRng::seed_from_u64(0);
    let mut values = buffer![T::from(1.234).unwrap(); n].into_mut();
    if fraction_patch > 0.0 {
        for index in 0..values.len() {
            if rng.random_bool(fraction_patch) {
                values[index] = T::from(1000.0).unwrap()
            }
        }
    }
    let validity = if fraction_valid < 1.0 {
        Validity::from_iter((0..values.len()).map(|_| rng.random_bool(fraction_valid)))
    } else {
        Validity::NonNullable
    };
    let values = values.freeze();
    bencher
        .with_inputs(|| {
            (
                alp_encode(
                    PrimitiveArray::new(Buffer::copy_from(&values), validity.clone()).as_view(),
                    None,
                    &mut SESSION.create_execution_ctx(),
                )
                .unwrap(),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_values(|(v, mut ctx)| decompress_into_array(v, &mut ctx));
}

const RD_BENCH_ARGS: &[(usize, f64)] = &[
    // length, fraction_patch
    (10_000, 0.0),
    (10_000, 0.01),
    (10_000, 0.1),
    (100_000, 0.0),
    (100_000, 0.01),
    (100_000, 0.1),
];

fn make_rd_array<T: ALPRDFloat + NativePType>(n: usize, fraction_patch: f64) -> PrimitiveArray {
    let base_val = T::from(1.23).unwrap();
    let mut rng = StdRng::seed_from_u64(42);
    let mut values = buffer![base_val; n].into_mut();
    if fraction_patch > 0.0 {
        let outlier = T::from(1000.0).unwrap();
        for index in 0..values.len() {
            if rng.random_bool(fraction_patch) {
                values[index] = outlier;
            }
        }
    }
    PrimitiveArray::new(values.freeze(), Validity::NonNullable)
}

#[divan::bench(types = [f32, f64], args = RD_BENCH_ARGS)]
fn compress_rd<T: ALPRDFloat + NativePType>(bencher: Bencher, args: (usize, f64)) {
    let (n, fraction_patch) = args;
    let primitive = make_rd_array::<T>(n, fraction_patch);
    let encoder = RDEncoder::new(primitive.as_slice::<T>());

    bencher
        .with_inputs(|| (&primitive, &encoder, SESSION.create_execution_ctx()))
        .bench_refs(|(primitive, encoder, ctx)| encoder.encode(primitive.as_view(), ctx))
}

#[divan::bench(types = [f32, f64], args = RD_BENCH_ARGS)]
fn decompress_rd<T: ALPRDFloat + NativePType>(bencher: Bencher, args: (usize, f64)) {
    let (n, fraction_patch) = args;
    let primitive = make_rd_array::<T>(n, fraction_patch);
    let encoder = RDEncoder::new(primitive.as_slice::<T>());
    let encoded = encoder.encode(primitive.as_view(), &mut SESSION.create_execution_ctx());

    bencher
        .with_inputs(|| (&encoded, SESSION.create_execution_ctx()))
        .bench_refs(|(encoded, ctx)| (**encoded).clone().into_array().execute::<Canonical>(ctx));
}
