// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]

use divan::Bencher;
use mimalloc::MiMalloc;
use rand::RngExt;
use rand::SeedableRng;
use rand::distr::Uniform;
use rand::prelude::StdRng;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DecimalDType;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    divan::main();
}

const ARRAY_SIZE: usize = 65_536;

fn bench_compare(bencher: Bencher, lhs: ArrayRef, rhs: ArrayRef, op: Operator) {
    let session = vortex_array::array_session();
    bencher
        .with_inputs(|| (&lhs, &rhs, session.create_execution_ctx()))
        .bench_refs(|input| {
            input
                .0
                .clone()
                .binary(input.1.clone(), op)
                .unwrap()
                .execute::<Canonical>(&mut input.2)
        });
}

fn bool_array(rng: &mut StdRng) -> ArrayRef {
    BoolArray::from_iter((0..ARRAY_SIZE).map(|_| rng.random_bool(0.5))).into_array()
}

fn bool_array_nullable(rng: &mut StdRng) -> ArrayRef {
    BoolArray::new(
        (0..ARRAY_SIZE).map(|_| rng.random_bool(0.5)).collect(),
        Validity::from_iter((0..ARRAY_SIZE).map(|_| rng.random_bool(0.9))),
    )
    .into_array()
}

fn int_array(rng: &mut StdRng) -> ArrayRef {
    let range = Uniform::new(0i64, 100_000_000).unwrap();
    (0..ARRAY_SIZE)
        .map(|_| rng.sample(range))
        .collect::<Buffer<_>>()
        .into_array()
}

fn int_array_nullable(rng: &mut StdRng) -> ArrayRef {
    let range = Uniform::new(0i64, 100_000_000).unwrap();
    PrimitiveArray::new(
        (0..ARRAY_SIZE)
            .map(|_| rng.sample(range))
            .collect::<Buffer<_>>(),
        Validity::from_iter((0..ARRAY_SIZE).map(|_| rng.random_bool(0.9))),
    )
    .into_array()
}

fn float_array(rng: &mut StdRng) -> ArrayRef {
    (0..ARRAY_SIZE)
        .map(|_| rng.random_range(0.0f64..1.0))
        .collect::<Buffer<_>>()
        .into_array()
}

fn string_array(rng: &mut StdRng) -> ArrayRef {
    VarBinViewArray::from_iter_str((0..ARRAY_SIZE).map(|_| {
        let len = rng.random_range(1usize..24);
        (0..len)
            .map(|_| char::from(rng.random_range(b'a'..=b'z')))
            .collect::<String>()
    }))
    .into_array()
}

fn decimal_array(rng: &mut StdRng) -> ArrayRef {
    DecimalArray::from_iter::<i128, _>(
        (0..ARRAY_SIZE).map(|_| rng.random_range(0i128..100_000_000)),
        DecimalDType::new(38, 2),
    )
    .into_array()
}

#[divan::bench]
fn compare_bool(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(0);
    let arr1 = bool_array(&mut rng);
    let arr2 = bool_array(&mut rng);
    bench_compare(bencher, arr1, arr2, Operator::Gte);
}

#[divan::bench]
fn compare_bool_nullable(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(0);
    let arr1 = bool_array_nullable(&mut rng);
    let arr2 = bool_array_nullable(&mut rng);
    bench_compare(bencher, arr1, arr2, Operator::Gte);
}

#[divan::bench]
fn compare_bool_constant(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(0);
    let arr = bool_array(&mut rng);
    let constant = ConstantArray::new(true, ARRAY_SIZE).into_array();
    bench_compare(bencher, arr, constant, Operator::Eq);
}

#[divan::bench]
fn compare_int(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(0);
    let arr1 = int_array(&mut rng);
    let arr2 = int_array(&mut rng);
    bench_compare(bencher, arr1, arr2, Operator::Gte);
}

#[divan::bench]
fn compare_int_nullable(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(0);
    let arr1 = int_array_nullable(&mut rng);
    let arr2 = int_array_nullable(&mut rng);
    bench_compare(bencher, arr1, arr2, Operator::Gte);
}

#[divan::bench]
fn compare_int_constant(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(0);
    let arr = int_array(&mut rng);
    let constant = ConstantArray::new(50_000_000i64, ARRAY_SIZE).into_array();
    bench_compare(bencher, arr, constant, Operator::Gte);
}

#[divan::bench]
fn compare_int_eq(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(0);
    let arr1 = int_array(&mut rng);
    let arr2 = int_array(&mut rng);
    bench_compare(bencher, arr1, arr2, Operator::Eq);
}

#[divan::bench]
fn compare_float(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(0);
    let arr1 = float_array(&mut rng);
    let arr2 = float_array(&mut rng);
    bench_compare(bencher, arr1, arr2, Operator::Gte);
}

#[divan::bench]
fn compare_decimal(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(0);
    let arr1 = decimal_array(&mut rng);
    let arr2 = decimal_array(&mut rng);
    bench_compare(bencher, arr1, arr2, Operator::Gte);
}

#[divan::bench]
fn compare_string_eq(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(0);
    let arr1 = string_array(&mut rng);
    let arr2 = string_array(&mut rng);
    bench_compare(bencher, arr1, arr2, Operator::Eq);
}

#[divan::bench]
fn compare_string_lt(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(0);
    let arr1 = string_array(&mut rng);
    let arr2 = string_array(&mut rng);
    bench_compare(bencher, arr1, arr2, Operator::Lt);
}

#[divan::bench]
fn compare_string_constant(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(0);
    let arr = string_array(&mut rng);
    let constant = ConstantArray::new(Scalar::from("mmmmmmmmmmmm"), ARRAY_SIZE).into_array();
    bench_compare(bencher, arr, constant, Operator::Lt);
}

fn struct_array(rng: &mut StdRng) -> ArrayRef {
    StructArray::from_fields(&[("a", int_array(rng)), ("b", int_array(rng))])
        .unwrap()
        .into_array()
}

#[divan::bench]
fn compare_struct_lt(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(0);
    let arr1 = struct_array(&mut rng);
    let arr2 = struct_array(&mut rng);
    bench_compare(bencher, arr1, arr2, Operator::Lt);
}

#[divan::bench]
fn compare_struct_eq(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(0);
    let arr1 = struct_array(&mut rng);
    let arr2 = struct_array(&mut rng);
    bench_compare(bencher, arr1, arr2, Operator::Eq);
}
