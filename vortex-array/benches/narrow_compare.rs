// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compare ordinary primitive buffers across native and NarrowArray storage widths.

#![expect(
    clippy::unwrap_used,
    reason = "benchmark inputs satisfy the encoding invariants"
)]

use std::hint::black_box;
use std::time::Duration;
use std::time::Instant;

use divan::Bencher;
use divan::counter::ItemsCount;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::NarrowArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PType;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::operators::Operator;

const CASES: &[(usize, Operator)] = &[
    (8_192, Operator::Lt),
    (1_048_576, Operator::Lt),
    (16_777_216, Operator::Lt),
    (8_192, Operator::Eq),
    (1_048_576, Operator::Eq),
    (16_777_216, Operator::Eq),
];

fn main() {
    if std::env::var_os("VORTEX_NARROW_INTERLEAVED").is_some() {
        interleaved();
    } else {
        divan::main();
    }
}

fn input<T: NativePType + From<i8>>(len: usize, offset: usize) -> ArrayRef {
    PrimitiveArray::from_iter((0..len).map(|i| {
        let value = i
            .wrapping_add(offset)
            .wrapping_mul(2_654_435_761)
            .rotate_right(13)
            & 31;
        <T as From<i8>>::from(i8::try_from(value).unwrap())
    }))
    .into_array()
}

fn narrow(values: ArrayRef) -> ArrayRef {
    NarrowArray::try_new(values, PType::I64.into())
        .unwrap()
        .into_array()
}

fn bench_compare(bencher: Bencher, lhs: ArrayRef, rhs: ArrayRef, operator: Operator) {
    let session = array_session();
    bencher
        .counter(ItemsCount::new(lhs.len()))
        .with_inputs(|| session.create_execution_ctx())
        .bench_refs(|ctx| {
            lhs.binary(rhs.clone(), operator)
                .unwrap()
                .execute::<BoolArray>(ctx)
                .unwrap()
        });
}

fn bench_constant(bencher: Bencher, lhs: ArrayRef, operator: Operator) {
    let rhs =
        ConstantArray::new(Scalar::from(16i64).cast(lhs.dtype()).unwrap(), lhs.len()).into_array();
    bench_compare(bencher, lhs, rhs, operator);
}

#[divan::bench(types = [i64, i32, i16, i8], args = CASES)]
fn constant_primitive<T: NativePType + From<i8>>(
    bencher: Bencher,
    (len, operator): (usize, Operator),
) {
    bench_constant(bencher, input::<T>(len, 0), operator);
}

#[divan::bench(types = [i32, i16, i8], args = CASES)]
fn constant_narrow<T: NativePType + From<i8>>(
    bencher: Bencher,
    (len, operator): (usize, Operator),
) {
    bench_constant(bencher, narrow(input::<T>(len, 0)), operator);
}

#[divan::bench(types = [i64, i32, i16, i8], args = CASES)]
fn pair_primitive<T: NativePType + From<i8>>(bencher: Bencher, (len, operator): (usize, Operator)) {
    bench_compare(bencher, input::<T>(len, 0), input::<T>(len, 17), operator);
}

#[divan::bench(types = [i32, i16, i8], args = CASES)]
fn pair_narrow<T: NativePType + From<i8>>(bencher: Bencher, (len, operator): (usize, Operator)) {
    bench_compare(
        bencher,
        narrow(input::<T>(len, 0)),
        narrow(input::<T>(len, 17)),
        operator,
    );
}

fn interleaved() {
    const SAMPLES: usize = 140;
    println!("rows,shape,operator,representation,sample,nanoseconds");
    let session = array_session();
    for len in [8_192, 1_048_576, 16_777_216] {
        let inputs = [
            ("i64", input::<i64>(len, 0), input::<i64>(len, 17)),
            ("i32", input::<i32>(len, 0), input::<i32>(len, 17)),
            ("i16", input::<i16>(len, 0), input::<i16>(len, 17)),
            ("i8", input::<i8>(len, 0), input::<i8>(len, 17)),
        ];
        for shape in ["constant", "pair"] {
            let mut cases = Vec::new();
            for (width, lhs, rhs) in &inputs {
                let rhs = if shape == "constant" {
                    ConstantArray::new(Scalar::from(16i64).cast(lhs.dtype()).unwrap(), len)
                        .into_array()
                } else {
                    rhs.clone()
                };
                cases.push((format!("primitive_{width}"), lhs.clone(), rhs.clone()));
                if *width != "i64" {
                    let rhs = if shape == "constant" {
                        ConstantArray::new(16i64, len).into_array()
                    } else {
                        narrow(rhs)
                    };
                    cases.push((format!("narrow_{width}"), narrow(lhs.clone()), rhs));
                }
            }
            for operator in [Operator::Lt, Operator::Eq] {
                let mut expected = None;
                for (_, lhs, rhs) in &cases {
                    let bits = lhs
                        .binary(rhs.clone(), operator)
                        .unwrap()
                        .execute::<BoolArray>(&mut session.create_execution_ctx())
                        .unwrap()
                        .into_bit_buffer();
                    if let Some(expected) = &expected {
                        assert_eq!(&bits, expected);
                    } else {
                        expected = Some(bits);
                    }
                }
                let mut samples = (0..cases.len())
                    .map(|_| Vec::<Duration>::with_capacity(SAMPLES))
                    .collect::<Vec<_>>();
                for sample in 0..SAMPLES {
                    // Rotate both the first case and direction to distribute timing/cache drift.
                    let start = sample % cases.len();
                    let reverse = !(sample / cases.len()).is_multiple_of(2);
                    for step in 0..cases.len() {
                        let offset = if reverse { cases.len() - step } else { step };
                        let index = (start + offset) % cases.len();
                        let (_, lhs, rhs) = &cases[index];
                        let mut ctx = session.create_execution_ctx();
                        let started = Instant::now();
                        let result = lhs
                            .binary(rhs.clone(), operator)
                            .unwrap()
                            .execute::<BoolArray>(&mut ctx)
                            .unwrap();
                        let elapsed = started.elapsed();
                        black_box(result);
                        samples[index].push(elapsed);
                    }
                }
                for ((representation, ..), durations) in cases.iter().zip(samples) {
                    for (sample, duration) in durations.into_iter().enumerate() {
                        println!(
                            "{len},{shape},{operator},{representation},{sample},{}",
                            duration.as_nanos(),
                        );
                    }
                }
            }
        }
    }
}
