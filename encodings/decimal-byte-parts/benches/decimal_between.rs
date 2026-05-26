// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compares `between` (`lower <= x <= upper`) over decimal data for:
//!   * Vortex `DecimalByteParts` with an i32 most-significant-part (pushdown kernel),
//!   * Vortex `DecimalByteParts` with an i64 most-significant-part (pushdown kernel),
//!   * Vortex canonical `DecimalArray` (i128 storage),
//!   * arrow-rs `Decimal128Array` via `gt_eq` + `lt_eq` + `and`.
//!
//! arrow-rs has no decimal storage narrower than 128 bits, so logically-small decimals that
//! Vortex keeps in an i32/i64 MSP must be materialised as i128 in Arrow. This benchmark
//! measures the resulting throughput difference.

use arrow_arith::boolean::and;
use arrow_array::Decimal128Array;
use arrow_array::Scalar as ArrowScalar;
use arrow_ord::cmp;
use divan::Bencher;
use divan::black_box;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::Nullability;
use vortex_array::scalar::DecimalValue;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::between::BetweenOptions;
use vortex_array::scalar_fn::fns::between::StrictComparison;
use vortex_array::validity::Validity;
use vortex_decimal_byte_parts::DecimalByteParts;

fn main() {
    divan::main();
}

const LENGTHS: &[usize] = &[1 << 16, 1 << 20];

// Logical decimal range [0, 1000), precision 9 scale 2 (fits i32) and precision 18 (fits i64).
const LOWER: i64 = 250;
const UPPER: i64 = 750;

const OPTIONS: BetweenOptions = BetweenOptions {
    lower_strict: StrictComparison::NonStrict,
    upper_strict: StrictComparison::NonStrict,
};

fn values(len: usize) -> Vec<i64> {
    let mut rng = StdRng::seed_from_u64(0x5eed);
    (0..len).map(|_| rng.random_range(0..1000i64)).collect()
}

// ---- Vortex DecimalByteParts (i32 MSP) ----

#[divan::bench(args = LENGTHS)]
fn vortex_byteparts_i32(bencher: Bencher, len: usize) {
    let dt = DecimalDType::new(9, 2);
    let msp = PrimitiveArray::from_iter(values(len).into_iter().map(|v| v as i32)).into_array();
    let arr = DecimalByteParts::try_new(msp, dt).unwrap().into_array();
    let lower = ConstantArray::new(
        Scalar::decimal(
            DecimalValue::I32(LOWER as i32),
            dt,
            Nullability::NonNullable,
        ),
        len,
    )
    .into_array();
    let upper = ConstantArray::new(
        Scalar::decimal(
            DecimalValue::I32(UPPER as i32),
            dt,
            Nullability::NonNullable,
        ),
        len,
    )
    .into_array();

    bencher
        .with_inputs(|| {
            (
                arr.clone(),
                lower.clone(),
                upper.clone(),
                LEGACY_SESSION.create_execution_ctx(),
            )
        })
        .bench_values(|(arr, lower, upper, mut ctx)| {
            black_box(
                arr.between(lower, upper, OPTIONS)
                    .unwrap()
                    .execute::<BoolArray>(&mut ctx)
                    .unwrap(),
            )
        });
}

// ---- Vortex DecimalByteParts (i64 MSP) ----

#[divan::bench(args = LENGTHS)]
fn vortex_byteparts_i64(bencher: Bencher, len: usize) {
    let dt = DecimalDType::new(18, 2);
    let msp = PrimitiveArray::from_iter(values(len)).into_array();
    let arr = DecimalByteParts::try_new(msp, dt).unwrap().into_array();
    let lower = ConstantArray::new(
        Scalar::decimal(DecimalValue::I64(LOWER), dt, Nullability::NonNullable),
        len,
    )
    .into_array();
    let upper = ConstantArray::new(
        Scalar::decimal(DecimalValue::I64(UPPER), dt, Nullability::NonNullable),
        len,
    )
    .into_array();

    bencher
        .with_inputs(|| {
            (
                arr.clone(),
                lower.clone(),
                upper.clone(),
                LEGACY_SESSION.create_execution_ctx(),
            )
        })
        .bench_values(|(arr, lower, upper, mut ctx)| {
            black_box(
                arr.between(lower, upper, OPTIONS)
                    .unwrap()
                    .execute::<BoolArray>(&mut ctx)
                    .unwrap(),
            )
        });
}

// ---- Vortex canonical DecimalArray (i128 storage) ----

#[divan::bench(args = LENGTHS)]
fn vortex_canonical_i128(bencher: Bencher, len: usize) {
    let dt = DecimalDType::new(9, 2);
    let arr = DecimalArray::new(
        values(len).into_iter().map(i128::from).collect(),
        dt,
        Validity::NonNullable,
    )
    .into_array();
    let lower = ConstantArray::new(
        Scalar::decimal(
            DecimalValue::I128(LOWER as i128),
            dt,
            Nullability::NonNullable,
        ),
        len,
    )
    .into_array();
    let upper = ConstantArray::new(
        Scalar::decimal(
            DecimalValue::I128(UPPER as i128),
            dt,
            Nullability::NonNullable,
        ),
        len,
    )
    .into_array();

    bencher
        .with_inputs(|| {
            (
                arr.clone(),
                lower.clone(),
                upper.clone(),
                LEGACY_SESSION.create_execution_ctx(),
            )
        })
        .bench_values(|(arr, lower, upper, mut ctx)| {
            black_box(
                arr.between(lower, upper, OPTIONS)
                    .unwrap()
                    .execute::<BoolArray>(&mut ctx)
                    .unwrap(),
            )
        });
}

// ---- arrow-rs Decimal128 (gt_eq + lt_eq + and) ----

#[divan::bench(args = LENGTHS)]
fn arrow_decimal128(bencher: Bencher, len: usize) {
    let arr = Decimal128Array::from_iter_values(values(len).into_iter().map(i128::from))
        .with_precision_and_scale(9, 2)
        .unwrap();
    let lower = ArrowScalar::new(
        Decimal128Array::from_iter_values([LOWER as i128])
            .with_precision_and_scale(9, 2)
            .unwrap(),
    );
    let upper = ArrowScalar::new(
        Decimal128Array::from_iter_values([UPPER as i128])
            .with_precision_and_scale(9, 2)
            .unwrap(),
    );

    bencher
        .with_inputs(|| (arr.clone(), lower.clone(), upper.clone()))
        .bench_values(|(arr, lower, upper)| {
            let ge = cmp::gt_eq(&arr, &lower).unwrap();
            let le = cmp::lt_eq(&arr, &upper).unwrap();
            black_box(and(&ge, &le).unwrap())
        });
}
