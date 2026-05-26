// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compares `x < threshold` (less-than against a constant) over decimal data for:
//!   * Vortex `DecimalByteParts` with an i32 most-significant-part (pushdown kernel),
//!   * Vortex `DecimalByteParts` with an i64 most-significant-part (pushdown kernel),
//!   * Vortex `DecimalByteParts` two-limb i128 (signed-high / unsigned-low limbs, fused kernel),
//!   * Vortex canonical `DecimalArray` (i128 storage),
//!   * arrow-rs `Decimal128Array` via `cmp::lt`.
//!
//! Unlike `between`, arrow evaluates `lt` in a single pass, so this isolates a one-sided
//! comparison. arrow-rs has no decimal storage narrower than 128 bits, so logically-small or
//! limb-split decimals that Vortex keeps narrower must be materialised as i128 in arrow.

#![allow(clippy::unwrap_used, clippy::cast_possible_truncation)]

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
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::validity::Validity;
use vortex_decimal_byte_parts::DecimalByteParts;

fn main() {
    divan::main();
}

const LENGTHS: &[usize] = &[1 << 16, 1 << 17];

// Logical decimal range [0, 1000); threshold in the middle so ~half the rows pass.
const THRESHOLD: i64 = 500;

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
    let rhs = ConstantArray::new(
        Scalar::decimal(
            DecimalValue::I32(THRESHOLD as i32),
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
                rhs.clone(),
                LEGACY_SESSION.create_execution_ctx(),
            )
        })
        .bench_values(|(arr, rhs, mut ctx)| {
            black_box(
                arr.binary(rhs, Operator::Lt)
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
    let rhs = ConstantArray::new(
        Scalar::decimal(DecimalValue::I64(THRESHOLD), dt, Nullability::NonNullable),
        len,
    )
    .into_array();

    bencher
        .with_inputs(|| {
            (
                arr.clone(),
                rhs.clone(),
                LEGACY_SESSION.create_execution_ctx(),
            )
        })
        .bench_values(|(arr, rhs, mut ctx)| {
            black_box(
                arr.binary(rhs, Operator::Lt)
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
    let rhs = ConstantArray::new(
        Scalar::decimal(
            DecimalValue::I128(THRESHOLD as i128),
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
                rhs.clone(),
                LEGACY_SESSION.create_execution_ctx(),
            )
        })
        .bench_values(|(arr, rhs, mut ctx)| {
            black_box(
                arr.binary(rhs, Operator::Lt)
                    .unwrap()
                    .execute::<BoolArray>(&mut ctx)
                    .unwrap(),
            )
        });
}

// ---- arrow-rs Decimal128 (cmp::lt) ----

#[divan::bench(args = LENGTHS)]
fn arrow_decimal128(bencher: Bencher, len: usize) {
    let arr = Decimal128Array::from_iter_values(values(len).into_iter().map(i128::from))
        .with_precision_and_scale(9, 2)
        .unwrap();
    let rhs = ArrowScalar::new(
        Decimal128Array::from_iter_values([THRESHOLD as i128])
            .with_precision_and_scale(9, 2)
            .unwrap(),
    );

    bencher
        .with_inputs(|| (arr.clone(), rhs.clone()))
        .bench_values(|(arr, rhs)| black_box(cmp::lt(&arr, &rhs).unwrap()));
}

// ---- Wide i128 decimals: two-limb ----
//
// These values genuinely occupy the i128 range (the high 64-bit limb varies), so neither Vortex
// nor arrow can keep them in a narrow integer. The two-limb representation splits each value into a
// signed i64 high limb and an unsigned u64 low limb, compared limb-wise (AVX-512 when available).
//
// The i128 baselines for this comparison are `arrow_decimal128` and `vortex_canonical_i128` above:
// an i128 comparison's cost is independent of the values and the declared precision/scale, so those
// benches measure the same kernel regardless of whether the data is logically narrow or wide.

fn wide_values(len: usize) -> Vec<i128> {
    let mut rng = StdRng::seed_from_u64(0x5eed);
    (0..len)
        .map(|_| {
            let high = i128::from(rng.random_range(0..1000i64));
            let low = i128::from(rng.random_range(0..u64::MAX));
            (high << 64) | low
        })
        .collect()
}

// Threshold with a non-zero low limb so the low-limb tie-break is exercised at the high-limb edge.
const WIDE_THRESHOLD: i128 = (500i128 << 64) | 0x90ab_cdef;

#[divan::bench(args = LENGTHS)]
fn vortex_byteparts_twolimb(bencher: Bencher, len: usize) {
    let dt = DecimalDType::new(38, 2);
    let values = wide_values(len);
    let highs = PrimitiveArray::from_iter(values.iter().map(|v| (v >> 64) as i64)).into_array();
    let lows = PrimitiveArray::from_iter(values.iter().map(|v| *v as u64)).into_array();
    let arr = DecimalByteParts::try_new_with_lower(highs, lows, dt)
        .unwrap()
        .into_array();
    let rhs = ConstantArray::new(
        Scalar::decimal(
            DecimalValue::I128(WIDE_THRESHOLD),
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
                rhs.clone(),
                LEGACY_SESSION.create_execution_ctx(),
            )
        })
        .bench_values(|(arr, rhs, mut ctx)| {
            black_box(
                arr.binary(rhs, Operator::Lt)
                    .unwrap()
                    .execute::<BoolArray>(&mut ctx)
                    .unwrap(),
            )
        });
}
