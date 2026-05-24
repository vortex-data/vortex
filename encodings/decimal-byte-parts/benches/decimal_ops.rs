// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Baseline micro-benchmarks for decimal value-wise operations.
//!
//! Compares three "engines" for compare / numeric / aggregation operations across three decimal
//! widths (fits-in-i64, i128, i256):
//!
//! - `canonical`: Vortex canonical [`DecimalArray`] (compare/numeric delegate to Arrow, aggregations
//!   are hand-written scalar loops).
//! - `byteparts`: Vortex [`DecimalByteParts`] with the values split into 64-bit parts. Today every
//!   value-wise op canonicalizes first, so this measures "canonicalize + canonical".
//! - `arrow`: Arrow `Decimal128Array` / `Decimal256Array` with the Arrow compute kernels directly.
//!
//! This establishes where the real cost is (especially the i128/i256 penalty in Arrow's wide-integer
//! kernels) before we add native SIMD kernels on the byte-parts representation.

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)]

use arrow_array::Decimal128Array;
use arrow_array::Decimal256Array;
use divan::Bencher;
use divan::black_box;
use rand::SeedableRng;
use rand::distr::Distribution;
use rand::distr::Uniform;
use rand::prelude::StdRng;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::fns::min_max::min_max;
use vortex_array::aggregate_fn::fns::sum::sum;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::i256;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_decimal_byte_parts::DecimalByteParts;
use vortex_session::VortexSession;

fn main() {
    divan::main();
}

const ARRAY_SIZE: usize = 65_536;

fn session() -> VortexSession {
    use vortex_array::session::ArraySession;
    let session = VortexSession::empty().with::<ArraySession>();
    vortex_decimal_byte_parts::initialize(&session);
    session
}

/// The three decimal widths under test.
#[derive(Clone, Copy, Debug)]
enum Width {
    /// Values fit in a signed 64-bit integer (single byte-parts column).
    Small,
    /// Values need 128 bits (byte-parts: msp + 1 lower column).
    I128,
    /// Values need 256 bits (byte-parts: msp + 3 lower columns).
    I256,
}

const WIDTHS: [Width; 3] = [Width::Small, Width::I128, Width::I256];

impl Width {
    fn dtype(self) -> DecimalDType {
        match self {
            Width::Small => DecimalDType::new(18, 2),
            Width::I128 => DecimalDType::new(38, 2),
            Width::I256 => DecimalDType::new(60, 2),
        }
    }
}

// ---- value generation -----------------------------------------------------------------------

fn vals_i64(seed: u64) -> Vec<i64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let range = Uniform::new(-100_000_000_000_000i64, 100_000_000_000_000).unwrap();
    (0..ARRAY_SIZE).map(|_| range.sample(&mut rng)).collect()
}

fn vals_i128(seed: u64) -> Vec<i128> {
    let mut rng = StdRng::seed_from_u64(seed);
    let hi = Uniform::new(1i128, 1_000_000_000_000_000i128).unwrap();
    let lo = Uniform::new(0i128, 1_000_000_000_000_000i128).unwrap();
    (0..ARRAY_SIZE)
        .map(|_| hi.sample(&mut rng) * 1_000_000_000_000_000_000_000i128 + lo.sample(&mut rng))
        .collect()
}

fn vals_i256(seed: u64) -> Vec<i256> {
    let mut rng = StdRng::seed_from_u64(seed);
    let upper = Uniform::new(1i128, 1_000_000_000_000_000i128).unwrap();
    let lo = Uniform::new(0u64, u64::MAX).unwrap();
    (0..ARRAY_SIZE)
        .map(|_| {
            let lower = (u128::from(lo.sample(&mut rng)) << 64) | u128::from(lo.sample(&mut rng));
            i256::from_parts(lower, upper.sample(&mut rng))
        })
        .collect()
}

// ---- Vortex array builders ------------------------------------------------------------------

fn canonical(width: Width, seed: u64) -> ArrayRef {
    let dtype = width.dtype();
    match width {
        Width::Small => DecimalArray::from_iter(vals_i64(seed), dtype).into_array(),
        Width::I128 => DecimalArray::from_iter(vals_i128(seed), dtype).into_array(),
        Width::I256 => DecimalArray::from_iter(vals_i256(seed), dtype).into_array(),
    }
}

fn byteparts(width: Width, seed: u64) -> ArrayRef {
    let dtype = width.dtype();
    match width {
        Width::Small => {
            let msp = PrimitiveArray::from_iter(vals_i64(seed)).into_array();
            DecimalByteParts::try_new(msp, dtype).unwrap().into_array()
        }
        Width::I128 => {
            let values = vals_i128(seed);
            let msp =
                PrimitiveArray::from_iter(values.iter().map(|v| (v >> 64) as i64)).into_array();
            let low = PrimitiveArray::from_iter(values.iter().map(|v| *v as u64)).into_array();
            DecimalByteParts::try_new_parts(msp, vec![low], dtype)
                .unwrap()
                .into_array()
        }
        Width::I256 => {
            let values = vals_i256(seed);
            let msp =
                PrimitiveArray::from_iter(values.iter().map(|v| (v.to_parts().1 >> 64) as i64))
                    .into_array();
            let p0 = PrimitiveArray::from_iter(values.iter().map(|v| v.to_parts().1 as u64))
                .into_array();
            let p1 =
                PrimitiveArray::from_iter(values.iter().map(|v| (v.to_parts().0 >> 64) as u64))
                    .into_array();
            let p2 = PrimitiveArray::from_iter(values.iter().map(|v| v.to_parts().0 as u64))
                .into_array();
            DecimalByteParts::try_new_parts(msp, vec![p0, p1, p2], dtype)
                .unwrap()
                .into_array()
        }
    }
}

// ---- fresh-array builders (defeat the statistics cache for aggregation benches) -------------
//
// `sum`/`min_max` memoize their result in the array's (Arc-shared) statistics cache, so cloning an
// `ArrayRef` across iterations would measure cache hits rather than compute. These return a cheap
// constructor that wraps pre-built buffers into a brand-new array (fresh stats) on every call.

fn fresh_canonical(width: Width, seed: u64) -> Box<dyn Fn() -> ArrayRef + Send + Sync> {
    let dtype = width.dtype();
    match width {
        Width::Small => {
            let buf: Buffer<i64> = vals_i64(seed).into_iter().collect();
            Box::new(move || {
                DecimalArray::new(buf.clone(), dtype, Validity::NonNullable).into_array()
            })
        }
        Width::I128 => {
            let buf: Buffer<i128> = vals_i128(seed).into_iter().collect();
            Box::new(move || {
                DecimalArray::new(buf.clone(), dtype, Validity::NonNullable).into_array()
            })
        }
        Width::I256 => {
            let buf: Buffer<i256> = vals_i256(seed).into_iter().collect();
            Box::new(move || {
                DecimalArray::new(buf.clone(), dtype, Validity::NonNullable).into_array()
            })
        }
    }
}

fn fresh_byteparts(width: Width, seed: u64) -> Box<dyn Fn() -> ArrayRef + Send + Sync> {
    let dtype = width.dtype();
    let prim = |buf: Buffer<i64>| PrimitiveArray::new(buf, Validity::NonNullable).into_array();
    let uprim = |buf: Buffer<u64>| PrimitiveArray::new(buf, Validity::NonNullable).into_array();
    match width {
        Width::Small => {
            let buf: Buffer<i64> = vals_i64(seed).into_iter().collect();
            Box::new(move || {
                DecimalByteParts::try_new(prim(buf.clone()), dtype)
                    .unwrap()
                    .into_array()
            })
        }
        Width::I128 => {
            let values = vals_i128(seed);
            let msp: Buffer<i64> = values.iter().map(|v| (v >> 64) as i64).collect();
            let low: Buffer<u64> = values.iter().map(|v| *v as u64).collect();
            Box::new(move || {
                DecimalByteParts::try_new_parts(prim(msp.clone()), vec![uprim(low.clone())], dtype)
                    .unwrap()
                    .into_array()
            })
        }
        Width::I256 => {
            let values = vals_i256(seed);
            let msp: Buffer<i64> = values
                .iter()
                .map(|v| (v.to_parts().1 >> 64) as i64)
                .collect();
            let p0: Buffer<u64> = values.iter().map(|v| v.to_parts().1 as u64).collect();
            let p1: Buffer<u64> = values
                .iter()
                .map(|v| (v.to_parts().0 >> 64) as u64)
                .collect();
            let p2: Buffer<u64> = values.iter().map(|v| v.to_parts().0 as u64).collect();
            Box::new(move || {
                DecimalByteParts::try_new_parts(
                    prim(msp.clone()),
                    vec![uprim(p0.clone()), uprim(p1.clone()), uprim(p2.clone())],
                    dtype,
                )
                .unwrap()
                .into_array()
            })
        }
    }
}

// ---- Arrow array builders -------------------------------------------------------------------

fn arrow_dec128(values: impl IntoIterator<Item = i128>, dtype: DecimalDType) -> Decimal128Array {
    Decimal128Array::from_iter_values(values)
        .with_precision_and_scale(dtype.precision(), dtype.scale())
        .unwrap()
}

fn arrow_dec256(values: &[i256], dtype: DecimalDType) -> Decimal256Array {
    Decimal256Array::from_iter_values(values.iter().map(|v| arrow_buffer::i256::from(*v)))
        .with_precision_and_scale(dtype.precision(), dtype.scale())
        .unwrap()
}

// =============================================================================================
// Compare (Lt), array vs array
// =============================================================================================

#[divan::bench(args = WIDTHS)]
fn cmp_lt_canonical(bencher: Bencher, width: Width) {
    let (a, b, session) = (canonical(width, 0), canonical(width, 1), session());
    bencher
        .with_inputs(|| (a.clone(), b.clone(), session.create_execution_ctx()))
        .bench_values(|(a, b, mut ctx)| {
            a.binary(b, Operator::Lt)
                .unwrap()
                .execute::<Canonical>(&mut ctx)
        });
}

#[divan::bench(args = WIDTHS)]
fn cmp_lt_byteparts(bencher: Bencher, width: Width) {
    let (a, b, session) = (byteparts(width, 0), byteparts(width, 1), session());
    bencher
        .with_inputs(|| (a.clone(), b.clone(), session.create_execution_ctx()))
        .bench_values(|(a, b, mut ctx)| {
            a.binary(b, Operator::Lt)
                .unwrap()
                .execute::<Canonical>(&mut ctx)
        });
}

#[divan::bench(args = WIDTHS)]
fn cmp_lt_arrow(bencher: Bencher, width: Width) {
    match width {
        Width::Small => {
            let a = arrow_dec128(vals_i64(0).into_iter().map(i128::from), width.dtype());
            let b = arrow_dec128(vals_i64(1).into_iter().map(i128::from), width.dtype());
            bencher.bench(|| arrow_ord::cmp::lt(black_box(&a), black_box(&b)).unwrap());
        }
        Width::I128 => {
            let a = arrow_dec128(vals_i128(0), width.dtype());
            let b = arrow_dec128(vals_i128(1), width.dtype());
            bencher.bench(|| arrow_ord::cmp::lt(black_box(&a), black_box(&b)).unwrap());
        }
        Width::I256 => {
            let a = arrow_dec256(&vals_i256(0), width.dtype());
            let b = arrow_dec256(&vals_i256(1), width.dtype());
            bencher.bench(|| arrow_ord::cmp::lt(black_box(&a), black_box(&b)).unwrap());
        }
    }
}

// =============================================================================================
// Numeric add, array vs array.
//
// NOTE: Vortex's `binary` rejects arithmetic on decimal dtypes (only primitive dtypes are allowed
// in `Binary::return_dtype`), so there is no `canonical`/`byteparts` engine for numeric ops today.
// Only the Arrow reference is benched here; it is the cost curve a future native byte-parts numeric
// kernel would target.
// =============================================================================================

#[divan::bench(args = WIDTHS)]
fn add_arrow(bencher: Bencher, width: Width) {
    match width {
        Width::Small => {
            let a = arrow_dec128(vals_i64(0).into_iter().map(i128::from), width.dtype());
            let b = arrow_dec128(vals_i64(1).into_iter().map(i128::from), width.dtype());
            bencher.bench(|| arrow_arith::numeric::add(black_box(&a), black_box(&b)).unwrap());
        }
        Width::I128 => {
            let a = arrow_dec128(vals_i128(0), width.dtype());
            let b = arrow_dec128(vals_i128(1), width.dtype());
            bencher.bench(|| arrow_arith::numeric::add(black_box(&a), black_box(&b)).unwrap());
        }
        Width::I256 => {
            let a = arrow_dec256(&vals_i256(0), width.dtype());
            let b = arrow_dec256(&vals_i256(1), width.dtype());
            bencher.bench(|| arrow_arith::numeric::add(black_box(&a), black_box(&b)).unwrap());
        }
    }
}

// =============================================================================================
// Aggregation: sum
// =============================================================================================

#[divan::bench(args = WIDTHS)]
fn sum_canonical(bencher: Bencher, width: Width) {
    let (make, session) = (fresh_canonical(width, 0), session());
    bencher
        .with_inputs(|| (make(), session.create_execution_ctx()))
        .bench_values(|(a, mut ctx)| sum(&a, &mut ctx).unwrap());
}

#[divan::bench(args = WIDTHS)]
fn sum_byteparts(bencher: Bencher, width: Width) {
    let (make, session) = (fresh_byteparts(width, 0), session());
    bencher
        .with_inputs(|| (make(), session.create_execution_ctx()))
        .bench_values(|(a, mut ctx)| sum(&a, &mut ctx).unwrap());
}

#[divan::bench(args = WIDTHS)]
fn sum_arrow(bencher: Bencher, width: Width) {
    match width {
        Width::Small => {
            let a = arrow_dec128(vals_i64(0).into_iter().map(i128::from), width.dtype());
            bencher.bench(|| arrow_arith::aggregate::sum(black_box(&a)));
        }
        Width::I128 => {
            let a = arrow_dec128(vals_i128(0), width.dtype());
            bencher.bench(|| arrow_arith::aggregate::sum(black_box(&a)));
        }
        Width::I256 => {
            let a = arrow_dec256(&vals_i256(0), width.dtype());
            bencher.bench(|| arrow_arith::aggregate::sum(black_box(&a)));
        }
    }
}

// =============================================================================================
// Aggregation: min/max
// =============================================================================================

#[divan::bench(args = WIDTHS)]
fn minmax_canonical(bencher: Bencher, width: Width) {
    let (make, session) = (fresh_canonical(width, 0), session());
    bencher
        .with_inputs(|| (make(), session.create_execution_ctx()))
        .bench_values(|(a, mut ctx)| min_max(&a, &mut ctx).unwrap());
}

#[divan::bench(args = WIDTHS)]
fn minmax_byteparts(bencher: Bencher, width: Width) {
    let (make, session) = (fresh_byteparts(width, 0), session());
    bencher
        .with_inputs(|| (make(), session.create_execution_ctx()))
        .bench_values(|(a, mut ctx)| min_max(&a, &mut ctx).unwrap());
}

#[divan::bench(args = WIDTHS)]
fn minmax_arrow(bencher: Bencher, width: Width) {
    match width {
        Width::Small => {
            let a = arrow_dec128(vals_i64(0).into_iter().map(i128::from), width.dtype());
            bencher.bench(|| {
                (
                    arrow_arith::aggregate::min(black_box(&a)),
                    arrow_arith::aggregate::max(black_box(&a)),
                )
            });
        }
        Width::I128 => {
            let a = arrow_dec128(vals_i128(0), width.dtype());
            bencher.bench(|| {
                (
                    arrow_arith::aggregate::min(black_box(&a)),
                    arrow_arith::aggregate::max(black_box(&a)),
                )
            });
        }
        Width::I256 => {
            let a = arrow_dec256(&vals_i256(0), width.dtype());
            bencher.bench(|| {
                (
                    arrow_arith::aggregate::min(black_box(&a)),
                    arrow_arith::aggregate::max(black_box(&a)),
                )
            });
        }
    }
}
