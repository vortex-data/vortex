// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! `list_contains(lit([...]), column)`, the `IN` shape: a constant set probed by a column of
//! needles, flat and split into chunks so that the cost of preparing the set shows next to the
//! cost of probing it.

#![expect(clippy::unwrap_used)]

use std::sync::Arc;

use divan::Bencher;
use rand::RngExt;
use rand::SeedableRng;
use rand::prelude::StdRng;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::expr::list_contains;
use vortex_array::expr::lit;
use vortex_array::expr::root;
use vortex_array::scalar::Scalar;

fn main() {
    divan::main();
}

// Sized to keep CodSpeed simulation under 1ms per benchmark.
const ROWS: usize = 4_096;
const CHUNKS: usize = 16;
const SET_LENS: &[usize] = &[4, 64, 1_024];

/// A random set of `len` values, and needles of which about half are members.
fn random_i64(len: usize) -> (Vec<i64>, Vec<i64>) {
    let mut rng = StdRng::seed_from_u64(0);
    let set: Vec<i64> = (0..len).map(|_| rng.random_range(0..1 << 40)).collect();
    let needles = (0..ROWS)
        .map(|_| {
            if rng.random_bool(0.5) {
                set[rng.random_range(0..len)]
            } else {
                rng.random_range(0..1 << 40)
            }
        })
        .collect();
    (set, needles)
}

/// The set `0..len`, and needles drawn from twice that range.
fn dense_i64(len: usize) -> (Vec<i64>, Vec<i64>) {
    let mut rng = StdRng::seed_from_u64(0);
    let set = (0..len as i64).collect();
    let needles = (0..ROWS)
        .map(|_| rng.random_range(0..2 * len as i64))
        .collect();
    (set, needles)
}

fn i64_set(values: &[i64]) -> Scalar {
    Scalar::list(
        Arc::new(DType::Primitive(PType::I64, Nullability::NonNullable)),
        values.iter().map(|&v| v.into()).collect(),
        Nullability::NonNullable,
    )
}

fn utf8_set(values: &[String]) -> Scalar {
    Scalar::list(
        Arc::new(DType::Utf8(Nullability::NonNullable)),
        values
            .iter()
            .map(|v| Scalar::utf8(v.as_str(), Nullability::NonNullable))
            .collect(),
        Nullability::NonNullable,
    )
}

fn chunked(needles: &[ArrayRef]) -> ArrayRef {
    let dtype = needles[0].dtype().clone();
    ChunkedArray::try_new(needles.iter().cloned(), dtype)
        .unwrap()
        .into_array()
}

fn i64_needles(needles: &[i64], chunks: usize) -> ArrayRef {
    let parts: Vec<ArrayRef> = needles
        .chunks(needles.len() / chunks)
        .map(|chunk| PrimitiveArray::from_iter(chunk.iter().copied()).into_array())
        .collect();
    if chunks == 1 {
        parts[0].clone()
    } else {
        chunked(&parts)
    }
}

fn utf8_needles(needles: &[String], chunks: usize) -> ArrayRef {
    let parts: Vec<ArrayRef> = needles
        .chunks(needles.len() / chunks)
        .map(|chunk| VarBinViewArray::from_iter_str(chunk.iter().map(String::as_str)).into_array())
        .collect();
    if chunks == 1 {
        parts[0].clone()
    } else {
        chunked(&parts)
    }
}

fn bench_in_set(bencher: Bencher, set: Scalar, needles: ArrayRef) {
    let session = vortex_array::array_session();
    // Optimized as a scan optimizes it, so the set arrives normalized.
    let expr = list_contains(lit(set), root())
        .optimize_recursive(needles.dtype())
        .unwrap();
    bencher
        .with_inputs(|| {
            (
                needles.clone().apply(&expr).unwrap(),
                session.create_execution_ctx(),
            )
        })
        .bench_values(|(array, mut ctx)| array.execute::<BoolArray>(&mut ctx).unwrap());
}

#[divan::bench(args = SET_LENS)]
fn i64_random(bencher: Bencher, set_len: usize) {
    let (set, needles) = random_i64(set_len);
    bench_in_set(bencher, i64_set(&set), i64_needles(&needles, 1));
}

#[divan::bench(args = SET_LENS)]
fn i64_random_chunked(bencher: Bencher, set_len: usize) {
    let (set, needles) = random_i64(set_len);
    bench_in_set(bencher, i64_set(&set), i64_needles(&needles, CHUNKS));
}

#[divan::bench(args = SET_LENS)]
fn i64_dense(bencher: Bencher, set_len: usize) {
    let (set, needles) = dense_i64(set_len);
    bench_in_set(bencher, i64_set(&set), i64_needles(&needles, 1));
}

#[divan::bench(args = SET_LENS)]
fn utf8_random(bencher: Bencher, set_len: usize) {
    let (set, needles) = random_i64(set_len);
    let set: Vec<String> = set.iter().map(|v| format!("value-{v}")).collect();
    let needles: Vec<String> = needles.iter().map(|v| format!("value-{v}")).collect();
    bench_in_set(bencher, utf8_set(&set), utf8_needles(&needles, 1));
}

#[divan::bench(args = SET_LENS)]
fn utf8_random_chunked(bencher: Bencher, set_len: usize) {
    let (set, needles) = random_i64(set_len);
    let set: Vec<String> = set.iter().map(|v| format!("value-{v}")).collect();
    let needles: Vec<String> = needles.iter().map(|v| format!("value-{v}")).collect();
    bench_in_set(bencher, utf8_set(&set), utf8_needles(&needles, CHUNKS));
}

#[divan::bench(args = [4_096, 16_384])]
fn i64_random_large(bencher: Bencher, set_len: usize) {
    let (set, needles) = random_i64(set_len);
    bench_in_set(bencher, i64_set(&set), i64_needles(&needles, 1));
}

#[divan::bench(args = SET_LENS)]
fn decimal_random(bencher: Bencher, set_len: usize) {
    let (set, needles) = random_i64(set_len);
    bench_decimal(bencher, set, needles);
}

#[divan::bench(args = SET_LENS)]
fn decimal_dense(bencher: Bencher, set_len: usize) {
    let (set, needles) = dense_i64(set_len);
    bench_decimal(bencher, set, needles);
}

fn bench_decimal(bencher: Bencher, set: Vec<i64>, needles: Vec<i64>) {
    let dtype = DecimalDType::new(20, 2);
    let set = Scalar::list(
        DType::Decimal(dtype, Nullability::NonNullable),
        set.into_iter()
            .map(|value| Scalar::decimal(value.into(), dtype, Nullability::NonNullable))
            .collect(),
        Nullability::NonNullable,
    );
    let needles = DecimalArray::from_iter::<i64, _>(needles, dtype).into_array();
    bench_in_set(bencher, set, needles);
}

#[divan::bench(args = SET_LENS)]
fn nested_list_random(bencher: Bencher, set_len: usize) {
    let (set, needles) = random_i64(set_len);
    let element_dtype = Arc::new(DType::Primitive(PType::I64, Nullability::NonNullable));
    let set = Scalar::list(
        DType::List(Arc::clone(&element_dtype), Nullability::NonNullable),
        set.into_iter()
            .map(|value| {
                Scalar::list(
                    Arc::clone(&element_dtype),
                    vec![value.into(), (value + 1).into()],
                    Nullability::NonNullable,
                )
            })
            .collect(),
        Nullability::NonNullable,
    );
    let needles = ListArray::from_iter_slow::<u64, _>(
        needles.into_iter().map(|value| vec![value, value + 1]),
        element_dtype,
    )
    .unwrap()
    .into_array();
    bench_in_set(bencher, set, needles);
}
