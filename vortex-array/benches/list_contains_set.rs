// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! `list_contains(lit([...]), column)`, the `IN` shape: a constant set probed by a column of
//! needles, so that the cost of preparing the set shows next to the cost of probing it.
//!
//! Each element type exercises one probe: integers, UTF-8 strings, and nested lists that compare
//! whole rows. Integer needles are also split into chunks, which share one prepared set.

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
use vortex_array::arrays::ListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::dtype::DType;
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
const ROWS: usize = 1_024;
const CHUNKS: usize = 4;
const SET_LENS: &[usize] = &[256];
/// A nested set compares whole rows to sort its elements and to probe them, so it stays smaller.
const NESTED_SET_LENS: &[usize] = &[32];

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

fn bench_in_set(bencher: Bencher, set: Scalar, needles: ArrayRef) {
    let session = vortex_array::array_session();
    // Optimized as a scan optimizes it, so the set arrives normalized.
    let expr = list_contains(lit(set), root())
        .bind(needles.dtype())
        .unwrap()
        .optimize_recursive()
        .unwrap();
    bencher
        .with_inputs(|| {
            (
                needles.clone().apply_bound(&expr).unwrap(),
                session.create_execution_ctx(),
            )
        })
        .bench_values(|(array, mut ctx)| array.execute::<BoolArray>(&mut ctx).unwrap());
}

fn i64_set(values: Vec<i64>) -> Scalar {
    Scalar::list(
        Arc::new(DType::Primitive(PType::I64, Nullability::NonNullable)),
        values.into_iter().map(Scalar::from).collect(),
        Nullability::NonNullable,
    )
}

#[divan::bench(args = SET_LENS)]
fn i64_random(bencher: Bencher, set_len: usize) {
    let (set, needles) = random_i64(set_len);
    let needles = PrimitiveArray::from_iter(needles);
    bench_in_set(bencher, i64_set(set), needles.into_array());
}

#[divan::bench(args = SET_LENS)]
fn i64_random_chunked(bencher: Bencher, set_len: usize) {
    let (set, needles) = random_i64(set_len);
    let chunks = needles
        .chunks(ROWS / CHUNKS)
        .map(|chunk| PrimitiveArray::from_iter(chunk.iter().copied()).into_array());
    let needles = ChunkedArray::try_new(
        chunks,
        DType::Primitive(PType::I64, Nullability::NonNullable),
    )
    .unwrap();
    bench_in_set(bencher, i64_set(set), needles.into_array());
}

#[divan::bench(args = SET_LENS)]
fn utf8_random(bencher: Bencher, set_len: usize) {
    let (set, needles) = random_i64(set_len);
    let set = Scalar::list(
        Arc::new(DType::Utf8(Nullability::NonNullable)),
        set.iter()
            .map(|v| Scalar::utf8(format!("value-{v}"), Nullability::NonNullable))
            .collect(),
        Nullability::NonNullable,
    );
    let needles = VarBinViewArray::from_iter_str(needles.iter().map(|v| format!("value-{v}")));
    bench_in_set(bencher, set, needles.into_array());
}

#[divan::bench(args = NESTED_SET_LENS)]
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
