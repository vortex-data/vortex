// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used)]
#![allow(clippy::cast_possible_truncation)]

use divan::Bencher;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::StructArray;
use vortex_array::expr::case_when;
use vortex_array::expr::get_item;
use vortex_array::expr::gt;
use vortex_array::expr::lit;
use vortex_array::expr::nested_case_when;
use vortex_array::expr::root;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_dtype::FieldNames;

fn main() {
    divan::main();
}

fn make_struct_array(size: usize) -> ArrayRef {
    let data: Buffer<i32> = (0..size as i32).collect();
    let field = data.into_array();
    StructArray::try_new(
        FieldNames::from(["value"]),
        vec![field],
        size,
        Validity::NonNullable,
    )
    .unwrap()
    .into_array()
}

/// Benchmark a simple binary CASE WHEN with varying array sizes.
#[divan::bench(args = [1000, 10000, 100000])]
fn case_when_simple(bencher: Bencher, size: usize) {
    let array = make_struct_array(size);

    // CASE WHEN value > 500 THEN 100 ELSE 0 END
    let expr = case_when(
        gt(get_item("value", root()), lit(500i32)),
        lit(100i32),
        lit(0i32),
    );

    bencher
        .with_inputs(|| (&expr, &array))
        .bench_refs(|(expr, array)| expr.evaluate(array).unwrap());
}

/// Benchmark nested CASE WHEN with multiple conditions.
#[divan::bench(args = [1000, 10000, 100000])]
fn case_when_nested_3_conditions(bencher: Bencher, size: usize) {
    let array = make_struct_array(size);

    // CASE WHEN value > 750 THEN 3 WHEN value > 500 THEN 2 WHEN value > 250 THEN 1 ELSE 0 END
    let expr = nested_case_when(
        vec![
            (gt(get_item("value", root()), lit(750i32)), lit(3i32)),
            (gt(get_item("value", root()), lit(500i32)), lit(2i32)),
            (gt(get_item("value", root()), lit(250i32)), lit(1i32)),
        ],
        Some(lit(0i32)),
    );

    bencher
        .with_inputs(|| (&expr, &array))
        .bench_refs(|(expr, array)| expr.evaluate(array).unwrap());
}

/// Benchmark CASE WHEN where all conditions are true (short-circuit path).
#[divan::bench(args = [1000, 10000, 100000])]
fn case_when_all_true(bencher: Bencher, size: usize) {
    let array = make_struct_array(size);

    // CASE WHEN value >= 0 THEN 100 ELSE 0 END (always true for our data)
    let expr = case_when(
        gt(get_item("value", root()), lit(-1i32)),
        lit(100i32),
        lit(0i32),
    );

    bencher
        .with_inputs(|| (&expr, &array))
        .bench_refs(|(expr, array)| expr.evaluate(array).unwrap());
}

/// Benchmark CASE WHEN where all conditions are false (short-circuit path).
#[divan::bench(args = [1000, 10000, 100000])]
fn case_when_all_false(bencher: Bencher, size: usize) {
    let array = make_struct_array(size);

    // CASE WHEN value > 1000000 THEN 100 ELSE 0 END (always false for our data)
    let expr = case_when(
        gt(get_item("value", root()), lit(1_000_000i32)),
        lit(100i32),
        lit(0i32),
    );

    bencher
        .with_inputs(|| (&expr, &array))
        .bench_refs(|(expr, array)| expr.evaluate(array).unwrap());
}
