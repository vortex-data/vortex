// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation)]

use std::sync::LazyLock;

use divan::Bencher;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::StructArray;
use vortex_array::expr::case_when;
use vortex_array::expr::case_when_no_else;
use vortex_array::expr::eq;
use vortex_array::expr::get_item;
use vortex_array::expr::gt;
use vortex_array::expr::lit;
use vortex_array::expr::lt;
use vortex_array::expr::nested_case_when;
use vortex_array::expr::root;
use vortex_array::session::ArraySession;
use vortex_buffer::Buffer;
use vortex_session::VortexSession;

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

fn main() {
    divan::main();
}

fn make_struct_array(size: usize) -> ArrayRef {
    let data: Buffer<i32> = (0..size as i32).collect();
    let field = data.into_array();
    StructArray::from_fields(&[("value", field)])
        .unwrap()
        .into_array()
}

/// Array with boolean columns cycling through thirds: `c0[i] = i%3==0`, `c1[i] = i%3==1`.
fn make_fragmented_array(size: usize) -> ArrayRef {
    StructArray::from_fields(&[
        (
            "c0",
            BoolArray::from_iter((0..size).map(|i| i % 3 == 0)).into_array(),
        ),
        (
            "c1",
            BoolArray::from_iter((0..size).map(|i| i % 3 == 1)).into_array(),
        ),
    ])
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
        .bench_refs(|(expr, array)| {
            let mut ctx = SESSION.create_execution_ctx();
            array
                .clone()
                .apply(expr)
                .unwrap()
                .execute::<Canonical>(&mut ctx)
                .unwrap()
        });
}

/// Benchmark n-ary CASE WHEN with 3 conditions.
#[divan::bench(args = [1000, 10000])]
fn case_when_nary_3_conditions(bencher: Bencher, size: usize) {
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
        .bench_refs(|(expr, array)| {
            let mut ctx = SESSION.create_execution_ctx();
            array
                .clone()
                .apply(expr)
                .unwrap()
                .execute::<Canonical>(&mut ctx)
                .unwrap()
        });
}

/// Benchmark n-ary CASE WHEN with 10 conditions.
#[divan::bench(args = [1000, 10000])]
fn case_when_nary_10_conditions(bencher: Bencher, size: usize) {
    let array = make_struct_array(size);

    let pairs: Vec<_> = (0..10)
        .map(|i| {
            let threshold = (i + 1) * (size as i32 / 10);
            (
                gt(get_item("value", root()), lit(threshold)),
                lit((i + 1) * 100),
            )
        })
        .collect();
    let expr = nested_case_when(pairs, Some(lit(0i32)));

    bencher
        .with_inputs(|| (&expr, &array))
        .bench_refs(|(expr, array)| {
            let mut ctx = SESSION.create_execution_ctx();
            array
                .clone()
                .apply(expr)
                .unwrap()
                .execute::<Canonical>(&mut ctx)
                .unwrap()
        });
}

/// Benchmark n-ary CASE WHEN with equality conditions (lookup-table style).
#[divan::bench(args = [1000, 10000])]
fn case_when_nary_equality_lookup(bencher: Bencher, size: usize) {
    let array = make_struct_array(size);

    // Map specific values: CASE WHEN value = 0 THEN 'a' WHEN value = 1 THEN 'b' ... ELSE 'other' END
    let pairs: Vec<_> = (0..5)
        .map(|i| (eq(get_item("value", root()), lit(i)), lit(i * 10)))
        .collect();
    let expr = nested_case_when(pairs, Some(lit(-1i32)));

    bencher
        .with_inputs(|| (&expr, &array))
        .bench_refs(|(expr, array)| {
            let mut ctx = SESSION.create_execution_ctx();
            array
                .clone()
                .apply(expr)
                .unwrap()
                .execute::<Canonical>(&mut ctx)
                .unwrap()
        });
}

/// Benchmark CASE WHEN without ELSE clause (result is nullable).
#[divan::bench(args = [1000, 10000, 100000])]
fn case_when_without_else(bencher: Bencher, size: usize) {
    let array = make_struct_array(size);

    // CASE WHEN value > 500 THEN 100 END
    let expr = case_when_no_else(gt(get_item("value", root()), lit(500i32)), lit(100i32));

    bencher
        .with_inputs(|| (&expr, &array))
        .bench_refs(|(expr, array)| {
            let mut ctx = SESSION.create_execution_ctx();
            array
                .clone()
                .apply(expr)
                .unwrap()
                .execute::<Canonical>(&mut ctx)
                .unwrap()
        });
}

/// Benchmark CASE WHEN where all conditions are true.
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
        .bench_refs(|(expr, array)| {
            let mut ctx = SESSION.create_execution_ctx();
            array
                .clone()
                .apply(expr)
                .unwrap()
                .execute::<Canonical>(&mut ctx)
                .unwrap()
        });
}

/// Benchmark n-ary CASE WHEN where the first branch dominates (~90% of rows).
/// This highlights the early-exit and deferred-merge optimizations: subsequent conditions
/// match no remaining rows and are skipped entirely.
#[divan::bench(args = [1000, 10000])]
fn case_when_nary_early_dominant(bencher: Bencher, size: usize) {
    let array = make_struct_array(size);

    // CASE WHEN value < 90% THEN 1 WHEN value < 95% THEN 2 WHEN value < 97.5% THEN 3 ELSE 4
    let t1 = (size as i32 * 9) / 10;
    let t2 = (size as i32 * 19) / 20;
    let t3 = (size as i32 * 39) / 40;

    let expr = nested_case_when(
        vec![
            (lt(get_item("value", root()), lit(t1)), lit(1i32)),
            (lt(get_item("value", root()), lit(t2)), lit(2i32)),
            (lt(get_item("value", root()), lit(t3)), lit(3i32)),
        ],
        Some(lit(4i32)),
    );

    bencher
        .with_inputs(|| (&expr, &array))
        .bench_refs(|(expr, array)| {
            let mut ctx = SESSION.create_execution_ctx();
            array
                .clone()
                .apply(expr)
                .unwrap()
                .execute::<Canonical>(&mut ctx)
                .unwrap()
        });
}

/// Benchmark CASE WHEN where all conditions are false.
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
        .bench_refs(|(expr, array)| {
            let mut ctx = SESSION.create_execution_ctx();
            array
                .clone()
                .apply(expr)
                .unwrap()
                .execute::<Canonical>(&mut ctx)
                .unwrap()
        });
}

/// Benchmark CASE WHEN cycling through 3 branches per row (triggers merge_row_by_row).
/// Run length = 1; exercises branch 0, branch 1, and the else fallback at every 3rd row.
#[divan::bench(args = [100, 1000])]
fn case_when_fragmented(bencher: Bencher, size: usize) {
    let array = make_fragmented_array(size);

    // CASE WHEN c0 THEN 0 WHEN c1 THEN 1 ELSE 2 END
    let expr = nested_case_when(
        vec![
            (get_item("c0", root()), lit(0i32)),
            (get_item("c1", root()), lit(1i32)),
        ],
        Some(lit(2i32)),
    );

    bencher
        .with_inputs(|| (&expr, &array))
        .bench_refs(|(expr, array)| {
            let mut ctx = SESSION.create_execution_ctx();
            array
                .clone()
                .apply(expr)
                .unwrap()
                .execute::<Canonical>(&mut ctx)
                .unwrap()
        });
}
