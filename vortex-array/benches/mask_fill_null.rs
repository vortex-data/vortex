// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Microbenchmark isolating why `fill_null(false)` + `Mask::execute` is slow on a *lazy*,
//! nullable predicate result (the shape produced by the dict layout reader for a filter such as
//! `URL LIKE '%google%'` over a nullable, dictionary-encoded string column).
//!
//! The lazy array is `Dict{ codes, values: distinct.apply(like) }`. Executing it into a `Mask`
//! has two strategies:
//!
//! * `fill_null_then_mask` — `array.fill_null(false)?.execute::<Mask>()`. `fill_null`'s
//!   precondition calls `array.validity()?` on the lazy array, which re-evaluates the `LIKE`
//!   predicate to derive validity (`Dict::validity` -> `ScalarFn::validity` -> `execute_expr`),
//!   then materializes an intermediate array, after which `Mask::execute` canonicalizes again.
//! * `coercing_nulls` — `array.null_as_false().execute()`. Canonicalizes once, then folds
//!   validity into the value bits with a single bitmap `AND`.
//!
//! `validity_lazy` vs `validity_canonical` isolates the single `.validity()` call that
//! `fill_null`'s precondition makes: O(predicate re-execution) on the lazy array vs O(1) on the
//! canonical one.

#![expect(clippy::unwrap_used)]

use divan::Bencher;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::expr::like;
use vortex_array::expr::lit;
use vortex_array::expr::root;
use vortex_mask::Mask;

fn main() {
    divan::main();
}

const LEN: usize = 65_536;

/// Distinct-value counts mirroring dict cardinality: from a low-cardinality categorical column up
/// to a near-unique high-cardinality column such as ClickBench `URL`. The re-executed `LIKE`
/// validity scales with this count, so it is the parameter that drives the regression.
const CARDINALITIES: &[usize] = &[256, 4_096, 32_768, 65_536];

/// A lazy `Dict{codes, values: distinct.apply(like('%google%'))}` of nullable booleans, mirroring
/// the dict reader's `values.take(codes)` for a `LIKE` filter over a nullable string column.
fn lazy_dict_predicate(n_distinct: usize) -> ArrayRef {
    // Distinct dictionary values: nullable URL-like strings, ~1/8 matching "%google%", 1/16 null.
    let distinct = VarBinViewArray::from_iter_nullable_str((0..n_distinct).map(|i| {
        if i % 16 == 0 {
            None
        } else if i % 8 == 0 {
            Some(format!("http://www.google.com/path/{i}"))
        } else {
            Some(format!("http://example.com/page/{i}"))
        }
    }))
    .into_array();

    // Lazy predicate over the distinct values: not canonicalized.
    let predicate = distinct.apply(&like(root(), lit("%google%"))).unwrap();

    let codes = PrimitiveArray::from_iter((0..LEN).map(|i| (i % n_distinct) as u64)).into_array();

    DictArray::try_new(codes, predicate).unwrap().into_array()
}

/// A lazy `ScalarFn` array: `column.apply(like('%google%'))` over a full-length nullable string
/// column, mirroring the flat reader's `array.apply(&expr)`. Unlike the dict case, `ScalarFn`'s
/// `validity()` *executes* the predicate expression, so `fill_null`'s precondition pays for it.
fn lazy_scalar_fn_predicate() -> ArrayRef {
    let column = VarBinViewArray::from_iter_nullable_str((0..LEN).map(|i| {
        if i % 16 == 0 {
            None
        } else if i % 8 == 0 {
            Some(format!("http://www.google.com/path/{i}"))
        } else {
            Some(format!("http://example.com/page/{i}"))
        }
    }))
    .into_array();

    column.apply(&like(root(), lit("%google%"))).unwrap()
}

/// #8121 path: `fill_null(false)` on the lazy array, then `Mask::execute`.
#[divan::bench(args = CARDINALITIES)]
fn fill_null_then_mask(bencher: Bencher, n_distinct: usize) {
    let array = lazy_dict_predicate(n_distinct);
    let session = vortex_array::array_session();
    bencher
        .with_inputs(|| (array.clone(), session.create_execution_ctx()))
        .bench_values(|(array, mut ctx)| {
            array
                .fill_null(false)
                .unwrap()
                .execute::<Mask>(&mut ctx)
                .unwrap()
        });
}

/// Fix: canonicalize once and fold validity into the bits.
#[divan::bench(args = CARDINALITIES)]
fn coercing_nulls(bencher: Bencher, n_distinct: usize) {
    let array = lazy_dict_predicate(n_distinct);
    let session = vortex_array::array_session();
    bencher
        .with_inputs(|| (array.clone(), session.create_execution_ctx()))
        .bench_values(|(array, mut ctx)| array.null_as_false().execute(&mut ctx).unwrap());
}

/// Isolation: the single `.validity()` call `fill_null`'s precondition makes, on the lazy array.
/// This re-executes the `LIKE` predicate to derive validity, so it scales with cardinality.
#[divan::bench(args = CARDINALITIES)]
fn validity_lazy(bencher: Bencher, n_distinct: usize) {
    let array = lazy_dict_predicate(n_distinct);
    bencher
        .with_inputs(|| array.clone())
        .bench_values(|array| array.validity().unwrap());
}

/// Isolation: the same `.validity()` call once the array is already canonical — O(1) bitmap read.
#[divan::bench(args = CARDINALITIES)]
fn validity_canonical(bencher: Bencher, n_distinct: usize) {
    let array = lazy_dict_predicate(n_distinct);
    let mut ctx = vortex_array::array_session().create_execution_ctx();
    let canonical = array.execute::<BoolArray>(&mut ctx).unwrap().into_array();
    bencher
        .with_inputs(|| canonical.clone())
        .bench_values(|array| array.validity().unwrap());
}

// --- ScalarFn (flat reader) shape: this is the case `fill_null` handles badly. ---

/// #8121 path on a `ScalarFn` array: `fill_null`'s precondition `.validity()` executes the predicate.
#[divan::bench]
fn scalar_fn_fill_null_then_mask(bencher: Bencher) {
    let array = lazy_scalar_fn_predicate();
    let session = vortex_array::array_session();
    bencher
        .with_inputs(|| (array.clone(), session.create_execution_ctx()))
        .bench_values(|(array, mut ctx)| {
            array
                .fill_null(false)
                .unwrap()
                .execute::<Mask>(&mut ctx)
                .unwrap()
        });
}

/// Fix on a `ScalarFn` array: canonicalize once, then fold validity.
#[divan::bench]
fn scalar_fn_coercing_nulls(bencher: Bencher) {
    let array = lazy_scalar_fn_predicate();
    let session = vortex_array::array_session();
    bencher
        .with_inputs(|| (array.clone(), session.create_execution_ctx()))
        .bench_values(|(array, mut ctx)| array.null_as_false().execute(&mut ctx).unwrap());
}

/// Isolation: `.validity()` on a lazy `ScalarFn` array — executes the predicate expression.
#[divan::bench]
fn scalar_fn_validity_lazy(bencher: Bencher) {
    let array = lazy_scalar_fn_predicate();
    bencher
        .with_inputs(|| array.clone())
        .bench_values(|array| array.validity().unwrap());
}
