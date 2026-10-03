// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks comparing a sorted array against a constant.
//!
//! Each case runs on the same values three ways: with no cached sortedness statistic (the linear
//! kernel), with an exact `IsSorted` statistic, and with an exact `IsStrictSorted` statistic
//! (both take the binary search path).

#![expect(clippy::unwrap_used)]

use divan::Bencher;
use divan::counter::ItemsCount;
use mimalloc::MiMalloc;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::Nullability;
use vortex_array::expr::stats::Precision;
use vortex_array::expr::stats::Stat;
use vortex_array::extension::datetime::TimeUnit;
use vortex_array::extension::datetime::Timestamp;
use vortex_array::scalar::Scalar;
use vortex_array::scalar::ScalarValue;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_buffer::Buffer;
use vortex_mask::Mask;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    divan::main();
}

#[derive(Debug, Clone, Copy)]
enum Sortedness {
    Unknown,
    Sorted,
    Strict,
}

const MODES: &[Sortedness] = &[Sortedness::Unknown, Sortedness::Sorted, Sortedness::Strict];

impl std::fmt::Display for Sortedness {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(self, f)
    }
}

fn mark(array: ArrayRef, sortedness: Sortedness) -> ArrayRef {
    let stat = match sortedness {
        Sortedness::Unknown => return array,
        Sortedness::Sorted => Stat::IsSorted,
        Sortedness::Strict => Stat::IsStrictSorted,
    };
    array
        .statistics()
        .set(stat, Precision::exact(ScalarValue::from(true)));
    array
}

/// How the comparison result is consumed.
#[derive(Debug, Clone, Copy)]
enum Consumer {
    /// One execution step: runs the comparison kernel and returns its output as-is.
    Step,
    /// Fully canonicalize into a `BoolArray`.
    Canonical,
    /// Execute into a filter `Mask`.
    Mask,
}

impl std::fmt::Display for Consumer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(self, f)
    }
}

const CONSUMERS: &[Consumer] = &[Consumer::Step, Consumer::Canonical, Consumer::Mask];

fn bench_compare(bencher: Bencher, lhs: ArrayRef, constant: Scalar, op: Operator) {
    bench_consume(bencher, lhs, constant, op, Consumer::Canonical);
}

fn bench_consume(
    bencher: Bencher,
    lhs: ArrayRef,
    constant: Scalar,
    op: Operator,
    consumer: Consumer,
) {
    let session = vortex_array::array_session();
    let len = lhs.len();
    let rhs = ConstantArray::new(constant, len).into_array();
    bencher
        .counter(ItemsCount::new(len))
        .with_inputs(|| (&lhs, &rhs, session.create_execution_ctx()))
        .bench_refs(|input| {
            let result = input.0.clone().binary(input.1.clone(), op).unwrap();
            let ctx = &mut input.2;
            match consumer {
                Consumer::Step => drop(result.execute::<ArrayRef>(ctx).unwrap()),
                Consumer::Canonical => drop(result.execute::<Canonical>(ctx).unwrap()),
                Consumer::Mask => drop(result.execute::<Mask>(ctx).unwrap()),
            }
        });
}

#[expect(clippy::cast_possible_wrap)]
fn ints(len: usize) -> Buffer<i64> {
    (0..len as i64).map(|i| i * 3).collect()
}

#[expect(clippy::cast_possible_wrap)]
fn needle(len: usize) -> i64 {
    // Midway through the array, between two values.
    (len as i64 / 2) * 3 + 1
}

#[divan::bench(args = MODES, consts = [8_192, 1_048_576])]
fn i64_lt<const N: usize>(bencher: Bencher, mode: Sortedness) {
    let array = mark(ints(N).into_array(), mode);
    bench_compare(bencher, array, Scalar::from(needle(N)), Operator::Lt);
}

#[divan::bench(args = MODES, consts = [8_192, 1_048_576])]
fn i64_eq<const N: usize>(bencher: Bencher, mode: Sortedness) {
    let array = mark(ints(N).into_array(), mode);
    bench_compare(bencher, array, Scalar::from(needle(N) - 1), Operator::Eq);
}

#[divan::bench(args = MODES, consts = [8_192, 1_048_576])]
fn timestamp_gte<const N: usize>(bencher: Bencher, mode: Sortedness) {
    let ext_dtype = Timestamp::new(TimeUnit::Milliseconds, Nullability::NonNullable).erased();
    let array = mark(
        ExtensionArray::new(ext_dtype.clone(), ints(N).into_array()).into_array(),
        mode,
    );
    let constant = Scalar::extension_ref(ext_dtype, Scalar::from(needle(N)));
    bench_compare(bencher, array, constant, Operator::Gte);
}

#[divan::bench(args = MODES, consts = [8_192, 1_048_576])]
fn utf8_lt<const N: usize>(bencher: Bencher, mode: Sortedness) {
    let array = mark(
        VarBinViewArray::from_iter_str((0..N).map(|i| format!("value-{i:012}"))).into_array(),
        mode,
    );
    let constant = Scalar::from(format!("value-{:012}x", N / 2).as_str());
    bench_compare(bencher, array, constant, Operator::Lt);
}

/// Sorted path only, consumed three ways, to separate kernel cost from materialization cost.
#[divan::bench(args = CONSUMERS, consts = [8_192, 1_048_576])]
fn i64_lt_sorted_consume<const N: usize>(bencher: Bencher, consumer: Consumer) {
    let array = mark(ints(N).into_array(), Sortedness::Sorted);
    bench_consume(
        bencher,
        array,
        Scalar::from(needle(N)),
        Operator::Lt,
        consumer,
    );
}

#[divan::bench(args = CONSUMERS, consts = [8_192, 1_048_576])]
fn i64_eq_sorted_consume<const N: usize>(bencher: Bencher, consumer: Consumer) {
    let array = mark(ints(N).into_array(), Sortedness::Sorted);
    bench_consume(
        bencher,
        array,
        Scalar::from(needle(N) - 1),
        Operator::Eq,
        consumer,
    );
}

#[divan::bench(args = CONSUMERS, consts = [8_192, 1_048_576])]
fn i64_lt_unsorted_consume<const N: usize>(bencher: Bencher, consumer: Consumer) {
    let array = mark(ints(N).into_array(), Sortedness::Unknown);
    bench_consume(
        bencher,
        array,
        Scalar::from(needle(N)),
        Operator::Lt,
        consumer,
    );
}
