// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Four sorted-compare cases: an integer baseline, cached integer sortedness, timestamp dispatch,
//! and UTF-8 byte comparisons. Canonicalization includes the cost of materializing the result.

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

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    divan::main();
}

// Target less than 1 ms per iteration, including before the sorted fast path is implemented.
const LEN: u32 = 4_096;

fn bench_compare(bencher: Bencher, lhs: ArrayRef, constant: Scalar, op: Operator, sorted: bool) {
    if sorted {
        lhs.statistics()
            .set(Stat::IsSorted, Precision::exact(ScalarValue::from(true)));
    }
    let session = vortex_array::array_session();
    let rhs = ConstantArray::new(constant, lhs.len()).into_array();
    bencher
        .counter(ItemsCount::new(lhs.len()))
        .with_inputs(|| (&lhs, &rhs, session.create_execution_ctx()))
        .bench_refs(|input| {
            input
                .0
                .clone()
                .binary(input.1.clone(), op)
                .unwrap()
                .execute::<Canonical>(&mut input.2)
                .unwrap()
        });
}

fn ints() -> Buffer<i64> {
    (0..LEN).map(|i| i64::from(i) * 3).collect()
}

fn needle() -> i64 {
    // Midway through the array, between two values.
    i64::from(LEN / 2) * 3 + 1
}

#[divan::bench(args = [false, true])]
fn i64_lt(bencher: Bencher, sorted: bool) {
    bench_compare(
        bencher,
        ints().into_array(),
        needle().into(),
        Operator::Lt,
        sorted,
    );
}

#[divan::bench]
fn timestamp_gte(bencher: Bencher) {
    let dtype = Timestamp::new(TimeUnit::Milliseconds, Nullability::NonNullable).erased();
    let array = ExtensionArray::new(dtype.clone(), ints().into_array()).into_array();
    let constant = Scalar::extension_ref(dtype, needle().into());
    bench_compare(bencher, array, constant, Operator::Gte, true);
}

#[divan::bench]
fn utf8_lt(bencher: Bencher) {
    let array =
        VarBinViewArray::from_iter_str((0..LEN).map(|i| format!("value-{i:012}"))).into_array();
    let constant = Scalar::from(format!("value-{:012}x", LEN / 2).as_str());
    bench_compare(bencher, array, constant, Operator::Lt, true);
}
