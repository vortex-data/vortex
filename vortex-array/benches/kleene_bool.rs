// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use divan::counter::ItemsCount;
use vortex_array::ArrayRef;
use vortex_array::Columnar;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrow::ArrowSession;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::session::ArraySession;
use vortex_session::VortexSession;

fn main() {
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    VortexSession::empty()
        .with::<ArraySession>()
        .with::<ArrowSession>()
});

const LEN: usize = 65_536;
const SHIFTED_OFFSET: usize = 1;

#[divan::bench]
fn and_bool_nonnull_arrays(bencher: Bencher) {
    bench_kleene(
        bencher,
        bool_nonnull(2).into_array(),
        bool_nonnull(3).into_array(),
        Operator::And,
    );
}

#[divan::bench]
fn or_bool_nonnull_arrays(bencher: Bencher) {
    bench_kleene(
        bencher,
        bool_nonnull(2).into_array(),
        bool_nonnull(3).into_array(),
        Operator::Or,
    );
}

#[divan::bench]
fn and_bool_nullable_arrays(bencher: Bencher) {
    and_bool_nullable_arrays_aligned(bencher);
}

#[divan::bench]
fn and_bool_nullable_arrays_aligned(bencher: Bencher) {
    bench_kleene(
        bencher,
        bool_nullable(2, 7, 0),
        bool_nullable(3, 5, 0),
        Operator::And,
    );
}

#[divan::bench]
fn or_bool_nullable_arrays(bencher: Bencher) {
    or_bool_nullable_arrays_aligned(bencher);
}

#[divan::bench]
fn or_bool_nullable_arrays_aligned(bencher: Bencher) {
    bench_kleene(
        bencher,
        bool_nullable(2, 7, 0),
        bool_nullable(3, 5, 0),
        Operator::Or,
    );
}

#[divan::bench]
fn and_bool_nullable_arrays_shifted(bencher: Bencher) {
    bench_kleene(
        bencher,
        bool_nullable(2, 7, SHIFTED_OFFSET),
        bool_nullable(3, 5, SHIFTED_OFFSET),
        Operator::And,
    );
}

#[divan::bench]
fn or_bool_nullable_arrays_shifted(bencher: Bencher) {
    bench_kleene(
        bencher,
        bool_nullable(2, 7, SHIFTED_OFFSET),
        bool_nullable(3, 5, SHIFTED_OFFSET),
        Operator::Or,
    );
}

#[divan::bench]
fn and_true_constant(bencher: Bencher) {
    bench_kleene(
        bencher,
        bool_nullable(2, 7, 0),
        ConstantArray::new(true, LEN).into_array(),
        Operator::And,
    );
}

#[divan::bench]
fn or_false_constant(bencher: Bencher) {
    bench_kleene(
        bencher,
        bool_nullable(2, 7, 0),
        ConstantArray::new(false, LEN).into_array(),
        Operator::Or,
    );
}

#[divan::bench]
fn and_false_constant(bencher: Bencher) {
    bench_kleene(
        bencher,
        bool_nullable(2, 7, 0),
        ConstantArray::new(false, LEN).into_array(),
        Operator::And,
    );
}

#[divan::bench]
fn or_true_constant(bencher: Bencher) {
    bench_kleene(
        bencher,
        bool_nullable(2, 7, 0),
        ConstantArray::new(true, LEN).into_array(),
        Operator::Or,
    );
}

#[divan::bench]
fn and_null_constant(bencher: Bencher) {
    and_null_constant_aligned(bencher);
}

#[divan::bench]
fn and_null_constant_aligned(bencher: Bencher) {
    bench_kleene(
        bencher,
        bool_nullable(2, 7, 0),
        null_bool_constant(),
        Operator::And,
    );
}

#[divan::bench]
fn or_null_constant(bencher: Bencher) {
    or_null_constant_aligned(bencher);
}

#[divan::bench]
fn or_null_constant_aligned(bencher: Bencher) {
    bench_kleene(
        bencher,
        bool_nullable(2, 7, 0),
        null_bool_constant(),
        Operator::Or,
    );
}

#[divan::bench]
fn and_null_constant_shifted(bencher: Bencher) {
    bench_kleene(
        bencher,
        bool_nullable(2, 7, SHIFTED_OFFSET),
        null_bool_constant(),
        Operator::And,
    );
}

#[divan::bench]
fn or_null_constant_shifted(bencher: Bencher) {
    bench_kleene(
        bencher,
        bool_nullable(2, 7, SHIFTED_OFFSET),
        null_bool_constant(),
        Operator::Or,
    );
}

fn bench_kleene(bencher: Bencher, lhs: ArrayRef, rhs: ArrayRef, operator: Operator) {
    let mut ctx = SESSION.create_execution_ctx();

    bencher.counter(ItemsCount::new(LEN)).bench_local(|| {
        lhs.clone()
            .binary(rhs.clone(), operator)
            .unwrap()
            .execute::<Columnar>(&mut ctx)
            .unwrap()
    });
}

fn bool_nonnull(true_every: usize) -> BoolArray {
    BoolArray::from_iter((0..LEN).map(|i| i.is_multiple_of(true_every)))
}

fn bool_nullable(true_every: usize, null_every: usize, offset: usize) -> ArrayRef {
    let len = LEN + offset;
    let array = BoolArray::from_iter(
        (0..len).map(|i| (!i.is_multiple_of(null_every)).then_some(i.is_multiple_of(true_every))),
    )
    .into_array();

    if offset == 0 {
        array
    } else {
        array.slice(offset..offset + LEN).unwrap()
    }
}

fn null_bool_constant() -> ArrayRef {
    ConstantArray::new(Scalar::null(DType::Bool(Nullability::Nullable)), LEN).into_array()
}
