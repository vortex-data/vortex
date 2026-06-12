// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]
#![expect(
    clippy::cast_possible_truncation,
    reason = "benchmark fixtures use indices that fit in the chosen widths"
)]

use std::sync::LazyLock;

use divan::Bencher;
use divan::counter::ItemsCount;
use vortex_array::ArrayRef;
use vortex_array::Executable;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::session::ArraySession;
use vortex_session::VortexSession;

fn main() {
    divan::main();
}

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

const LEN: usize = 65_536;

#[divan::bench]
fn add_i64_nonnull(bencher: Bencher) {
    let lhs = primitive_nonnull(0).into_array();
    let rhs = primitive_nonnull(1_000_000).into_array();

    bench_primitive(bencher, lhs, rhs, Operator::Add);
}

#[divan::bench]
fn add_i64_nullable(bencher: Bencher) {
    let lhs = primitive_nullable(0, 7).into_array();
    let rhs = primitive_nullable(1_000_000, 5).into_array();

    bench_primitive(bencher, lhs, rhs, Operator::Add);
}

#[divan::bench]
fn mul_i64_nonnull(bencher: Bencher) {
    let lhs = primitive_small_nonnull(1).into_array();
    let rhs = primitive_small_nonnull(17).into_array();

    bench_primitive(bencher, lhs, rhs, Operator::Mul);
}

#[divan::bench]
fn div_i64_nonnull(bencher: Bencher) {
    let lhs = primitive_nonnull(1_000_000).into_array();
    let rhs = primitive_nonzero().into_array();

    bench_primitive(bencher, lhs, rhs, Operator::Div);
}

#[divan::bench]
fn sub_i64_constant(bencher: Bencher) {
    let lhs = primitive_nonnull(0).into_array();
    let rhs = ConstantArray::new(37i64, LEN).into_array();

    bench_primitive(bencher, lhs, rhs, Operator::Sub);
}

#[divan::bench]
fn eq_i64_constant(bencher: Bencher) {
    let lhs = primitive_nonnull(0).into_array();
    let rhs = ConstantArray::new(1024i64, LEN).into_array();

    bench_bool(bencher, lhs, rhs, Operator::Eq);
}

#[divan::bench]
fn lt_i64_nullable(bencher: Bencher) {
    let lhs = primitive_nullable(0, 7).into_array();
    let rhs = primitive_nullable(1_000_000, 5).into_array();

    bench_bool(bencher, lhs, rhs, Operator::Lt);
}

#[divan::bench]
fn and_bool_nullable(bencher: Bencher) {
    let lhs = bool_nullable(2, 7).into_array();
    let rhs = bool_nullable(3, 5).into_array();

    bench_bool(bencher, lhs, rhs, Operator::And);
}

#[divan::bench]
fn or_bool_constant(bencher: Bencher) {
    let lhs = bool_nullable(2, 7).into_array();
    let rhs = ConstantArray::new(true, LEN).into_array();

    bench_bool(bencher, lhs, rhs, Operator::Or);
}

fn bench_primitive(bencher: Bencher, lhs: ArrayRef, rhs: ArrayRef, operator: Operator) {
    bench_binary::<PrimitiveArray>(bencher, lhs, rhs, operator);
}

fn bench_bool(bencher: Bencher, lhs: ArrayRef, rhs: ArrayRef, operator: Operator) {
    bench_binary::<BoolArray>(bencher, lhs, rhs, operator);
}

fn bench_binary<T: Executable + 'static>(
    bencher: Bencher,
    lhs: ArrayRef,
    rhs: ArrayRef,
    operator: Operator,
) {
    let mut ctx = SESSION.create_execution_ctx();

    bencher.counter(ItemsCount::new(LEN)).bench_local(|| {
        lhs.clone()
            .binary(rhs.clone(), operator)
            .unwrap()
            .execute::<T>(&mut ctx)
            .unwrap()
    });
}

fn primitive_nonnull(base: i64) -> PrimitiveArray {
    PrimitiveArray::from_iter((0..LEN as i64).map(|i| base + i))
}

fn primitive_small_nonnull(offset: i64) -> PrimitiveArray {
    PrimitiveArray::from_iter((0..LEN as i64).map(|i| ((i + offset) % 1024) + 1))
}

fn primitive_nonzero() -> PrimitiveArray {
    PrimitiveArray::from_iter((0..LEN as i64).map(|i| (i % 255) + 1))
}

fn primitive_nullable(base: i64, null_every: usize) -> PrimitiveArray {
    PrimitiveArray::from_option_iter(
        (0..LEN as i64).map(|i| (!(i as usize).is_multiple_of(null_every)).then_some(base + i)),
    )
}

fn bool_nullable(true_every: usize, null_every: usize) -> BoolArray {
    BoolArray::from_iter(
        (0..LEN).map(|i| (!i.is_multiple_of(null_every)).then_some(i.is_multiple_of(true_every))),
    )
}
