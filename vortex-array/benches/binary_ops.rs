// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks for the binary `Operator` path, over primitive, decimal, and boolean inputs.
//!
//! The primitive arithmetic and comparison cases carry `#[cpu_features]`, so they are
//! measured on every walltime CPU-feature leg rather than in simulation. Each is written
//! once and compiled differently per leg: the kernel underneath is a portable lane loop, and
//! how wide the compiler vectorizes it is decided by the build flags, not by the source.
//!
//! The decimal and boolean cases are not tagged. Decimal arithmetic is `i128` widening and
//! per-lane rescaling, and the boolean kernels are already word-at-a-time over a bitmap;
//! neither is where a wider vector register shows up. They stay in simulation.

#![expect(clippy::unwrap_used)]
#![expect(
    clippy::cast_possible_truncation,
    reason = "benchmark fixtures use indices that fit in the chosen widths"
)]

use std::mem::size_of;
use std::sync::LazyLock;

use divan::Bencher;
use divan::counter::ItemsCount;
use mimalloc::MiMalloc;
use vortex_array::ArrayRef;
use vortex_array::Executable;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DecimalDType;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(array_session);

/// Sized to keep CodSpeed simulation under 1ms per benchmark.
const LEN: usize = 4_096;

/// Primitive cases process at least this many rows and this many value bytes per varying input.
/// This lengthens narrow integer cases while keeping the current CodSpeed simulations below 1 ms.
const MIN_PRIMITIVE_LEN: usize = 16_384;
const MIN_PRIMITIVE_INPUT_BYTES: usize = 96 * 1_024;

const I8_LEN: usize = primitive_len::<i8>();
const I16_LEN: usize = primitive_len::<i16>();
const I32_LEN: usize = primitive_len::<i32>();
const I64_LEN: usize = primitive_len::<i64>();

const BINARY_SHAPE_CASES: &[(usize, BinaryShape)] = &[
    (128, BinaryShape::PerRowPerRow),
    (128, BinaryShape::PerRowConstant),
    (128, BinaryShape::ConstantPerRow),
    (128, BinaryShape::PerRowNullableConstant),
    (I64_LEN, BinaryShape::PerRowPerRow),
    (I64_LEN, BinaryShape::PerRowConstant),
    (I64_LEN, BinaryShape::ConstantPerRow),
    (I64_LEN, BinaryShape::PerRowNullableConstant),
];

#[derive(Clone, Copy, Debug)]
enum BinaryShape {
    PerRowPerRow,
    PerRowConstant,
    ConstantPerRow,
    PerRowNullableConstant,
}

/// Decimal Mul and Div cost far more per lane than Add, so they run over a shorter array to keep
/// the instrumented CodSpeed runs quick.
const DECIMAL_MUL_DIV_LEN: usize = 1_024;

#[vortex_bench_support::cpu_features]
#[divan::bench(args = BINARY_SHAPE_CASES)]
fn add_shapes(bencher: Bencher, &(len, shape): &(usize, BinaryShape)) {
    bench_binary_shape(bencher, len, shape, Operator::Add);
}

#[vortex_bench_support::cpu_features]
#[divan::bench(args = BINARY_SHAPE_CASES)]
fn subtract_shapes(bencher: Bencher, &(len, shape): &(usize, BinaryShape)) {
    bench_binary_shape(bencher, len, shape, Operator::Sub);
}

#[vortex_bench_support::cpu_features]
#[divan::bench(args = BINARY_SHAPE_CASES)]
fn multiply_shapes(bencher: Bencher, &(len, shape): &(usize, BinaryShape)) {
    bench_binary_shape(bencher, len, shape, Operator::Mul);
}

fn bench_binary_shape(bencher: Bencher, len: usize, shape: BinaryShape, operator: Operator) {
    let per_row =
        || PrimitiveArray::from_iter((0..len).map(|index| (index % 1_024) as i64 + 1)).into_array();
    let constant = || ConstantArray::new(17_i64, len).into_array();
    let nullable_constant = || ConstantArray::new(Some(17_i64), len).into_array();

    let (lhs, rhs) = match shape {
        BinaryShape::PerRowPerRow => (per_row(), per_row()),
        BinaryShape::PerRowConstant => (per_row(), constant()),
        BinaryShape::ConstantPerRow => (constant(), per_row()),
        BinaryShape::PerRowNullableConstant => (per_row(), nullable_constant()),
    };

    bench_primitive(bencher, lhs, rhs, operator);
}

#[vortex_bench_support::cpu_features]
#[divan::bench]
fn add_i64_nonnull(bencher: Bencher) {
    let lhs = primitive_nonnull(0, I64_LEN).into_array();
    let rhs = primitive_nonnull(1_000_000, I64_LEN).into_array();

    bench_primitive(bencher, lhs, rhs, Operator::Add);
}

#[vortex_bench_support::cpu_features]
#[divan::bench]
fn add_i64_nullable(bencher: Bencher) {
    let lhs = primitive_nullable(0, 7, I64_LEN).into_array();
    let rhs = primitive_nullable(1_000_000, 5, I64_LEN).into_array();

    bench_primitive(bencher, lhs, rhs, Operator::Add);
}

#[vortex_bench_support::cpu_features]
#[divan::bench]
fn add_i64_constant(bencher: Bencher) {
    let lhs = primitive_nonnull(0, I64_LEN).into_array();
    let rhs = ConstantArray::new(1_000_000i64, I64_LEN).into_array();

    bench_primitive(bencher, lhs, rhs, Operator::Add);
}

#[vortex_bench_support::cpu_features]
#[divan::bench]
fn add_i32_nonnull(bencher: Bencher) {
    let lhs = primitive_i32_small_nonnull(1, I32_LEN).into_array();
    let rhs = primitive_i32_small_nonnull(17, I32_LEN).into_array();

    bench_primitive(bencher, lhs, rhs, Operator::Add);
}

#[vortex_bench_support::cpu_features]
#[divan::bench]
fn add_u32_nonnull(bencher: Bencher) {
    let lhs = primitive_u32_small_nonnull(1, I32_LEN).into_array();
    let rhs = primitive_u32_small_nonnull(17, I32_LEN).into_array();

    bench_primitive(bencher, lhs, rhs, Operator::Add);
}

#[vortex_bench_support::cpu_features]
#[divan::bench]
fn mul_i64_nonnull(bencher: Bencher) {
    let lhs = primitive_small_nonnull(1, I64_LEN).into_array();
    let rhs = primitive_small_nonnull(17, I64_LEN).into_array();

    bench_primitive(bencher, lhs, rhs, Operator::Mul);
}

#[vortex_bench_support::cpu_features]
#[divan::bench]
fn mul_i8_nonnull(bencher: Bencher) {
    let lhs = primitive_i8_small_nonnull(1, I8_LEN).into_array();
    let rhs = primitive_i8_small_nonnull(7, I8_LEN).into_array();

    bench_primitive(bencher, lhs, rhs, Operator::Mul);
}

#[vortex_bench_support::cpu_features]
#[divan::bench]
fn mul_u8_nonnull(bencher: Bencher) {
    let lhs = primitive_u8_small_nonnull(1, I8_LEN).into_array();
    let rhs = primitive_u8_small_nonnull(7, I8_LEN).into_array();

    bench_primitive(bencher, lhs, rhs, Operator::Mul);
}

#[vortex_bench_support::cpu_features]
#[divan::bench]
fn mul_i16_nonnull(bencher: Bencher) {
    let lhs = primitive_i16_small_nonnull(1, I16_LEN).into_array();
    let rhs = primitive_i16_small_nonnull(17, I16_LEN).into_array();

    bench_primitive(bencher, lhs, rhs, Operator::Mul);
}

#[vortex_bench_support::cpu_features]
#[divan::bench]
fn mul_u16_nonnull(bencher: Bencher) {
    let lhs = primitive_u16_small_nonnull(1, I16_LEN).into_array();
    let rhs = primitive_u16_small_nonnull(17, I16_LEN).into_array();

    bench_primitive(bencher, lhs, rhs, Operator::Mul);
}

#[vortex_bench_support::cpu_features]
#[divan::bench]
fn mul_i32_nonnull(bencher: Bencher) {
    let lhs = primitive_i32_small_nonnull(1, I32_LEN).into_array();
    let rhs = primitive_i32_small_nonnull(17, I32_LEN).into_array();

    bench_primitive(bencher, lhs, rhs, Operator::Mul);
}

#[vortex_bench_support::cpu_features]
#[divan::bench]
fn mul_u32_nonnull(bencher: Bencher) {
    let lhs = primitive_u32_small_nonnull(1, I32_LEN).into_array();
    let rhs = primitive_u32_small_nonnull(17, I32_LEN).into_array();

    bench_primitive(bencher, lhs, rhs, Operator::Mul);
}

#[vortex_bench_support::cpu_features]
#[divan::bench]
fn mul_u64_nonnull(bencher: Bencher) {
    let lhs = primitive_u64_small_nonnull(1, I64_LEN).into_array();
    let rhs = primitive_u64_small_nonnull(17, I64_LEN).into_array();

    bench_primitive(bencher, lhs, rhs, Operator::Mul);
}

#[vortex_bench_support::cpu_features]
#[divan::bench]
fn mul_i32_nullable(bencher: Bencher) {
    let lhs = primitive_i32_small_nullable(1, 7, I32_LEN).into_array();
    let rhs = primitive_i32_small_nullable(17, 5, I32_LEN).into_array();

    bench_primitive(bencher, lhs, rhs, Operator::Mul);
}

#[vortex_bench_support::cpu_features]
#[divan::bench]
fn mul_i32_constant(bencher: Bencher) {
    let lhs = primitive_i32_small_nonnull(1, I32_LEN).into_array();
    let rhs = ConstantArray::new(31i32, I32_LEN).into_array();

    bench_primitive(bencher, lhs, rhs, Operator::Mul);
}

#[vortex_bench_support::cpu_features]
#[divan::bench]
fn div_i64_nonnull(bencher: Bencher) {
    let lhs = primitive_nonnull(1_000_000, I64_LEN).into_array();
    let rhs = primitive_nonzero(I64_LEN).into_array();

    bench_primitive(bencher, lhs, rhs, Operator::Div);
}

#[vortex_bench_support::cpu_features]
#[divan::bench]
fn div_i64_nullable(bencher: Bencher) {
    let lhs = primitive_nullable(1_000_000, 7, I64_LEN).into_array();
    let rhs = primitive_nullable(17, 5, I64_LEN).into_array();

    bench_primitive(bencher, lhs, rhs, Operator::Div);
}

#[vortex_bench_support::cpu_features]
#[divan::bench]
fn sub_i64_constant(bencher: Bencher) {
    let lhs = primitive_nonnull(0, I64_LEN).into_array();
    let rhs = ConstantArray::new(37i64, I64_LEN).into_array();

    bench_primitive(bencher, lhs, rhs, Operator::Sub);
}

#[divan::bench]
fn add_decimal_i64_nonnull(bencher: Bencher) {
    let lhs = decimal_i64_nonnull(0, LEN).into_array();
    let rhs = decimal_i64_nonnull(1_000_000, LEN).into_array();

    bench_decimal(bencher, lhs, rhs, Operator::Add);
}

#[divan::bench]
fn add_decimal_i128_nullable(bencher: Bencher) {
    let lhs = decimal_i128_nullable(0, 7, LEN).into_array();
    let rhs = decimal_i128_nullable(1_000_000, 5, LEN).into_array();

    bench_decimal(bencher, lhs, rhs, Operator::Add);
}

#[divan::bench]
fn mul_decimal_i64_nonnull(bencher: Bencher) {
    let lhs = decimal_i64_nonnull(0, DECIMAL_MUL_DIV_LEN).into_array();
    let rhs = decimal_i64_nonnull(1_000_000, DECIMAL_MUL_DIV_LEN).into_array();

    bench_decimal(bencher, lhs, rhs, Operator::Mul);
}

#[divan::bench]
fn mul_decimal_i128_nullable(bencher: Bencher) {
    let lhs = decimal_i128_nullable(0, 7, DECIMAL_MUL_DIV_LEN).into_array();
    let rhs = decimal_i128_nullable(1_000_000, 5, DECIMAL_MUL_DIV_LEN).into_array();

    bench_decimal(bencher, lhs, rhs, Operator::Mul);
}

#[divan::bench]
fn div_decimal_i64_nonnull(bencher: Bencher) {
    let lhs = decimal_i64_nonnull(0, DECIMAL_MUL_DIV_LEN).into_array();
    let rhs = decimal_i64_nonnull(1_000_000, DECIMAL_MUL_DIV_LEN).into_array();

    bench_decimal(bencher, lhs, rhs, Operator::Div);
}

#[divan::bench]
fn div_decimal_i128_nullable(bencher: Bencher) {
    let lhs = decimal_i128_nullable(0, 7, DECIMAL_MUL_DIV_LEN).into_array();
    let rhs = decimal_i128_nullable(1_000_000, 5, DECIMAL_MUL_DIV_LEN).into_array();

    bench_decimal(bencher, lhs, rhs, Operator::Div);
}

#[vortex_bench_support::cpu_features]
#[divan::bench]
fn eq_i64_constant(bencher: Bencher) {
    let lhs = primitive_nonnull(0, LEN).into_array();
    let rhs = ConstantArray::new(1024i64, LEN).into_array();

    bench_bool(bencher, lhs, rhs, Operator::Eq);
}

#[vortex_bench_support::cpu_features]
#[divan::bench]
fn lt_i64_nullable(bencher: Bencher) {
    let lhs = primitive_nullable(0, 7, LEN).into_array();
    let rhs = primitive_nullable(1_000_000, 5, LEN).into_array();

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

fn bench_decimal(bencher: Bencher, lhs: ArrayRef, rhs: ArrayRef, operator: Operator) {
    bench_binary::<DecimalArray>(bencher, lhs, rhs, operator);
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
    let len = lhs.len();

    bencher.counter(ItemsCount::new(len)).bench_local(|| {
        lhs.clone()
            .binary(rhs.clone(), operator)
            .unwrap()
            .execute::<T>(&mut ctx)
            .unwrap()
    });
}

const fn primitive_len<T>() -> usize {
    let input_len = MIN_PRIMITIVE_INPUT_BYTES / size_of::<T>();
    if input_len > MIN_PRIMITIVE_LEN {
        input_len
    } else {
        MIN_PRIMITIVE_LEN
    }
}

fn primitive_nonnull(base: i64, len: usize) -> PrimitiveArray {
    PrimitiveArray::from_iter((0..len as i64).map(|i| base + i))
}

fn decimal_i64_nonnull(base: i64, len: usize) -> DecimalArray {
    DecimalArray::from_iter::<i64, _>((0..len as i64).map(|i| base + i), DecimalDType::new(18, 2))
}

fn decimal_i128_nullable(base: i128, null_every: usize, len: usize) -> DecimalArray {
    DecimalArray::from_option_iter::<i128, _>(
        (0..len as i128).map(|i| (!(i as usize).is_multiple_of(null_every)).then_some(base + i)),
        DecimalDType::new(38, 2),
    )
}

fn primitive_small_nonnull(offset: i64, len: usize) -> PrimitiveArray {
    PrimitiveArray::from_iter((0..len as i64).map(|i| ((i + offset) % 1024) + 1))
}

fn primitive_i8_small_nonnull(offset: i8, len: usize) -> PrimitiveArray {
    PrimitiveArray::from_iter((0..len).map(|i| (((i as i32 + offset as i32) % 21) - 10) as i8))
}

fn primitive_u8_small_nonnull(offset: u8, len: usize) -> PrimitiveArray {
    PrimitiveArray::from_iter((0..len).map(|i| ((i + offset as usize) % 15 + 1) as u8))
}

fn primitive_i16_small_nonnull(offset: i16, len: usize) -> PrimitiveArray {
    PrimitiveArray::from_iter((0..len).map(|i| (((i as i32 + offset as i32) % 255) - 127) as i16))
}

fn primitive_u16_small_nonnull(offset: u16, len: usize) -> PrimitiveArray {
    PrimitiveArray::from_iter((0..len).map(|i| ((i + offset as usize) % 251 + 1) as u16))
}

fn primitive_i32_small_nonnull(offset: i32, len: usize) -> PrimitiveArray {
    PrimitiveArray::from_iter((0..len).map(|i| (((i as i64 + offset as i64) % 4096) - 2048) as i32))
}

fn primitive_u32_small_nonnull(offset: u32, len: usize) -> PrimitiveArray {
    PrimitiveArray::from_iter((0..len).map(|i| ((i + offset as usize) % 4096 + 1) as u32))
}

fn primitive_u64_small_nonnull(offset: u64, len: usize) -> PrimitiveArray {
    PrimitiveArray::from_iter((0..len).map(|i| ((i + offset as usize) % 4096 + 1) as u64))
}

fn primitive_nonzero(len: usize) -> PrimitiveArray {
    PrimitiveArray::from_iter((0..len as i64).map(|i| (i % 255) + 1))
}

fn primitive_nullable(base: i64, null_every: usize, len: usize) -> PrimitiveArray {
    PrimitiveArray::from_option_iter(
        (0..len as i64).map(|i| (!(i as usize).is_multiple_of(null_every)).then_some(base + i)),
    )
}

fn primitive_i32_small_nullable(offset: i32, null_every: usize, len: usize) -> PrimitiveArray {
    PrimitiveArray::from_option_iter((0..len).map(|i| {
        (!i.is_multiple_of(null_every))
            .then_some((((i as i64 + offset as i64) % 4096) - 2048) as i32)
    }))
}

fn bool_nullable(true_every: usize, null_every: usize) -> BoolArray {
    BoolArray::from_iter(
        (0..LEN).map(|i| (!i.is_multiple_of(null_every)).then_some(i.is_multiple_of(true_every))),
    )
}
