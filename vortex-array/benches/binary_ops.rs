// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]
#![expect(
    clippy::cast_possible_truncation,
    reason = "benchmark fixtures use indices that fit in the chosen widths"
)]

use std::sync::LazyLock;

use arrow_arith::numeric::div as arrow_div;
use arrow_arith::numeric::mul as arrow_mul;
use arrow_array::ArrowPrimitiveType;
use arrow_array::PrimitiveArray as ArrowPrimitiveArray;
use arrow_array::types::Decimal32Type;
use arrow_array::types::Decimal64Type;
use arrow_array::types::Decimal128Type;
use arrow_array::types::Decimal256Type;
use arrow_array::types::DecimalType as ArrowDecimalType;
use arrow_buffer::ArrowNativeType;
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
use vortex_array::dtype::BigCast;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::NativeDecimalType;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(array_session);

const LEN: usize = 32_768;

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
fn add_i64_constant(bencher: Bencher) {
    let lhs = primitive_nonnull(0).into_array();
    let rhs = ConstantArray::new(1_000_000i64, LEN).into_array();

    bench_primitive(bencher, lhs, rhs, Operator::Add);
}

#[divan::bench]
fn add_i32_nonnull(bencher: Bencher) {
    let lhs = primitive_i32_small_nonnull(1).into_array();
    let rhs = primitive_i32_small_nonnull(17).into_array();

    bench_primitive(bencher, lhs, rhs, Operator::Add);
}

#[divan::bench]
fn add_u32_nonnull(bencher: Bencher) {
    let lhs = primitive_u32_small_nonnull(1).into_array();
    let rhs = primitive_u32_small_nonnull(17).into_array();

    bench_primitive(bencher, lhs, rhs, Operator::Add);
}

#[divan::bench]
fn mul_i64_nonnull(bencher: Bencher) {
    let lhs = primitive_small_nonnull(1).into_array();
    let rhs = primitive_small_nonnull(17).into_array();

    bench_primitive(bencher, lhs, rhs, Operator::Mul);
}

#[divan::bench]
fn mul_i8_nonnull(bencher: Bencher) {
    let lhs = primitive_i8_small_nonnull(1).into_array();
    let rhs = primitive_i8_small_nonnull(7).into_array();

    bench_primitive(bencher, lhs, rhs, Operator::Mul);
}

#[divan::bench]
fn mul_u8_nonnull(bencher: Bencher) {
    let lhs = primitive_u8_small_nonnull(1).into_array();
    let rhs = primitive_u8_small_nonnull(7).into_array();

    bench_primitive(bencher, lhs, rhs, Operator::Mul);
}

#[divan::bench]
fn mul_i16_nonnull(bencher: Bencher) {
    let lhs = primitive_i16_small_nonnull(1).into_array();
    let rhs = primitive_i16_small_nonnull(17).into_array();

    bench_primitive(bencher, lhs, rhs, Operator::Mul);
}

#[divan::bench]
fn mul_u16_nonnull(bencher: Bencher) {
    let lhs = primitive_u16_small_nonnull(1).into_array();
    let rhs = primitive_u16_small_nonnull(17).into_array();

    bench_primitive(bencher, lhs, rhs, Operator::Mul);
}

#[divan::bench]
fn mul_i32_nonnull(bencher: Bencher) {
    let lhs = primitive_i32_small_nonnull(1).into_array();
    let rhs = primitive_i32_small_nonnull(17).into_array();

    bench_primitive(bencher, lhs, rhs, Operator::Mul);
}

#[divan::bench]
fn mul_u32_nonnull(bencher: Bencher) {
    let lhs = primitive_u32_small_nonnull(1).into_array();
    let rhs = primitive_u32_small_nonnull(17).into_array();

    bench_primitive(bencher, lhs, rhs, Operator::Mul);
}

#[divan::bench]
fn mul_u64_nonnull(bencher: Bencher) {
    let lhs = primitive_u64_small_nonnull(1).into_array();
    let rhs = primitive_u64_small_nonnull(17).into_array();

    bench_primitive(bencher, lhs, rhs, Operator::Mul);
}

#[divan::bench]
fn mul_i32_nullable(bencher: Bencher) {
    let lhs = primitive_i32_small_nullable(1, 7).into_array();
    let rhs = primitive_i32_small_nullable(17, 5).into_array();

    bench_primitive(bencher, lhs, rhs, Operator::Mul);
}

#[divan::bench]
fn mul_i32_constant(bencher: Bencher) {
    let lhs = primitive_i32_small_nonnull(1).into_array();
    let rhs = ConstantArray::new(31i32, LEN).into_array();

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
fn add_decimal_i64_nonnull(bencher: Bencher) {
    let lhs = decimal_i64_nonnull(0).into_array();
    let rhs = decimal_i64_nonnull(1_000_000).into_array();

    bench_decimal(bencher, lhs, rhs, Operator::Add);
}

#[divan::bench]
fn add_decimal_i128_nullable(bencher: Bencher) {
    let lhs = decimal_i128_nullable(0, 7).into_array();
    let rhs = decimal_i128_nullable(1_000_000, 5).into_array();

    bench_decimal(bencher, lhs, rhs, Operator::Add);
}

macro_rules! decimal_compare_benches {
    (
        $vortex_mul:ident,
        $arrow_mul:ident,
        $vortex_div:ident,
        $arrow_div:ident,
        $vortex_type:ty,
        $arrow_type:ty,
        $precision:expr,
        $scale:expr
    ) => {
        #[divan::bench]
        fn $vortex_mul(bencher: Bencher) {
            let dtype = DecimalDType::new($precision, $scale);
            let lhs = comparison_vortex_decimal::<$vortex_type>(dtype, 1).into_array();
            let rhs = comparison_vortex_decimal::<$vortex_type>(dtype, 17).into_array();
            bench_decimal(bencher, lhs, rhs, Operator::Mul);
        }

        #[divan::bench]
        fn $arrow_mul(bencher: Bencher) {
            let lhs = comparison_arrow_decimal::<$arrow_type>($precision, $scale, 1);
            let rhs = comparison_arrow_decimal::<$arrow_type>($precision, $scale, 17);
            bench_arrow_decimal(bencher, lhs, rhs, ArrowDecimalOp::Mul);
        }

        #[divan::bench]
        fn $vortex_div(bencher: Bencher) {
            let dtype = DecimalDType::new($precision, $scale);
            let lhs = comparison_vortex_decimal::<$vortex_type>(dtype, 1).into_array();
            let rhs = comparison_vortex_decimal::<$vortex_type>(dtype, 17).into_array();
            bench_decimal(bencher, lhs, rhs, Operator::Div);
        }

        #[divan::bench]
        fn $arrow_div(bencher: Bencher) {
            let lhs = comparison_arrow_decimal::<$arrow_type>($precision, $scale, 1);
            let rhs = comparison_arrow_decimal::<$arrow_type>($precision, $scale, 17);
            bench_arrow_decimal(bencher, lhs, rhs, ArrowDecimalOp::Div);
        }
    };
}

decimal_compare_benches!(
    vortex_mul_decimal_i32_nonnull,
    arrow_mul_decimal_i32_nonnull,
    vortex_div_decimal_i32_nonnull,
    arrow_div_decimal_i32_nonnull,
    i32,
    Decimal32Type,
    4,
    1
);
decimal_compare_benches!(
    vortex_mul_decimal_i64_nonnull,
    arrow_mul_decimal_i64_nonnull,
    vortex_div_decimal_i64_nonnull,
    arrow_div_decimal_i64_nonnull,
    i64,
    Decimal64Type,
    8,
    2
);
decimal_compare_benches!(
    vortex_mul_decimal_i128_nonnull,
    arrow_mul_decimal_i128_nonnull,
    vortex_div_decimal_i128_nonnull,
    arrow_div_decimal_i128_nonnull,
    i128,
    Decimal128Type,
    18,
    2
);
decimal_compare_benches!(
    vortex_mul_decimal_i256_nonnull,
    arrow_mul_decimal_i256_nonnull,
    vortex_div_decimal_i256_nonnull,
    arrow_div_decimal_i256_nonnull,
    vortex_array::dtype::i256,
    Decimal256Type,
    38,
    2
);

#[divan::bench]
fn vortex_mul_decimal_i256_nullable(bencher: Bencher) {
    let dtype = DecimalDType::new(38, 2);
    let lhs =
        comparison_vortex_decimal_nullable::<vortex_array::dtype::i256>(dtype, 1, 7).into_array();
    let rhs =
        comparison_vortex_decimal_nullable::<vortex_array::dtype::i256>(dtype, 17, 5).into_array();
    bench_decimal(bencher, lhs, rhs, Operator::Mul);
}

#[divan::bench]
fn arrow_mul_decimal_i256_nullable(bencher: Bencher) {
    let lhs = comparison_arrow_decimal_nullable::<Decimal256Type>(38, 2, 1, 7);
    let rhs = comparison_arrow_decimal_nullable::<Decimal256Type>(38, 2, 17, 5);
    bench_arrow_decimal(bencher, lhs, rhs, ArrowDecimalOp::Mul);
}

#[divan::bench]
fn vortex_div_decimal_i256_nullable(bencher: Bencher) {
    let dtype = DecimalDType::new(38, 2);
    let lhs =
        comparison_vortex_decimal_nullable::<vortex_array::dtype::i256>(dtype, 1, 7).into_array();
    let rhs =
        comparison_vortex_decimal_nullable::<vortex_array::dtype::i256>(dtype, 17, 5).into_array();
    bench_decimal(bencher, lhs, rhs, Operator::Div);
}

#[divan::bench]
fn arrow_div_decimal_i256_nullable(bencher: Bencher) {
    let lhs = comparison_arrow_decimal_nullable::<Decimal256Type>(38, 2, 1, 7);
    let rhs = comparison_arrow_decimal_nullable::<Decimal256Type>(38, 2, 17, 5);
    bench_arrow_decimal(bencher, lhs, rhs, ArrowDecimalOp::Div);
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

fn bench_decimal(bencher: Bencher, lhs: ArrayRef, rhs: ArrayRef, operator: Operator) {
    bench_binary::<DecimalArray>(bencher, lhs, rhs, operator);
}

#[derive(Clone, Copy)]
enum ArrowDecimalOp {
    Mul,
    Div,
}

fn bench_arrow_decimal<T>(
    bencher: Bencher,
    lhs: ArrowPrimitiveArray<T>,
    rhs: ArrowPrimitiveArray<T>,
    op: ArrowDecimalOp,
) where
    T: ArrowPrimitiveType,
{
    bencher
        .counter(ItemsCount::new(LEN))
        .bench_local(|| match op {
            ArrowDecimalOp::Mul => arrow_mul(&lhs, &rhs).unwrap(),
            ArrowDecimalOp::Div => arrow_div(&lhs, &rhs).unwrap(),
        });
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

fn decimal_i64_nonnull(base: i64) -> DecimalArray {
    DecimalArray::from_iter::<i64, _>((0..LEN as i64).map(|i| base + i), DecimalDType::new(18, 2))
}

fn decimal_i128_nullable(base: i128, null_every: usize) -> DecimalArray {
    DecimalArray::from_option_iter::<i128, _>(
        (0..LEN as i128).map(|i| (!(i as usize).is_multiple_of(null_every)).then_some(base + i)),
        DecimalDType::new(38, 2),
    )
}

fn comparison_vortex_decimal<T>(dtype: DecimalDType, offset: usize) -> DecimalArray
where
    T: NativeDecimalType,
{
    DecimalArray::from_iter::<T, _>(
        (0..LEN).map(|idx| <T as BigCast>::from(((idx + offset) % 89 + 1) as i64).unwrap()),
        dtype,
    )
}

fn comparison_vortex_decimal_nullable<T>(
    dtype: DecimalDType,
    offset: usize,
    null_every: usize,
) -> DecimalArray
where
    T: NativeDecimalType,
{
    DecimalArray::from_option_iter::<T, _>(
        (0..LEN).map(|idx| {
            (!idx.is_multiple_of(null_every))
                .then(|| <T as BigCast>::from(((idx + offset) % 89 + 1) as i64).unwrap())
        }),
        dtype,
    )
}

fn comparison_arrow_decimal<T>(precision: u8, scale: i8, offset: usize) -> ArrowPrimitiveArray<T>
where
    T: ArrowPrimitiveType + ArrowDecimalType,
{
    ArrowPrimitiveArray::<T>::from_iter_values(
        (0..LEN).map(|idx| T::Native::usize_as((idx + offset) % 89 + 1)),
    )
    .with_precision_and_scale(precision, scale)
    .unwrap()
}

fn comparison_arrow_decimal_nullable<T>(
    precision: u8,
    scale: i8,
    offset: usize,
    null_every: usize,
) -> ArrowPrimitiveArray<T>
where
    T: ArrowPrimitiveType + ArrowDecimalType,
{
    ArrowPrimitiveArray::<T>::from_iter((0..LEN).map(|idx| {
        (!idx.is_multiple_of(null_every)).then(|| T::Native::usize_as((idx + offset) % 89 + 1))
    }))
    .with_precision_and_scale(precision, scale)
    .unwrap()
}

fn primitive_small_nonnull(offset: i64) -> PrimitiveArray {
    PrimitiveArray::from_iter((0..LEN as i64).map(|i| ((i + offset) % 1024) + 1))
}

fn primitive_i8_small_nonnull(offset: i8) -> PrimitiveArray {
    PrimitiveArray::from_iter((0..LEN).map(|i| (((i as i32 + offset as i32) % 21) - 10) as i8))
}

fn primitive_u8_small_nonnull(offset: u8) -> PrimitiveArray {
    PrimitiveArray::from_iter((0..LEN).map(|i| ((i + offset as usize) % 15 + 1) as u8))
}

fn primitive_i16_small_nonnull(offset: i16) -> PrimitiveArray {
    PrimitiveArray::from_iter((0..LEN).map(|i| (((i as i32 + offset as i32) % 255) - 127) as i16))
}

fn primitive_u16_small_nonnull(offset: u16) -> PrimitiveArray {
    PrimitiveArray::from_iter((0..LEN).map(|i| ((i + offset as usize) % 251 + 1) as u16))
}

fn primitive_i32_small_nonnull(offset: i32) -> PrimitiveArray {
    PrimitiveArray::from_iter((0..LEN).map(|i| (((i as i64 + offset as i64) % 4096) - 2048) as i32))
}

fn primitive_u32_small_nonnull(offset: u32) -> PrimitiveArray {
    PrimitiveArray::from_iter((0..LEN).map(|i| ((i + offset as usize) % 4096 + 1) as u32))
}

fn primitive_u64_small_nonnull(offset: u64) -> PrimitiveArray {
    PrimitiveArray::from_iter((0..LEN).map(|i| ((i + offset as usize) % 4096 + 1) as u64))
}

fn primitive_nonzero() -> PrimitiveArray {
    PrimitiveArray::from_iter((0..LEN as i64).map(|i| (i % 255) + 1))
}

fn primitive_nullable(base: i64, null_every: usize) -> PrimitiveArray {
    PrimitiveArray::from_option_iter(
        (0..LEN as i64).map(|i| (!(i as usize).is_multiple_of(null_every)).then_some(base + i)),
    )
}

fn primitive_i32_small_nullable(offset: i32, null_every: usize) -> PrimitiveArray {
    PrimitiveArray::from_option_iter((0..LEN).map(|i| {
        (!i.is_multiple_of(null_every))
            .then_some((((i as i64 + offset as i64) % 4096) - 2048) as i32)
    }))
}

fn bool_nullable(true_every: usize, null_every: usize) -> BoolArray {
    BoolArray::from_iter(
        (0..LEN).map(|i| (!i.is_multiple_of(null_every)).then_some(i.is_multiple_of(true_every))),
    )
}
