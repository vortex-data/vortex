// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Decimal Add/Sub and comparison over the native decimal kernels.
//!
//! Three costs are isolated here:
//!
//! * `checked_*` — the `i256` overflow test and the result-precision bounds test, benchmarked over
//!   contiguous buffers without the surrounding compute stack.
//! * `add_*` / `sub_*` — the same work through the kernel, at each precision. The `p37` / `p38`
//!   pair covers the working width chosen for the result precision: both are stored as `i128`, so
//!   any difference between them is the width the result dtype forces the lane loop into.
//! * `compare_*` — comparison of operands whose storage widths differ, which is the normal case
//!   because compressed chunks are narrowed independently of each other.

#![expect(clippy::unwrap_used)]
#![expect(
    clippy::cast_possible_truncation,
    reason = "benchmark fixtures use indices that fit in the chosen widths"
)]

use std::sync::LazyLock;

use divan::Bencher;
use divan::counter::ItemsCount;
use mimalloc::MiMalloc;
use num_traits::CheckedAdd;
use num_traits::WrappingSub;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::Executable;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::DecimalArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::NativeDecimalType;
use vortex_array::dtype::i256;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_buffer::Buffer;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(array_session);

const LEN: usize = 65_536;

// The overflow and bounds tests on their own, over contiguous i256 buffers.

#[divan::bench]
fn checked_add_i256(bencher: Bencher) {
    let (lhs, rhs) = i256_operands();

    bencher.counter(ItemsCount::new(LEN)).bench_local(|| {
        let mut sum = 0usize;
        for (a, b) in lhs.iter().zip(rhs.iter()) {
            sum += a.checked_add(b).is_some() as usize;
        }
        sum
    });
}

#[divan::bench]
fn checked_add_bounds_i256(bencher: Bencher) {
    let (lhs, rhs) = i256_operands();
    let lower = <i256 as NativeDecimalType>::MIN_BY_PRECISION[76];
    let upper = <i256 as NativeDecimalType>::MAX_BY_PRECISION[76];

    bencher.counter(ItemsCount::new(LEN)).bench_local(|| {
        let mut sum = 0usize;
        for (a, b) in lhs.iter().zip(rhs.iter()) {
            sum += a.checked_add(b).is_some_and(|v| lower <= v && v <= upper) as usize;
        }
        sum
    });
}

/// The fused form the kernel applies: overflow read off the high-limb sign bits, and both
/// precision bounds folded into a single unsigned comparison. Spelled out here so the three
/// `checked_*` benchmarks can be compared against each other from any revision.
#[divan::bench]
fn overflowing_add_bounds_i256(bencher: Bencher) {
    let (lhs, rhs) = i256_operands();
    let lower = <i256 as NativeDecimalType>::MIN_BY_PRECISION[76];
    let span = <i256 as WrappingSub>::wrapping_sub(
        &<i256 as NativeDecimalType>::MAX_BY_PRECISION[76],
        &lower,
    );

    #[expect(clippy::cast_sign_loss, reason = "reinterpreting the bit pattern")]
    fn le_unsigned(lhs: i256, rhs: i256) -> bool {
        let (low, high) = lhs.to_parts();
        let (rhs_low, rhs_high) = rhs.to_parts();
        (high as u128, low) <= (rhs_high as u128, rhs_low)
    }

    fn overflowing_add(lhs: i256, rhs: i256) -> (i256, bool) {
        let sum = lhs.wrapping_add(rhs);
        let (_, lhs_high) = lhs.to_parts();
        let (_, rhs_high) = rhs.to_parts();
        let (_, sum_high) = sum.to_parts();
        (sum, ((lhs_high ^ sum_high) & (rhs_high ^ sum_high)) < 0)
    }

    bencher.counter(ItemsCount::new(LEN)).bench_local(|| {
        let mut sum = 0usize;
        for (a, b) in lhs.iter().zip(rhs.iter()) {
            let (value, overflow) = overflowing_add(*a, *b);
            sum += (!overflow
                & le_unsigned(<i256 as WrappingSub>::wrapping_sub(&value, &lower), span))
                as usize;
        }
        sum
    });
}

// Add/Sub through the kernel.

#[divan::bench]
fn add_i64_p18(bencher: Bencher) {
    let lhs = decimal_i64(18, 0).into_array();
    let rhs = decimal_i64(18, 1).into_array();

    bench_decimal(bencher, lhs, rhs, Operator::Add);
}

#[divan::bench]
fn add_i128_p37(bencher: Bencher) {
    let lhs = decimal_i128(37, 0).into_array();
    let rhs = decimal_i128(37, 1).into_array();

    bench_decimal(bencher, lhs, rhs, Operator::Add);
}

#[divan::bench]
fn add_i128_p38(bencher: Bencher) {
    let lhs = decimal_i128(38, 0).into_array();
    let rhs = decimal_i128(38, 1).into_array();

    bench_decimal(bencher, lhs, rhs, Operator::Add);
}

#[divan::bench]
fn add_i256_p76(bencher: Bencher) {
    let lhs = decimal_i256(0).into_array();
    let rhs = decimal_i256(1).into_array();

    bench_decimal(bencher, lhs, rhs, Operator::Add);
}

#[divan::bench]
fn sub_i256_p76(bencher: Bencher) {
    let lhs = decimal_i256(0).into_array();
    let rhs = decimal_i256(1).into_array();

    bench_decimal(bencher, lhs, rhs, Operator::Sub);
}

#[divan::bench]
fn add_i256_p76_nullable(bencher: Bencher) {
    let lhs = decimal_i256_nullable(0, 7).into_array();
    let rhs = decimal_i256_nullable(1, 5).into_array();

    bench_decimal(bencher, lhs, rhs, Operator::Add);
}

// Comparison, same and mixed storage widths.

#[divan::bench]
fn compare_i128_i128_p38(bencher: Bencher) {
    let lhs = decimal_i128(38, 0).into_array();
    let rhs = decimal_i128(38, 1).into_array();

    bench_compare(bencher, lhs, rhs, Operator::Gte);
}

#[divan::bench]
fn compare_i32_i128_p38(bencher: Bencher) {
    let lhs = decimal_i32(38, 0).into_array();
    let rhs = decimal_i128(38, 1).into_array();

    bench_compare(bencher, lhs, rhs, Operator::Gte);
}

#[divan::bench]
fn compare_i128_i32_p38(bencher: Bencher) {
    let lhs = decimal_i128(38, 0).into_array();
    let rhs = decimal_i32(38, 1).into_array();

    bench_compare(bencher, lhs, rhs, Operator::Gte);
}

#[divan::bench]
fn compare_i64_i256_p76(bencher: Bencher) {
    let lhs = decimal_i64(76, 0).into_array();
    let rhs = decimal_i256(1).into_array();

    bench_compare(bencher, lhs, rhs, Operator::Gte);
}

fn bench_decimal(bencher: Bencher, lhs: ArrayRef, rhs: ArrayRef, operator: Operator) {
    bench_binary::<DecimalArray>(bencher, lhs, rhs, operator);
}

fn bench_compare(bencher: Bencher, lhs: ArrayRef, rhs: ArrayRef, operator: Operator) {
    bench_binary::<Canonical>(bencher, lhs, rhs, operator);
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

/// Two operand buffers whose sums stay well inside precision 76, so the benchmark measures the
/// checks themselves rather than an early exit out of the lane loop.
fn i256_operands() -> (Buffer<i256>, Buffer<i256>) {
    let scale = i256::from_i128(10).wrapping_pow(60);
    let lhs = (0..LEN as i128)
        .map(|i| i256::from_i128(i % 977) * scale)
        .collect();
    let rhs = (0..LEN as i128)
        .map(|i| i256::from_i128(i % 599) * scale)
        .collect();
    (lhs, rhs)
}

fn decimal_i32(precision: u8, offset: i32) -> DecimalArray {
    DecimalArray::from_iter::<i32, _>(
        (0..LEN as i32).map(|i| (i + offset) % 1_000_000),
        DecimalDType::new(precision, 2),
    )
}

fn decimal_i64(precision: u8, offset: i64) -> DecimalArray {
    DecimalArray::from_iter::<i64, _>(
        (0..LEN as i64).map(|i| (i + offset) % 1_000_000_000),
        DecimalDType::new(precision, 2),
    )
}

/// Values wide enough to occupy most of the declared precision, so a narrower working width could
/// not represent them.
fn decimal_i128(precision: u8, offset: i128) -> DecimalArray {
    let base = <i128 as NativeDecimalType>::MAX_BY_PRECISION[precision as usize] / 4;
    DecimalArray::from_iter::<i128, _>(
        (0..LEN as i128).map(|i| base + ((i + offset) % 1_000_000)),
        DecimalDType::new(precision, 2),
    )
}

fn decimal_i256(offset: i128) -> DecimalArray {
    let base = <i256 as NativeDecimalType>::MAX_BY_PRECISION[76] / i256::from_i128(4);
    DecimalArray::from_iter::<i256, _>(
        (0..LEN as i128).map(|i| base + i256::from_i128((i + offset) % 1_000_000)),
        DecimalDType::new(76, 2),
    )
}

fn decimal_i256_nullable(offset: i128, null_every: usize) -> DecimalArray {
    let base = <i256 as NativeDecimalType>::MAX_BY_PRECISION[76] / i256::from_i128(4);
    DecimalArray::from_option_iter::<i256, _>(
        (0..LEN as i128).map(|i| {
            (!(i as usize).is_multiple_of(null_every))
                .then(|| base + i256::from_i128((i + offset) % 1_000_000))
        }),
        DecimalDType::new(76, 2),
    )
}
