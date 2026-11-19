// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks for decimal-to-decimal casting across the two dimensions that drive its cost:
//!
//! - **validity**: non-nullable vs nullable inputs. Nullable inputs route the rescale through the
//!   masked kernel (`try_map_masked_into`) instead of the dense one (`try_map_into`).
//! - **work**: an *in-place* cast that only widens precision at the same scale, so the values
//!   buffer is reused untouched (`O(1)`), vs a *copy* cast that must allocate a new buffer and
//!   rescan every value (`O(n)`), vs a *widening* cast whose target precision has a wider
//!   canonical width than the input.

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use rand::prelude::*;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::DecimalArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::BigCast;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::NativeDecimalType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::optimizer::kernels::KernelSession;
use vortex_array::test_harness::mark_optimized;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
use vortex_session::VortexSession;

// The copy casts allocate their output buffer inside the timed region, so route allocation
// through vendored mimalloc to keep glibc malloc (which varies across runner images) out of
// the measured trace.
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

// Kept small enough to stay in L2 so the kernel cost shows up rather than DRAM bandwidth,
// and to keep the CodSpeed simulation under 1ms per benchmark.
const SIZES: &[usize] = &[16_384];

// Casts that divide every value by the scale factor cost several times more per value under
// CodSpeed's simulation, so they run on half as many values to stay well under 1ms.
const DIVIDING_SIZES: &[usize] = &[8_192];

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<KernelSession>());

/// Builds a `T`-backed `Decimal(precision, scale)` array of `n` values that are all below
/// `10^9`, optionally with ~50% nulls.
fn decimal_array<T: NativeDecimalType>(
    n: usize,
    precision: u8,
    scale: i8,
    nullable: bool,
) -> ArrayRef {
    let mut rng = StdRng::seed_from_u64(42);
    let values: BufferMut<T> = (0..n)
        .map(|_| <T as BigCast>::from(rng.random_range(0..1_000_000_000i64)).unwrap())
        .collect();
    let validity = if nullable {
        Validity::from_iter((0..n).map(|_| rng.random_bool(0.5)))
    } else {
        Validity::NonNullable
    };
    DecimalArray::new(
        values.freeze(),
        DecimalDType::new(precision, scale),
        validity,
        &mut SESSION.create_execution_ctx(),
    )
    .into_array()
}

/// Casts `array` to `target` and forces execution, the unit measured by every bench below.
fn bench_cast(bencher: Bencher, array: ArrayRef, target: DType) {
    mark_optimized(&array, &SESSION).unwrap();

    bencher
        .with_inputs(|| (array.clone(), SESSION.create_execution_ctx()))
        .bench_refs(|(a, ctx)| a.cast(target.clone()).unwrap().execute::<Canonical>(ctx));
}

// In place: widening precision at the same scale keeps the physical type, so the cast reuses the
// values buffer untouched.

#[divan::bench(args = SIZES)]
fn in_place_non_nullable(bencher: Bencher, n: usize) {
    bench_cast(
        bencher,
        decimal_array::<i64>(n, 10, 2, false),
        DType::Decimal(DecimalDType::new(18, 2), Nullability::NonNullable),
    );
}

#[divan::bench(args = SIZES)]
fn in_place_nullable(bencher: Bencher, n: usize) {
    bench_cast(
        bencher,
        decimal_array::<i64>(n, 10, 2, true),
        DType::Decimal(DecimalDType::new(18, 2), Nullability::Nullable),
    );
}

// Copy: narrowing precision at the same scale cannot reuse the buffer; every value is re-scanned
// and range-checked into a freshly allocated buffer.

#[divan::bench(args = SIZES)]
fn copy_non_nullable(bencher: Bencher, n: usize) {
    bench_cast(
        bencher,
        decimal_array::<i64>(n, 18, 2, false),
        DType::Decimal(DecimalDType::new(10, 2), Nullability::NonNullable),
    );
}

#[divan::bench(args = SIZES)]
fn copy_nullable(bencher: Bencher, n: usize) {
    bench_cast(
        bencher,
        decimal_array::<i64>(n, 18, 2, true),
        DType::Decimal(DecimalDType::new(10, 2), Nullability::Nullable),
    );
}

// Widening: the target precision has a wider canonical width than the input, so every value is
// copied into a buffer of the wider type.

#[divan::bench(args = SIZES)]
fn widen_non_nullable(bencher: Bencher, n: usize) {
    bench_cast(
        bencher,
        decimal_array::<i32>(n, 9, 2, false),
        DType::Decimal(DecimalDType::new(18, 2), Nullability::NonNullable),
    );
}

#[divan::bench(args = SIZES)]
fn widen_nullable(bencher: Bencher, n: usize) {
    bench_cast(
        bencher,
        decimal_array::<i32>(n, 9, 2, true),
        DType::Decimal(DecimalDType::new(18, 2), Nullability::Nullable),
    );
}

/// Builds a `T`-backed `Decimal(precision, scale)` array of `n` multiples of `step` below `10^9`,
/// with ~50% nulls.
fn decimal_multiples<T: NativeDecimalType>(
    n: usize,
    precision: u8,
    scale: i8,
    step: i64,
) -> ArrayRef {
    let mut rng = StdRng::seed_from_u64(42);
    let values: BufferMut<T> = (0..n)
        .map(|_| <T as BigCast>::from(rng.random_range(0..1_000_000_000i64 / step) * step).unwrap())
        .collect();
    let validity = Validity::from_iter((0..n).map(|_| rng.random_bool(0.5)));
    DecimalArray::new(
        values.freeze(),
        DecimalDType::new(precision, scale),
        validity,
        &mut SESSION.create_execution_ctx(),
    )
    .into_array()
}

// Narrowing: the target precision has a narrower canonical width, so every value is range-checked
// and copied into a buffer of the narrower type.

#[divan::bench(args = SIZES)]
fn narrow_i64_to_i32_nullable(bencher: Bencher, n: usize) {
    bench_cast(
        bencher,
        decimal_array::<i64>(n, 18, 2, true),
        DType::Decimal(DecimalDType::new(9, 2), Nullability::Nullable),
    );
}

// Rescaling: changing the scale multiplies or exactly divides every value by a power of ten.

#[divan::bench(args = SIZES)]
fn scale_up_i32_to_i64_nullable(bencher: Bencher, n: usize) {
    bench_cast(
        bencher,
        decimal_array::<i32>(n, 9, 2, true),
        DType::Decimal(DecimalDType::new(18, 4), Nullability::Nullable),
    );
}

#[divan::bench(args = DIVIDING_SIZES)]
fn scale_down_i64_nullable(bencher: Bencher, n: usize) {
    bench_cast(
        bencher,
        decimal_multiples::<i64>(n, 18, 4, 100),
        DType::Decimal(DecimalDType::new(16, 2), Nullability::Nullable),
    );
}

// To integer: values are divided by the scale factor, truncating toward zero, and narrowed.

#[divan::bench(args = DIVIDING_SIZES)]
fn to_i32_nullable(bencher: Bencher, n: usize) {
    bench_cast(
        bencher,
        decimal_array::<i64>(n, 18, 2, true),
        DType::Primitive(PType::I32, Nullability::Nullable),
    );
}
