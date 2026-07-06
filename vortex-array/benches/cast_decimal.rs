// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks for decimal-to-decimal casting across the two dimensions that drive its cost:
//!
//! - **validity**: non-nullable vs nullable inputs. Nullable inputs route the rescale through the
//!   masked kernel (`try_map_masked_into`) instead of the dense one (`try_map_into`).
//! - **work**: an *in-place* cast that only widens precision at the same scale, so the values
//!   buffer is reused untouched (`O(1)`), vs a *copy* cast that must allocate a new buffer and
//!   rescan every value (`O(n)`).

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use rand::prelude::*;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::DecimalArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::Nullability;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
use vortex_session::VortexSession;

fn main() {
    divan::main();
}

// Kept small enough to stay in L2 so the kernel cost shows up rather than DRAM bandwidth.
const SIZES: &[usize] = &[65_536];

static SESSION: LazyLock<VortexSession> = LazyLock::new(vortex_array::array_session);

/// Builds an `i64`-backed `Decimal(precision, scale)` array of `n` values that all fit in
/// precision 10, optionally with ~50% nulls.
fn decimal_array(n: usize, precision: u8, scale: i8, nullable: bool) -> ArrayRef {
    let mut rng = StdRng::seed_from_u64(42);
    let values: BufferMut<i64> = (0..n)
        .map(|_| rng.random_range(0..1_000_000_000i64))
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
    )
    .into_array()
}

/// Casts `array` to `target` and forces execution, the unit measured by every bench below.
fn bench_cast(bencher: Bencher, array: ArrayRef, target: DType) {
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
        decimal_array(n, 10, 2, false),
        DType::Decimal(DecimalDType::new(18, 2), Nullability::NonNullable),
    );
}

#[divan::bench(args = SIZES)]
fn in_place_nullable(bencher: Bencher, n: usize) {
    bench_cast(
        bencher,
        decimal_array(n, 10, 2, true),
        DType::Decimal(DecimalDType::new(18, 2), Nullability::Nullable),
    );
}

// Copy: narrowing precision at the same scale cannot reuse the buffer; every value is re-scanned
// and range-checked into a freshly allocated buffer.

#[divan::bench(args = SIZES)]
fn copy_non_nullable(bencher: Bencher, n: usize) {
    bench_cast(
        bencher,
        decimal_array(n, 18, 2, false),
        DType::Decimal(DecimalDType::new(10, 2), Nullability::NonNullable),
    );
}

#[divan::bench(args = SIZES)]
fn copy_nullable(bencher: Bencher, n: usize) {
    bench_cast(
        bencher,
        decimal_array(n, 18, 2, true),
        DType::Decimal(DecimalDType::new(10, 2), Nullability::Nullable),
    );
}
