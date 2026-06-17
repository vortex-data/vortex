// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use rand::prelude::*;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::expr::stats::Stat;
use vortex_session::VortexSession;

fn main() {
    divan::main();
}

// Sizes used for the fallible-path benches below. Kept small enough to fit in L2 so
// the kernel cost shows up clearly rather than being hidden by DRAM bandwidth.
const SIZES: &[usize] = &[65_536];

static SESSION: LazyLock<VortexSession> = LazyLock::new(vortex_array::array_session);

#[divan::bench(args = SIZES)]
fn cast_u16_to_u32(bencher: Bencher, n: usize) {
    let mut rng = StdRng::seed_from_u64(42);
    let arr = PrimitiveArray::from_option_iter((0..n).map(|i| {
        #[expect(clippy::cast_possible_truncation)]
        rng.random_bool(0.5).then_some(i as u16)
    }))
    .into_array();
    // Pre-compute min/max so values_fit_in is a cache hit during the benchmark.
    arr.statistics()
        .compute_all(&[Stat::Min, Stat::Max], &mut SESSION.create_execution_ctx())
        .ok();
    bencher
        .with_inputs(|| (arr.clone(), SESSION.create_execution_ctx()))
        .bench_refs(|(a, ctx)| {
            a.cast(DType::Primitive(PType::U32, Nullability::Nullable))
                .unwrap()
                .execute::<Canonical>(ctx)
        });
}

/// Narrowing fallible cast that goes through `try_map_with_mask`. Inputs are bounded
/// so every value fits, isolating the kernel's per-lane checked-cast overhead.
#[divan::bench(args = SIZES)]
fn cast_u32_to_u8(bencher: Bencher, n: usize) {
    let mut rng = StdRng::seed_from_u64(42);
    let arr = PrimitiveArray::from_option_iter((0..n).map(|_| {
        rng.random_bool(0.7)
            .then(|| rng.random_range(0..u8::MAX) as u32)
    }))
    .into_array();
    bencher
        .with_inputs(|| (arr.clone(), SESSION.create_execution_ctx()))
        .bench_refs(|(a, ctx)| {
            a.cast(DType::Primitive(PType::U8, Nullability::Nullable))
                .unwrap()
                .execute::<Canonical>(ctx)
        });
}

/// Sign-change cast i32 → u32. Values are non-negative so the kernel succeeds
/// but still pays the per-lane `try_from` check.
#[divan::bench(args = SIZES)]
fn cast_i32_to_u32(bencher: Bencher, n: usize) {
    let mut rng = StdRng::seed_from_u64(42);
    let arr = PrimitiveArray::from_option_iter(
        (0..n).map(|_| rng.random_bool(0.7).then(|| rng.random_range(0..i32::MAX))),
    )
    .into_array();
    bencher
        .with_inputs(|| (arr.clone(), SESSION.create_execution_ctx()))
        .bench_refs(|(a, ctx)| {
            a.cast(DType::Primitive(PType::U32, Nullability::Nullable))
                .unwrap()
                .execute::<Canonical>(ctx)
        });
}
