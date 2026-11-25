// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation)]

use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::NativePType;
use vortex_array::session::ArraySession;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

const LEN: usize = 65_536;

/// `(top_fraction, null_fraction)`: each value is the top value with probability
/// `top_fraction`, and otherwise drawn uniformly from a wide range.
const ARGS: &[(f64, f64)] = &[
    (0.95, 0.0),
    (0.6, 0.0),
    (0.3, 0.0),
    (0.0, 0.0),
    (0.95, 0.1),
    (0.3, 0.1),
];

fn make_array<T: NativePType>(
    (top_fraction, null_fraction): (f64, f64),
    from_u32: impl Fn(u32) -> T,
) -> PrimitiveArray {
    let mut rng = StdRng::seed_from_u64(0);
    let values: Buffer<T> = (0..LEN)
        .map(|_| {
            if rng.random_bool(top_fraction) {
                from_u32(7)
            } else {
                from_u32(rng.random_range(0..1_000_000))
            }
        })
        .collect();
    let validity = if null_fraction == 0.0 {
        Validity::NonNullable
    } else {
        Validity::from_iter((0..LEN).map(|_| !rng.random_bool(null_fraction)))
    };
    PrimitiveArray::new(values, validity)
}

#[divan::bench(args = ARGS)]
fn top_value_u32(bencher: Bencher, args: (f64, f64)) {
    let array = make_array::<u32>(args, |v| v);
    bencher
        .with_inputs(|| (&array, SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| array.top_value(ctx).unwrap());
}

#[divan::bench(args = ARGS)]
fn top_value_i64(bencher: Bencher, args: (f64, f64)) {
    let array = make_array::<i64>(args, i64::from);
    bencher
        .with_inputs(|| (&array, SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| array.top_value(ctx).unwrap());
}

#[divan::bench(args = ARGS)]
fn top_value_f64(bencher: Bencher, args: (f64, f64)) {
    let array = make_array::<f64>(args, f64::from);
    bencher
        .with_inputs(|| (&array, SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| array.top_value(ctx).unwrap());
}

#[divan::bench(args = [(0.95, 0.0), (0.3, 0.0)])]
fn top_value_u8(bencher: Bencher, args: (f64, f64)) {
    let array = make_array::<u8>(args, |v| v as u8);
    bencher
        .with_inputs(|| (&array, SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| array.top_value(ctx).unwrap());
}
