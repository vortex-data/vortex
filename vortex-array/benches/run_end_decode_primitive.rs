// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Decoding primitive run-end arrays, in the shapes that dictionary codes and nullable keys take
//! in TPC-DS: short runs, with and without nulls.

#![expect(clippy::cast_possible_truncation)]

use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::session::AggregateFnSession;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::runend::compress::runend_decode_primitive;
use vortex_array::dtype::NativePType;
use vortex_array::optimizer::kernels::KernelSession;
use vortex_array::session::ArraySession;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    VortexSession::empty()
        .with::<ArraySession>()
        .with::<KernelSession>()
        .with::<AggregateFnSession>()
});

const LEN: usize = 16_384;

/// `(average run length, fraction of null runs)`.
const ARGS: &[(usize, f64)] = &[
    (1, 0.0),
    (2, 0.04),
    (7, 0.0),
    (7, 0.04),
    (26, 0.0),
    (100, 0.0),
];

fn bench_decode<T: NativePType>(
    bencher: Bencher,
    avg_run: usize,
    null_frac: f64,
    value: fn(u64) -> T,
) {
    let mut rng = StdRng::seed_from_u64(7);
    let (mut ends, mut values, mut valid) = (Vec::new(), Vec::new(), Vec::new());
    let mut pos = 0usize;
    while pos < LEN {
        pos = (pos + rng.random_range(1..=2 * avg_run - 1)).min(LEN);
        ends.push(pos as u32);
        values.push(value(rng.random_range(0..1_000)));
        valid.push(!rng.random_bool(null_frac));
    }
    let ends = PrimitiveArray::new(Buffer::from(ends), Validity::NonNullable);
    let validity = if null_frac > 0.0 {
        Validity::from(BitBuffer::from_iter(valid))
    } else {
        Validity::NonNullable
    };
    let values = PrimitiveArray::new(Buffer::from(values), validity);
    bencher
        .with_inputs(|| (ends.clone(), values.clone(), SESSION.create_execution_ctx()))
        .bench_refs(|(ends, values, ctx)| {
            runend_decode_primitive(ends.clone(), values.clone(), 0, LEN, ctx)
        });
}

#[divan::bench(args = ARGS)]
fn decode_i64(bencher: Bencher, (avg_run, null_frac): (usize, f64)) {
    bench_decode(bencher, avg_run, null_frac, |v| v as i64);
}

#[divan::bench(args = ARGS)]
fn decode_u16(bencher: Bencher, (avg_run, null_frac): (usize, f64)) {
    bench_decode(bencher, avg_run, null_frac, |v| v as u16);
}
