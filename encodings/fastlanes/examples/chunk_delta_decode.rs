// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Decode-kernel microbenchmark: ChunkDelta over per-chunk bit-packed deltas vs plain bit-packing
//! of the same width, 64K values (L2-resident), best of 2000 runs.

#![expect(clippy::unwrap_used)]

use std::hint::black_box;
use std::time::Duration;
use std::time::Instant;

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_fastlanes::ChunkDelta;
use vortex_fastlanes::ChunkDeltaArraySlotsExt;
use vortex_fastlanes::VarBitPacked;

const N: usize = 1 << 16;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn bench(name: &str, array: &ArrayRef) {
    let session = vortex_array::array_session();
    vortex_fastlanes::initialize(&session);
    let mut ctx = session.create_execution_ctx();
    let mut best = Duration::MAX;
    for _ in 0..2000 {
        let start = Instant::now();
        black_box(array.clone().execute::<PrimitiveArray>(&mut ctx).unwrap());
        best = best.min(start.elapsed());
    }
    let ns = best.as_secs_f64() * 1e9 / N as f64;
    println!("{name:<44} {ns:>6.3} ns/value  {:>7.0} M values/s", 1e3 / ns);
}

/// A random walk with steps in `-2^(bits-1)..2^(bits-1)`, so the deltas less their minimum need
/// `bits`.
fn walk<T: Copy>(bits: u32, cast: impl Fn(i64) -> T) -> Vec<T> {
    let mut x = 0x1234_5678u64;
    let mut v = 1_000_000i64;
    (0..N)
        .map(|_| {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            v += (x >> (64 - bits)) as i64 - (1 << (bits - 1));
            cast(v)
        })
        .collect()
}

fn run(name: &str, values: PrimitiveArray) {
    let session = vortex_array::array_session();
    vortex_fastlanes::initialize(&session);
    let mut ctx = session.create_execution_ctx();
    let plain = ChunkDelta::encode(&values, &mut ctx).unwrap();
    let deltas = plain
        .deltas()
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)
        .unwrap();
    let packed = VarBitPacked::encode(&deltas, &mut ctx).unwrap().into_array();
    bench(&format!("{name}: deltas unpack only"), &packed);
    let fused = ChunkDelta::try_new(
        packed,
        plain.bases().clone(),
        plain.mins().clone(),
        Validity::NonNullable,
        N,
        0,
    )
    .unwrap()
    .into_array();
    bench(&format!("{name}: ChunkDelta fused"), &fused);
}

fn main() {
    run("i64, 6-bit steps", PrimitiveArray::from_iter(walk(6, |v| v)));
    run("i32, 6-bit steps", PrimitiveArray::from_iter(walk(6, |v| v as i32)));
    run("i16, 4-bit steps", PrimitiveArray::from_iter(walk(4, |v| v as i16)));
}
