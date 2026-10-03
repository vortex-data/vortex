// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Decode-kernel microbenchmark: ChunkDelta over per-chunk bit-packed deltas vs FastLanes Delta,
//! and the unpack alone, on the same values, 64K values (L2-resident), best of 2000 runs.

#![expect(clippy::unwrap_used)]

use std::hint::black_box;
use std::time::Duration;
use std::time::Instant;

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_fastlanes::BitPackedData;
use vortex_fastlanes::ChunkDelta;
use vortex_fastlanes::ChunkDeltaArraySlotsExt;
use vortex_fastlanes::Delta;
use vortex_fastlanes::FoR;
use vortex_fastlanes::FoRArraySlotsExt;
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

fn run(name: &str, values: PrimitiveArray, non_negative_deltas: bool) {
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

    // FastLanes Delta on the same values: over plain deltas (its undelta kernel alone), and over
    // deltas FoR-encoded and bit-packed per chunk, as a compressed tree holds them.
    let (bases, deltas) = vortex_fastlanes::delta_compress(&values, &mut ctx).unwrap();
    let plain_delta = Delta::try_new(bases.clone().into_array(), deltas.clone().into_array(), 0, N)
        .unwrap()
        .into_array();
    bench(&format!("{name}: FastLanes Delta, plain deltas"), &plain_delta);
    let for_ = FoR::encode_chunked(deltas.clone(), &mut ctx).unwrap();
    let width = bits(for_.encoded(), &mut ctx);
    let for_bp = FoR::try_new_chunked(
        BitPackedData::encode(for_.encoded(), width, &mut ctx).unwrap().into_array(),
        for_.references().clone(),
        0,
    )
    .unwrap()
    .into_array();
    let packed_delta = Delta::try_new(bases.clone().into_array(), for_bp, 0, N)
        .unwrap()
        .into_array();
    bench(&format!("{name}: FastLanes Delta, FoR+bitpacked"), &packed_delta);
    // Non-negative deltas bit-pack directly, which FastLanes' fused unpack-and-undelta can read.
    if non_negative_deltas {
        let deltas = deltas.into_array();
        let width = bits(&deltas, &mut ctx);
        let bp = BitPackedData::encode(&deltas, width, &mut ctx).unwrap().into_array();
        let bp_delta = Delta::try_new(bases.into_array(), bp, 0, N).unwrap().into_array();
        bench(&format!("{name}: FastLanes Delta, bitpacked"), &bp_delta);
    }
}

/// An increasing series with steps in `0..2^bits`, so every delta is non-negative.
fn rising<T: Copy>(bits: u32, cast: impl Fn(i64) -> T) -> Vec<T> {
    let mut x = 0x1234_5678u64;
    let mut v = 1_000_000i64;
    (0..N)
        .map(|_| {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            v += (x >> (64 - bits)) as i64;
            cast(v)
        })
        .collect()
}

fn bits(a: &ArrayRef, ctx: &mut vortex_array::ExecutionCtx) -> u8 {
    let p = a.clone().execute::<PrimitiveArray>(ctx).unwrap();
    let max = vortex_array::match_each_integer_ptype!(p.ptype(), |T| {
        p.as_slice::<T>().iter().map(|&v| v as u64).max().unwrap_or(0)
    });
    (64 - max.leading_zeros()).max(1) as u8
}

/// FastLanes' own u64 kernels on 64 chunks, outside Vortex: what undelta, untranspose and the
/// fused unpack-and-undelta cost by themselves.
fn raw_kernels() {
    use fastlanes::BitPacking;
    use fastlanes::Delta as _;
    use fastlanes::Transpose;
    const W: usize = 6;
    const B: usize = 1024 * W / 64;
    let chunks = N / 1024;
    let input: Vec<[u64; 1024]> = (0..chunks)
        .map(|c| std::array::from_fn(|i| ((c * 1024 + i) as u64 * 2654435761) % 64))
        .collect();
    let packed: Vec<[u64; B]> = input
        .iter()
        .map(|chunk| {
            let mut out = [0u64; B];
            BitPacking::pack::<W, B>(chunk, &mut out);
            out
        })
        .collect();
    let base = [0u64; 16];
    let mut out = vec![[0u64; 1024]; chunks];
    let mut tmp = [0u64; 1024];
    let time = |name: &str, f: &mut dyn FnMut()| {
        let mut best = Duration::MAX;
        for _ in 0..2000 {
            let start = Instant::now();
            f();
            best = best.min(start.elapsed());
        }
        println!("raw u64 {name:<36} {:>6.3} ns/value", best.as_secs_f64() * 1e9 / N as f64);
    };
    time("unpack", &mut || {
        for (p, o) in packed.iter().zip(out.iter_mut()) {
            BitPacking::unpack::<W, B>(p, o);
        }
        black_box(&out);
    });
    time("undelta", &mut || {
        for (i, o) in input.iter().zip(out.iter_mut()) {
            u64::undelta::<16>(i, &base, o);
        }
        black_box(&out);
    });
    time("untranspose", &mut || {
        for (i, o) in input.iter().zip(out.iter_mut()) {
            u64::untranspose(i, o);
        }
        black_box(&out);
    });
    time("undelta_pack", &mut || {
        for (p, o) in packed.iter().zip(out.iter_mut()) {
            u64::undelta_pack::<16, W, B>(p, &base, o);
        }
        black_box(&out);
    });
    time("undelta_pack + untranspose", &mut || {
        for (p, o) in packed.iter().zip(out.iter_mut()) {
            u64::undelta_pack::<16, W, B>(p, &base, &mut tmp);
            u64::untranspose(&tmp, o);
        }
        black_box(&out);
    });
}

fn main() {
    raw_kernels();
    run("i64, 6-bit steps", PrimitiveArray::from_iter(walk(6, |v| v)), false);
    run("i32, 6-bit steps", PrimitiveArray::from_iter(walk(6, |v| v as i32)), false);
    run("i16, 4-bit steps", PrimitiveArray::from_iter(walk(4, |v| v as i16)), false);
    run("i64 rising, 6-bit steps", PrimitiveArray::from_iter(rising(6, |v| v)), true);
    run("u32 rising, 6-bit steps", PrimitiveArray::from_iter(rising(6, |v| v as u32)), true);
}
