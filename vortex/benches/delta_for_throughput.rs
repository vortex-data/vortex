// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Decompression throughput comparison of `delta(for(fastlanes))` against `for(fastlanes)`.
//!
//! Both encoding trees are built over the *same* realistic, monotonically increasing 64k-row
//! `i64` column (the kind of column where Delta is selected over plain FoR). The benchmark
//! canonicalizes each tree so the timed region is exactly the decode path:
//!
//! * `for_fastlanes`        : `FoR <- BitPacked`
//! * `delta_for_fastlanes`  : `Delta <- { bases: FoR <- BitPacked, deltas: FoR <- BitPacked }`
//! * `delta_bitpacked`      : `Delta <- { bases: FoR <- BitPacked, deltas: BitPacked }` — the tree
//!   the BtrBlocks compressor emits for RLE run indices, and the one decoded by the fused
//!   `delta_decompress` fast path (per-chunk unpack feeding straight into `undelta`).

#![expect(clippy::unwrap_used)]

use std::fmt;

use divan::Bencher;
#[cfg(not(codspeed))]
use divan::counter::BytesCount;
use mimalloc::MiMalloc;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::ExecutionCtx;
use vortex::array::IntoArray;
use vortex::array::LEGACY_SESSION;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::PrimitiveArray;
use vortex::encodings::fastlanes::Delta;
use vortex::encodings::fastlanes::FoR;
use vortex::encodings::fastlanes::FoRArrayExt;
use vortex::encodings::fastlanes::bitpack_compress::bitpack_to_best_bit_width;
use vortex::encodings::fastlanes::delta_compress;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    // `MANUAL=<iters>` runs a plain wall-clock loop (no divan/codspeed harness). This is used for
    // local A/B timing and for recording focused Samply profiles of just the decode path.
    if let Ok(iters) = std::env::var("MANUAL") {
        let iters: usize = iters.parse().unwrap_or(2000);
        manual(iters);
        return;
    }
    divan::main();
}

fn manual(iters: usize) {
    use std::hint::black_box;
    use std::time::Instant;

    let bytes = (NUM_VALUES * size_of::<i64>()) as f64;
    let only = std::env::var("ONLY").ok();
    for (name, build) in [
        ("for_fastlanes", for_fastlanes as fn() -> ArrayRef),
        (
            "delta_for_fastlanes",
            delta_for_fastlanes as fn() -> ArrayRef,
        ),
        ("delta_bitpacked", delta_bitpacked as fn() -> ArrayRef),
    ] {
        if let Some(only) = &only
            && only != name
        {
            continue;
        }
        let compressed = build();
        // Warmup.
        for _ in 0..(iters / 10).max(1) {
            let mut ctx = LEGACY_SESSION.create_execution_ctx();
            black_box(compressed.clone().execute::<Canonical>(&mut ctx).unwrap());
        }
        let start = Instant::now();
        for _ in 0..iters {
            let mut ctx = LEGACY_SESSION.create_execution_ctx();
            black_box(compressed.clone().execute::<Canonical>(&mut ctx).unwrap());
        }
        let elapsed = start.elapsed();
        let per_iter = elapsed.as_secs_f64() / iters as f64;
        let gibs = bytes / per_iter / (1024.0 * 1024.0 * 1024.0);
        println!(
            "{name:<22} {:>8.2} us/iter   {gibs:>6.2} GiB/s   (nbytes_compressed={})",
            per_iter * 1e6,
            compressed.nbytes(),
        );
    }
}

/// 64 full FastLanes chunks of 1,024 values.
const NUM_VALUES: usize = 64 * 1024;

fn with_byte_counter<'a, 'b>(bencher: Bencher<'a, 'b>, bytes: u64) -> Bencher<'a, 'b> {
    #[cfg(not(codspeed))]
    return bencher.counter(BytesCount::new(bytes));
    #[cfg(codspeed)]
    {
        _ = bytes;
        return bencher;
    }
}

/// A realistic monotonically increasing `i64` "timestamp" column: a cumulative sum of small
/// positive jitter on top of a large base offset. This is the regime where Delta beats FoR on
/// size, so it is the fair comparison point for decode throughput.
fn monotone_i64() -> PrimitiveArray {
    let mut rng = StdRng::seed_from_u64(0);
    let mut acc: i64 = 1_700_000_000_000; // ~milliseconds since epoch
    let values = (0..NUM_VALUES).map(|_| {
        acc += rng.random_range(1i64..1000);
        acc
    });
    PrimitiveArray::from_iter(values)
}

/// Build a `FoR <- BitPacked` tree over `prim`, matching what the compressor emits for an
/// integer column with a non-trivial minimum.
fn for_bp(prim: PrimitiveArray, ctx: &mut ExecutionCtx) -> ArrayRef {
    let for_ = FoR::encode(prim).unwrap();
    let reference = for_.reference_scalar().clone();
    let inner = for_
        .encoded()
        .clone()
        .execute::<PrimitiveArray>(ctx)
        .unwrap();
    let bp = bitpack_to_best_bit_width(&inner, ctx).unwrap();
    FoR::try_new(bp.into_array(), reference)
        .unwrap()
        .into_array()
}

/// `for(fastlanes)` over the monotone column.
fn for_fastlanes() -> ArrayRef {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    for_bp(monotone_i64(), &mut ctx)
}

/// `delta(for(fastlanes))`: delta-encode the column, then compress both the bases and deltas
/// children with `FoR <- BitPacked`.
fn delta_for_fastlanes() -> ArrayRef {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let prim = monotone_i64();
    let len = prim.len();
    let (bases, deltas) = delta_compress(&prim, &mut ctx).unwrap();
    let bases = for_bp(bases, &mut ctx);
    let deltas = for_bp(deltas, &mut ctx);
    Delta::try_new(bases, deltas, 0, len).unwrap().into_array()
}

/// `delta(...)` with `FoR <- BitPacked` bases and bare `BitPacked` deltas. For a monotone column
/// the deltas are non-negative and `FoR`'s min-subtraction yields no bit-width reduction, so this
/// is the representative tree a cost-based compressor emits — and the one the fused decode path
/// optimizes.
fn delta_bitpacked() -> ArrayRef {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let prim = monotone_i64();
    let len = prim.len();
    let (bases, deltas) = delta_compress(&prim, &mut ctx).unwrap();
    let bases = for_bp(bases, &mut ctx);
    let deltas = bitpack_to_best_bit_width(&deltas, &mut ctx)
        .unwrap()
        .into_array();
    Delta::try_new(bases, deltas, 0, len).unwrap().into_array()
}

#[derive(Copy, Clone)]
struct SetupFn {
    func: fn() -> ArrayRef,
    name: &'static str,
}

impl fmt::Display for SetupFn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

#[divan::bench(
    args = [
        SetupFn { func: for_fastlanes, name: "for_fastlanes" },
        SetupFn { func: delta_for_fastlanes, name: "delta_for_fastlanes" },
        SetupFn { func: delta_bitpacked, name: "delta_bitpacked" },
    ]
)]
fn decompress(bencher: Bencher, setup_fn: SetupFn) {
    let compressed = (setup_fn.func)();

    with_byte_counter(bencher, (NUM_VALUES * size_of::<i64>()) as u64)
        .with_inputs(|| (&compressed, LEGACY_SESSION.create_execution_ctx()))
        .bench_refs(|(a, ctx)| (**a).clone().execute::<Canonical>(ctx));
}
