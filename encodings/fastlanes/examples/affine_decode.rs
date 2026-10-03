// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Decode-kernel microbenchmark: Affine modes vs FoR vs plain BitPacked on the same bit-packed
//! residual widths, 64K i64 values (L2-resident), best of 2000 runs.

#![expect(clippy::unwrap_used)]

use std::hint::black_box;
use std::time::Duration;
use std::time::Instant;

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_fastlanes::Affine;
use vortex_fastlanes::AffineArrayExt;
use vortex_fastlanes::AffineArraySlotsExt;
use vortex_fastlanes::AffineOptions;
use vortex_fastlanes::BitPackedData;
use vortex_fastlanes::FoR;
use vortex_fastlanes::FoRArraySlotsExt;

const N: usize = 1 << 16;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn bits(a: &ArrayRef) -> u8 {
    let mut ctx = vortex_array::array_session().create_execution_ctx();
    let p = a.clone().execute::<PrimitiveArray>(&mut ctx).unwrap();
    let max = p.as_slice::<i64>().iter().map(|&v| v as u64).max().unwrap_or(0);
    (64 - max.leading_zeros()) as u8
}

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

fn main() {
    let session = vortex_array::array_session();
    vortex_fastlanes::initialize(&session);
    let mut ctx = session.create_execution_ctx();

    // Microsecond timestamps on a one-second grid with 16 bits of spread per chunk, plus a
    // regular nanosecond grid with small jitter for the slope modes.
    let mut x = 0x1234_5678u64;
    let mut rnd = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    let grid = PrimitiveArray::from_iter(
        (0..N as i64).map(|i| 1_672_531_200_000_000 + (i / 1024) * 1_000_000_000 + (rnd() % 65_536) as i64 * 1_000_000),
    );
    let trend = PrimitiveArray::from_iter(
        (0..N as i64).map(|i| 1_700_000_000_000_000_000 + i * 60_000_000_000 + (rnd() % 65_536) as i64),
    );

    // Lower bounds: memcpy-like copy of a primitive, and unpacking a 16-bit BitPacked child.
    let prim = grid.clone().into_array();
    let bp = BitPackedData::encode(&prim_residuals(&grid), 16, &mut ctx).unwrap().into_array();
    bench("BitPacked unpack only (16 bit)", &bp);

    let for_ = FoR::encode_chunked(grid.clone(), &mut ctx).unwrap();
    let for_bp = FoR::try_new_chunked(
        BitPackedData::encode(for_.encoded(), bits(for_.encoded()), &mut ctx).unwrap().into_array(),
        for_.references().clone(),
        0,
    )
    .unwrap()
    .into_array();
    bench(&format!("FoR per-chunk, fused ({} bit)", bits(for_.encoded())), &for_bp);

    for (name, values, options) in [
        ("Affine FoR mode", &grid, AffineOptions::FOR),
        ("Affine GCD", &grid, AffineOptions::SCALE),
        ("Affine slope", &trend, AffineOptions::SLOPE),
        ("Affine GCD+slope", &trend, AffineOptions::ALL),
    ] {
        let affine = Affine::encode(values, options, &mut ctx).unwrap();
        let width = bits(affine.encoded());
        let packed = BitPackedData::encode(affine.encoded(), width, &mut ctx).unwrap().into_array();
        let array = Affine::try_new(
            packed,
            affine.references().clone(),
            affine.scales().clone(),
            affine.slopes().clone(),
            0,
            affine.slope_shift(),
        )
        .unwrap()
        .into_array();
        bench(&format!("{name} ({width} bit)"), &array);
    }
    black_box(prim);
}

fn prim_residuals(values: &PrimitiveArray) -> ArrayRef {
    PrimitiveArray::from_iter(values.as_slice::<i64>().iter().map(|&v| v & 0xffff)).into_array()
}
