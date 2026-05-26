// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Fair checked `u32 + u32 -> u32` over two nullable columns.
//!
//! The timed region in both impls does **the same work**:
//! 1. AND the two validity bitmaps to produce the result validity.
//! 2. Allocate the output u32 buffer.
//! 3. Compute lane-wise checked add.
//!
//! That mirrors what `arrow_arith::numeric::add` does internally (null-buffer
//! union, output buffer allocation, inner add loop), so the comparison isolates
//! the inner-loop codegen rather than rewarding our impl for already-merged
//! validity and pre-allocated output.

#![expect(clippy::unwrap_used, clippy::clone_on_ref_ptr)]

use std::mem::MaybeUninit;
use std::sync::Arc;

use arrow_array::Datum;
use arrow_array::UInt32Array;
use arrow_buffer::NullBuffer;
use arrow_buffer::ScalarBuffer;
use divan::Bencher;
use rand::SeedableRng;
use rand::prelude::*;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_buffer::Buffer;
use vortex_buffer::lane_ops_indexed::LaneZip;
use vortex_buffer::lane_ops_indexed::map;
use vortex_buffer::lane_ops_indexed::try_map;
use vortex_buffer::lane_ops_indexed::try_map_nullable;
use vortex_buffer::lane_ops_indexed::try_map_with_mask;

fn main() {
    divan::main();
}

const SIZES: &[usize] = &[4_096, 65_536, 1_048_576, 2_097_152, 4_194_304];
const LHS_VALID_RATE: f64 = 0.7;
const RHS_VALID_RATE: f64 = 0.8;
const LHS_SEED: u64 = 0;
const RHS_SEED: u64 = 1;
const LHS_VALID_SEED: u64 = 2;
const RHS_VALID_SEED: u64 = 3;

struct Fixture {
    lhs: Buffer<u32>,
    rhs: Buffer<u32>,
    lhs_mask: BitBuffer,
    rhs_mask: BitBuffer,
    lhs_arrow: Arc<UInt32Array>,
    rhs_arrow: Arc<UInt32Array>,
    lhs_arrow_nonnull: Arc<UInt32Array>,
    rhs_arrow_nonnull: Arc<UInt32Array>,
}

fn fixture(n: usize) -> Fixture {
    let mut lhs_rng = StdRng::seed_from_u64(LHS_SEED);
    let mut rhs_rng = StdRng::seed_from_u64(RHS_SEED);
    let mut lvr = StdRng::seed_from_u64(LHS_VALID_SEED);
    let mut rvr = StdRng::seed_from_u64(RHS_VALID_SEED);

    let raw_lhs: Vec<u32> = (0..n)
        .map(|_| lhs_rng.random_range(0..u16::MAX as u32))
        .collect();
    let raw_rhs: Vec<u32> = (0..n)
        .map(|_| rhs_rng.random_range(0..u16::MAX as u32))
        .collect();
    let lhs_valid: Vec<bool> = (0..n).map(|_| lvr.random_bool(LHS_VALID_RATE)).collect();
    let rhs_valid: Vec<bool> = (0..n).map(|_| rvr.random_bool(RHS_VALID_RATE)).collect();

    let lhs: Buffer<u32> = raw_lhs.iter().copied().collect();
    let rhs: Buffer<u32> = raw_rhs.iter().copied().collect();

    let lhs_mask = {
        let mut m = BitBufferMut::with_capacity(n);
        for &v in &lhs_valid {
            m.append(v);
        }
        m.freeze()
    };
    let rhs_mask = {
        let mut m = BitBufferMut::with_capacity(n);
        for &v in &rhs_valid {
            m.append(v);
        }
        m.freeze()
    };

    // No-null arrays: arrow takes its `try_binary_no_nulls` path (no null-buffer union).
    let lhs_arrow_nonnull = Arc::new(UInt32Array::from(raw_lhs.clone()));
    let rhs_arrow_nonnull = Arc::new(UInt32Array::from(raw_rhs.clone()));

    let lhs_arrow = Arc::new(UInt32Array::new(
        ScalarBuffer::from(raw_lhs),
        Some(NullBuffer::from(lhs_valid)),
    ));
    let rhs_arrow = Arc::new(UInt32Array::new(
        ScalarBuffer::from(raw_rhs),
        Some(NullBuffer::from(rhs_valid)),
    ));

    Fixture {
        lhs,
        rhs,
        lhs_mask,
        rhs_mask,
        lhs_arrow,
        rhs_arrow,
        lhs_arrow_nonnull,
        rhs_arrow_nonnull,
    }
}

/// `LaneZip` + `try_map_with_mask`, doing the validity AND and output alloc
/// inside the timed region — matches the work arrow does internally.
#[divan::bench(args = SIZES)]
fn indexed_lane_zip_fair(bencher: Bencher, n: usize) {
    let f = fixture(n);
    bencher
        .with_inputs(|| {
            (
                f.lhs.clone(),
                f.rhs.clone(),
                f.lhs_mask.clone(),
                f.rhs_mask.clone(),
            )
        })
        .bench_refs(|(lhs, rhs, lhs_mask, rhs_mask)| {
            // 1. AND the validity bitmaps.
            let combined_mask = lhs_mask as &BitBuffer & rhs_mask as &BitBuffer;
            // 2. Allocate the output u32 buffer.
            let mut out: Vec<MaybeUninit<u32>> = Vec::with_capacity(n);
            // SAFETY: every lane is written before any read inside the kernel.
            unsafe { out.set_len(n) };
            // 3. Run the kernel.
            try_map_with_mask(
                LaneZip::new(lhs.as_slice(), rhs.as_slice()),
                &combined_mask,
                out.as_mut_slice(),
                |(a, b), valid| {
                    if valid { a.checked_add(b) } else { Some(0) }
                },
            )
            .unwrap();
            (combined_mask, out)
        });
}

/// As above, but skips the AND + alloc — the "unfair" baseline showing what the
/// inner loop alone costs.
#[divan::bench(args = SIZES)]
fn indexed_lane_zip_kernel_only(bencher: Bencher, n: usize) {
    let f = fixture(n);
    let combined_mask = &f.lhs_mask & &f.rhs_mask;
    bencher
        .with_inputs(|| {
            let mut out: Vec<MaybeUninit<u32>> = Vec::with_capacity(n);
            unsafe { out.set_len(n) };
            (f.lhs.clone(), f.rhs.clone(), combined_mask.clone(), out)
        })
        .bench_refs(|(lhs, rhs, mask, out)| {
            try_map_with_mask(
                LaneZip::new(lhs.as_slice(), rhs.as_slice()),
                mask,
                out.as_mut_slice(),
                |(a, b), valid| {
                    if valid { a.checked_add(b) } else { Some(0) }
                },
            )
            .unwrap();
        });
}

/// Validity-decoupled variant: same fair work (AND the masks, allocate output),
/// but the arithmetic loop never reads the mask — it runs `a.checked_add(b)` over
/// every lane (null lanes compute irrelevant values masked off by `combined_mask`).
/// This mirrors arrow's architecture, where the null union is a separate pass from
/// the inner add loop.
#[divan::bench(args = SIZES)]
fn indexed_no_mask_fair(bencher: Bencher, n: usize) {
    let f = fixture(n);
    bencher
        .with_inputs(|| {
            (
                f.lhs.clone(),
                f.rhs.clone(),
                f.lhs_mask.clone(),
                f.rhs_mask.clone(),
            )
        })
        .bench_refs(|(lhs, rhs, lhs_mask, rhs_mask)| {
            // 1. AND the validity bitmaps (word-parallel, separate from arithmetic).
            let combined_mask = lhs_mask as &BitBuffer & rhs_mask as &BitBuffer;
            // 2. Allocate the output u32 buffer.
            let mut out: Vec<MaybeUninit<u32>> = Vec::with_capacity(n);
            // SAFETY: every lane is written before any read inside the kernel.
            unsafe { out.set_len(n) };
            // 3. Run the mask-free kernel.
            try_map(
                LaneZip::new(lhs.as_slice(), rhs.as_slice()),
                out.as_mut_slice(),
                |(a, b)| a.checked_add(b),
            )
            .unwrap();
            (combined_mask, out)
        });
}

/// Arrow-parity variant: `try_map_nullable` reproduces arrow's checked-add semantics
/// exactly — `checked_add` runs on every lane, but an overflow only faults at a *valid*
/// lane (the per-lane overflow bitmap is `& combined_mask` once per 64-lane chunk, never
/// in the arithmetic loop). Same fair timed region: AND the masks + allocate output.
#[divan::bench(args = SIZES)]
fn indexed_nullable_fair(bencher: Bencher, n: usize) {
    let f = fixture(n);
    bencher
        .with_inputs(|| {
            (
                f.lhs.clone(),
                f.rhs.clone(),
                f.lhs_mask.clone(),
                f.rhs_mask.clone(),
            )
        })
        .bench_refs(|(lhs, rhs, lhs_mask, rhs_mask)| {
            // 1. AND the validity bitmaps (== arrow's null-buffer union).
            let combined_mask = lhs_mask as &BitBuffer & rhs_mask as &BitBuffer;
            // 2. Allocate the output u32 buffer.
            let mut out: Vec<MaybeUninit<u32>> = Vec::with_capacity(n);
            // SAFETY: every lane is written before any read inside the kernel.
            unsafe { out.set_len(n) };
            // 3. Run the null-lenient kernel (overflow ignored at null lanes, like arrow).
            try_map_nullable(
                LaneZip::new(lhs.as_slice(), rhs.as_slice()),
                &combined_mask,
                out.as_mut_slice(),
                |(a, b)| a.checked_add(b),
            )
            .unwrap();
            (combined_mask, out)
        });
}

/// CONTROL: arrow's **wrapping** add on two non-null columns. Same harness and inputs as
/// `arrow_add_nonnull`, but `add_wrapping` has no per-element `?`, so it goes through arrow's
/// infallible `binary` which the autovectorizer can SIMD. If this is fast while the checked
/// `arrow_add_nonnull` is slow, the harness is fair and the `?` short-circuit is the cause.
#[divan::bench(args = SIZES)]
fn arrow_add_wrapping_nonnull(bencher: Bencher, n: usize) {
    let f = fixture(n);
    bencher
        .with_inputs(|| (f.lhs_arrow_nonnull.clone(), f.rhs_arrow_nonnull.clone()))
        .bench_refs(|(lhs, rhs)| {
            arrow_arith::numeric::add_wrapping(
                lhs.as_ref() as &dyn Datum,
                rhs.as_ref() as &dyn Datum,
            )
            .unwrap()
        });
}

/// Like-for-like wrapping comparison against `arrow_add_wrapping_nonnull`: our infallible
/// `map` doing `wrapping_add` over two non-null columns. Neither side does overflow work,
/// so this isolates pure vectorized add throughput.
#[divan::bench(args = SIZES)]
fn indexed_map_wrapping_nonnull(bencher: Bencher, n: usize) {
    let f = fixture(n);
    bencher
        .with_inputs(|| (f.lhs.clone(), f.rhs.clone()))
        .bench_refs(|(lhs, rhs)| {
            let mut out: Vec<MaybeUninit<u32>> = Vec::with_capacity(n);
            // SAFETY: every lane is written before any read inside the kernel.
            unsafe { out.set_len(n) };
            map(
                LaneZip::new(lhs.as_slice(), rhs.as_slice()),
                out.as_mut_slice(),
                |(a, b)| a.wrapping_add(b),
            );
            out
        });
}

/// No-validity comparison: `try_map` checked add over two non-null columns, timing only
/// allocate + kernel (there is no validity to merge). Pairs with `arrow_add_nonnull`.
#[divan::bench(args = SIZES)]
fn indexed_try_map_nonnull(bencher: Bencher, n: usize) {
    let f = fixture(n);
    bencher
        .with_inputs(|| (f.lhs.clone(), f.rhs.clone()))
        .bench_refs(|(lhs, rhs)| {
            let mut out: Vec<MaybeUninit<u32>> = Vec::with_capacity(n);
            // SAFETY: every lane is written before any read inside the kernel.
            unsafe { out.set_len(n) };
            try_map(
                LaneZip::new(lhs.as_slice(), rhs.as_slice()),
                out.as_mut_slice(),
                |(a, b)| a.checked_add(b),
            )
            .unwrap();
            out
        });
}

/// `arrow_arith::numeric::add` on two **non-null** `UInt32Array`s — arrow's
/// `try_binary_no_nulls` path: a dense scalar `op(a,b)?` loop, no null-buffer union.
#[divan::bench(args = SIZES)]
fn arrow_add_nonnull(bencher: Bencher, n: usize) {
    let f = fixture(n);
    bencher
        .with_inputs(|| (f.lhs_arrow_nonnull.clone(), f.rhs_arrow_nonnull.clone()))
        .bench_refs(|(lhs, rhs)| {
            arrow_arith::numeric::add(lhs.as_ref() as &dyn Datum, rhs.as_ref() as &dyn Datum)
                .unwrap()
        });
}

/// `arrow_arith::numeric::add` on two `UInt32Array`s — does null-union + alloc
/// + inner add loop inside, with type dispatch through `&dyn Datum`.
#[divan::bench(args = SIZES)]
fn arrow_add(bencher: Bencher, n: usize) {
    let _ = n;
    let f = fixture(n);
    bencher
        .with_inputs(|| (f.lhs_arrow.clone(), f.rhs_arrow.clone()))
        .bench_refs(|(lhs, rhs)| {
            arrow_arith::numeric::add(lhs.as_ref() as &dyn Datum, rhs.as_ref() as &dyn Datum)
                .unwrap()
        });
}
