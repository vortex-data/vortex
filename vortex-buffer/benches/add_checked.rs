// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Checked `u32 + u32 -> u32` over two nullable columns.
//!
//! Two implementations:
//!
//! - [`bitpack_value_only`] — production path via [`try_map_with_mask`] with a
//!   value-only closure. Per-lane `is_none()` flags are bit-packed and AND-ed
//!   with the chunk validity word so null-lane overflow is filtered without
//!   the closure ever inspecting `valid`.
//! - [`premask_then_simd`] — hand-rolled ceiling. Bit-broadcasts each mask bit
//!   to `0x00000000`/`0xFFFFFFFF`, ANDs into both operands (null lanes become
//!   `0+0`), then unconditional `overflowing_add` with a per-chunk OR-reduced
//!   `fail_acc` and cold scalar attribution. Same pattern that beat arrow on
//!   the primitive cast bench (37 µs vs 55 µs).
//!
//! Both are verified at startup via [`assert_overflow_parity`] (valid-lane
//! overflow propagates as `Err`) and [`assert_null_overflow_suppressed`]
//! (null-lane overflow does not).

#![expect(clippy::unwrap_used)]

use std::mem::MaybeUninit;

use divan::Bencher;
use rand::SeedableRng;
use rand::prelude::*;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_buffer::Buffer;
use vortex_buffer::lane_ops_indexed::LaneZip;
use vortex_buffer::lane_ops_indexed::try_map_with_mask;

fn main() {
    assert_overflow_parity();
    assert_null_overflow_suppressed();
    divan::main();
}

const SIZES: &[usize] = &[4_096, 65_536, 1_048_576, 2_097_152, 4_194_304];
const LHS_VALID_RATE: f64 = 0.7;
const RHS_VALID_RATE: f64 = 0.8;

struct Fixture {
    /// Valid lanes carry bounded values; null lanes hold `u32::MAX` so a kernel
    /// that ignores validity would Err on them. Both implementations under test
    /// must suppress that.
    lhs: Buffer<u32>,
    rhs: Buffer<u32>,
    lhs_mask: BitBuffer,
    rhs_mask: BitBuffer,
}

fn fixture(n: usize) -> Fixture {
    let mut lhs_rng = StdRng::seed_from_u64(0);
    let mut rhs_rng = StdRng::seed_from_u64(1);
    let mut lvr = StdRng::seed_from_u64(2);
    let mut rvr = StdRng::seed_from_u64(3);

    let lhs_valid: Vec<bool> = (0..n).map(|_| lvr.random_bool(LHS_VALID_RATE)).collect();
    let rhs_valid: Vec<bool> = (0..n).map(|_| rvr.random_bool(RHS_VALID_RATE)).collect();

    let lhs: Buffer<u32> = (0..n)
        .map(|i| {
            if lhs_valid[i] {
                lhs_rng.random_range(0..u16::MAX as u32)
            } else {
                u32::MAX
            }
        })
        .collect();
    let rhs: Buffer<u32> = (0..n)
        .map(|i| {
            if rhs_valid[i] {
                rhs_rng.random_range(0..u16::MAX as u32)
            } else {
                u32::MAX
            }
        })
        .collect();

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

    Fixture {
        lhs,
        rhs,
        lhs_mask,
        rhs_mask,
    }
}

fn alloc_out(n: usize) -> Vec<MaybeUninit<u32>> {
    let mut out = Vec::with_capacity(n);
    // SAFETY: every lane is written before any read inside the kernel.
    unsafe { out.set_len(n) };
    out
}

// ---------------------------------------------------------------------------
// bitpack_value_only — production path via try_map_with_mask.
// ---------------------------------------------------------------------------

#[divan::bench(args = SIZES)]
fn bitpack_value_only(bencher: Bencher, n: usize) {
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
        .bench_refs(|(lhs, rhs, lm, rm)| {
            let combined = lm as &BitBuffer & rm as &BitBuffer;
            let mut out = alloc_out(n);
            try_map_with_mask(
                LaneZip::new(lhs.as_slice(), rhs.as_slice()),
                &combined,
                out.as_mut_slice(),
                |(a, b), _valid| a.checked_add(b),
            )
            .unwrap();
            (combined, out)
        });
}

// ---------------------------------------------------------------------------
// premask_then_simd — hand-rolled ceiling.
// ---------------------------------------------------------------------------

#[inline]
fn handrolled_premask(
    lhs: &[u32],
    rhs: &[u32],
    mask: &BitBuffer,
    out: &mut [MaybeUninit<u32>],
) -> Result<(), usize> {
    /// Per-chunk hot loop. Bit-broadcasts each validity bit to 0x00 / 0xFF,
    /// ANDs both operands, then `overflowing_add`. Returns true if any lane in
    /// `[base, base+count)` overflowed. `#[inline(always)]` keeps the literal
    /// `64` at the full-chunk call site for const propagation.
    #[inline(always)]
    fn chunk(
        lhs: &[u32],
        rhs: &[u32],
        out: &mut [MaybeUninit<u32>],
        src_chunk: u64,
        base: usize,
        count: usize,
    ) -> bool {
        let mut fail_acc: u64 = 0;
        for bit_idx in 0..count {
            let i = base + bit_idx;
            let lane_mask = (((src_chunk >> bit_idx) & 1) as u32).wrapping_neg();
            // SAFETY: caller guarantees base + count <= len.
            let a = unsafe { *lhs.get_unchecked(i) } & lane_mask;
            let b = unsafe { *rhs.get_unchecked(i) } & lane_mask;
            let (sum, overflow) = a.overflowing_add(b);
            fail_acc |= overflow as u64;
            // SAFETY: caller guarantees base + count <= len.
            unsafe { out.get_unchecked_mut(i).write(sum) };
        }
        fail_acc != 0
    }

    /// Cold attribution. Walks the chunk on raw (unmasked) operands and reports
    /// the first valid lane that overflows. Null lanes were premasked to `0+0`
    /// in the hot loop so they cannot contribute here.
    #[cold]
    #[inline(never)]
    fn attribute(lhs: &[u32], rhs: &[u32], src_chunk: u64, base: usize, count: usize) -> usize {
        for bit_idx in 0..count {
            if (src_chunk >> bit_idx) & 1 == 0 {
                continue;
            }
            let i = base + bit_idx;
            // SAFETY: caller guarantees base + count <= len.
            let a = unsafe { *lhs.get_unchecked(i) };
            let b = unsafe { *rhs.get_unchecked(i) };
            if a.checked_add(b).is_none() {
                return i;
            }
        }
        unreachable!("attribute called without a failing valid lane")
    }

    let len = lhs.len();
    assert_eq!(len, rhs.len());
    assert_eq!(len, mask.len());
    assert_eq!(len, out.len());
    let chunks = mask.chunks();
    let chunks_count = len / 64;
    let remainder = len % 64;

    for (chunk_idx, src_chunk) in chunks.iter().enumerate() {
        let base = chunk_idx * 64;
        if chunk(lhs, rhs, out, src_chunk, base, 64) {
            return Err(attribute(lhs, rhs, src_chunk, base, 64));
        }
    }
    if remainder != 0 {
        let src_chunk = chunks.remainder_bits();
        let base = chunks_count * 64;
        if chunk(lhs, rhs, out, src_chunk, base, remainder) {
            return Err(attribute(lhs, rhs, src_chunk, base, remainder));
        }
    }
    Ok(())
}

#[divan::bench(args = SIZES)]
fn premask_then_simd(bencher: Bencher, n: usize) {
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
        .bench_refs(|(lhs, rhs, lm, rm)| {
            let combined = lm as &BitBuffer & rm as &BitBuffer;
            let mut out = alloc_out(n);
            handrolled_premask(
                lhs.as_slice(),
                rhs.as_slice(),
                &combined,
                out.as_mut_slice(),
            )
            .unwrap();
            (combined, out)
        });
}

// ---------------------------------------------------------------------------
// Parity assertions — must pass before divan runs benches.
// ---------------------------------------------------------------------------

/// Both implementations must Err on overflow at a valid lane.
fn assert_overflow_parity() {
    let lhs: Vec<u32> = vec![1, 2, u32::MAX, 4];
    let rhs: Vec<u32> = vec![10, 20, 1, 40];
    let valid = vec![true; 4];

    let mask = {
        let mut m = BitBufferMut::with_capacity(4);
        for &v in &valid {
            m.append(v);
        }
        m.freeze()
    };

    let mut out: Vec<MaybeUninit<u32>> = (0..4).map(|_| MaybeUninit::uninit()).collect();
    let bitpack = try_map_with_mask(
        LaneZip::new(lhs.as_slice(), rhs.as_slice()),
        &mask,
        out.as_mut_slice(),
        |(a, b), _| a.checked_add(b),
    );
    assert!(bitpack.is_err(), "bitpack should Err on overflow");

    let mut out: Vec<MaybeUninit<u32>> = (0..4).map(|_| MaybeUninit::uninit()).collect();
    let prem = handrolled_premask(&lhs, &rhs, &mask, &mut out);
    assert!(prem.is_err(), "premask should Err on overflow");
}

/// Both implementations must NOT Err when only null lanes would overflow.
fn assert_null_overflow_suppressed() {
    // Lane 2 is null and holds an overflowing value; valid lanes are safe.
    let lhs: Vec<u32> = vec![1, 2, u32::MAX, 4];
    let rhs: Vec<u32> = vec![10, 20, 1, 40];
    let valid = vec![true, true, false, true];

    let mask = {
        let mut m = BitBufferMut::with_capacity(4);
        for &v in &valid {
            m.append(v);
        }
        m.freeze()
    };

    let mut out = alloc_out(4);
    let bitpack = try_map_with_mask(
        LaneZip::new(lhs.as_slice(), rhs.as_slice()),
        &mask,
        out.as_mut_slice(),
        |(a, b), _| a.checked_add(b),
    );
    assert!(bitpack.is_ok(), "bitpack: null-lane overflow leaked");

    let mut out = alloc_out(4);
    let prem = handrolled_premask(&lhs, &rhs, &mask, &mut out);
    assert!(prem.is_ok(), "premask: null-lane overflow leaked");
}
