// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Checked `u32 + u32 -> u32` over two nullable columns via [`try_map_with_mask`]
//! with a value-only closure. Per-lane `is_none()` flags are bit-packed and
//! AND-ed with the chunk validity word so null-lane overflow is filtered
//! without the closure ever inspecting `valid`.
//!
//! Verified at startup via [`assert_overflow_parity`] (valid-lane overflow
//! propagates as `Err`) and [`assert_null_overflow_suppressed`] (null-lane
//! overflow does not).

#![expect(clippy::unwrap_used)]

use std::mem::MaybeUninit;

use divan::Bencher;
use rand::SeedableRng;
use rand::prelude::*;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_buffer::Buffer;
use vortex_buffer::lane_ops_indexed::IndexedSourceExt;
use vortex_buffer::lane_ops_indexed::LaneZip;

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
    /// that ignores validity would Err on them. The implementation under test
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
            LaneZip::new(lhs.as_slice(), rhs.as_slice())
                .try_map_with_mask(&combined, out.as_mut_slice(), |(a, b), _valid| {
                    a.checked_add(b)
                })
                .unwrap();
            (combined, out)
        });
}

// ---------------------------------------------------------------------------
// Parity assertions — must pass before divan runs benches.
// ---------------------------------------------------------------------------

/// Overflow at a valid lane must propagate as `Err`.
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
    let r = LaneZip::new(lhs.as_slice(), rhs.as_slice()).try_map_with_mask(
        &mask,
        out.as_mut_slice(),
        |(a, b), _| a.checked_add(b),
    );
    assert!(r.is_err(), "bitpack should Err on overflow");
}

/// Overflow at a null lane must NOT propagate.
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
    let r = LaneZip::new(lhs.as_slice(), rhs.as_slice()).try_map_with_mask(
        &mask,
        out.as_mut_slice(),
        |(a, b), _| a.checked_add(b),
    );
    assert!(r.is_ok(), "bitpack: null-lane overflow leaked");
}
