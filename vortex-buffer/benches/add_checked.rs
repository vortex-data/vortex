// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Checked `u32 + u32 -> u32` over two nullable columns — exhaustive variant
//! comparison.
//!
//! Variants differ along three axes:
//!
//! 1. **Closure suppression strategy** — how the closure (if any) handles null lanes
//!    - `value_only`: `|(a,b), _|` ignores validity
//!    - `if_else`: `|(a,b), valid| if valid { ... } else { Some(default) }`
//!    - `or_else`: `|(a,b), valid| ....or_else(|| (!valid).then(...))`
//!    - `mul_trick`: `(a * valid as u32).checked_add(b * valid as u32)`
//!
//! 2. **Fail tracking scheme**
//!    - bit-pack: `fail_bits |= (is_none << bit_idx)`; chunk-AND with mask
//!    - boolean: `fail_acc |= is_none as u64`; cold replay attribution
//!
//! 3. **Validity application**
//!    - in closure: closure consumes `valid`
//!    - post-mask: kernel ANDs fail bitmap with `src_chunk`
//!    - pre-mask: kernel zeros null-lane values via bit-broadcast before SIMD add
//!    - none: ignore validity (ceiling only — not correct for real inputs)
//!
//! All correctness-preserving variants are verified via [`assert_overflow_parity`]
//! and [`assert_null_overflow_suppressed`] at startup. The `pure_simd_no_validity`
//! variant is benched as a ceiling only — it does not respect nullability.

#![expect(clippy::unwrap_used)]

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
use vortex_buffer::lane_ops_indexed::try_map_with_mask;

fn main() {
    assert_overflow_parity();
    assert_null_overflow_suppressed();
    assert_pure_simd_errs_on_realistic_data();
    divan::main();
}

const SIZES: &[usize] = &[4_096, 65_536, 1_048_576, 2_097_152, 4_194_304];
const LHS_VALID_RATE: f64 = 0.7;
const RHS_VALID_RATE: f64 = 0.8;

struct Fixture {
    /// **Realistic** lhs: valid lanes bounded, null lanes `u32::MAX`.
    /// A kernel that ignores validity will see overflow at null lanes.
    lhs: Buffer<u32>,
    rhs: Buffer<u32>,
    /// **Sanitized** lhs: valid lanes bounded, null lanes pre-zeroed.
    /// Used by `pure_simd_no_validity_sanitized` only — its precondition is
    /// "someone already zeroed the nulls."
    lhs_sanitized: Buffer<u32>,
    rhs_sanitized: Buffer<u32>,
    lhs_mask: BitBuffer,
    rhs_mask: BitBuffer,
    lhs_arrow: Arc<UInt32Array>,
    rhs_arrow: Arc<UInt32Array>,
}

fn fixture(n: usize) -> Fixture {
    let mut lhs_rng = StdRng::seed_from_u64(0);
    let mut rhs_rng = StdRng::seed_from_u64(1);
    let mut lvr = StdRng::seed_from_u64(2);
    let mut rvr = StdRng::seed_from_u64(3);

    let lhs_valid: Vec<bool> = (0..n).map(|_| lvr.random_bool(LHS_VALID_RATE)).collect();
    let rhs_valid: Vec<bool> = (0..n).map(|_| rvr.random_bool(RHS_VALID_RATE)).collect();

    // **Realistic null storage**: null lanes contain u32::MAX. Adding two such
    // values overflows — a kernel that ignores validity will spuriously Err.
    // Valid lanes carry bounded values so the success path is measured at lanes
    // where overflow shouldn't fire.
    let raw_lhs: Vec<u32> = (0..n)
        .map(|i| {
            if lhs_valid[i] {
                lhs_rng.random_range(0..u16::MAX as u32)
            } else {
                u32::MAX
            }
        })
        .collect();
    let raw_rhs: Vec<u32> = (0..n)
        .map(|i| {
            if rhs_valid[i] {
                rhs_rng.random_range(0..u16::MAX as u32)
            } else {
                u32::MAX
            }
        })
        .collect();

    let lhs: Buffer<u32> = raw_lhs.iter().copied().collect();
    let rhs: Buffer<u32> = raw_rhs.iter().copied().collect();

    let lhs_sanitized: Buffer<u32> = (0..n)
        .map(|i| if lhs_valid[i] { raw_lhs[i] } else { 0 })
        .collect();
    let rhs_sanitized: Buffer<u32> = (0..n)
        .map(|i| if rhs_valid[i] { raw_rhs[i] } else { 0 })
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
        lhs_sanitized,
        rhs_sanitized,
        lhs_mask,
        rhs_mask,
        lhs_arrow,
        rhs_arrow,
    }
}

fn alloc_out(n: usize) -> Vec<MaybeUninit<u32>> {
    let mut out = Vec::with_capacity(n);
    // SAFETY: every lane is written before any read inside the kernel.
    unsafe { out.set_len(n) };
    out
}

// ---------------------------------------------------------------------------
// Variant 0: arrow_arith::numeric::add — baseline
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Variant 1: try_map_with_mask + closure `|(a, b), _|` (value-only)
// Fail tracking: bit-pack via the kernel.
// LLVM DCEs per-lane mask extract.
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
// Variant 2: try_map_with_mask + closure `|(a, b), valid|` with if-else
// Fail tracking: bit-pack via the kernel.
// Closure explicitly suppresses null-lane fails (redundant with bit-pack filter).
// ---------------------------------------------------------------------------

#[divan::bench(args = SIZES)]
fn bitpack_closure_suppresses_if_else(bencher: Bencher, n: usize) {
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
                |(a, b), valid| {
                    if valid { a.checked_add(b) } else { Some(0) }
                },
            )
            .unwrap();
            (combined, out)
        });
}

// ---------------------------------------------------------------------------
// Variant 3: try_map_with_mask + closure `.or_else(|| (!valid).then(...))`
// Fail tracking: bit-pack via the kernel.
// Lazy suppression: closure only consults `valid` when overflow actually fires.
// ---------------------------------------------------------------------------

#[divan::bench(args = SIZES)]
fn bitpack_closure_suppresses_or_else(bencher: Bencher, n: usize) {
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
                |(a, b), valid| a.checked_add(b).or_else(|| (!valid).then_some(0)),
            )
            .unwrap();
            (combined, out)
        });
}

// ---------------------------------------------------------------------------
// Variant 4: try_map_with_mask + closure with `(a * valid).checked_add(b * valid)`
// Fail tracking: bit-pack via the kernel.
// The multiply-by-valid trick zeroes null-lane operands so they can't overflow.
// ---------------------------------------------------------------------------

#[divan::bench(args = SIZES)]
fn bitpack_closure_mul_trick(bencher: Bencher, n: usize) {
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
                |(a, b), valid| {
                    let m = valid as u32;
                    (a * m).checked_add(b * m)
                },
            )
            .unwrap();
            (combined, out)
        });
}

// ---------------------------------------------------------------------------
// Variant 5: hand-rolled, boolean fail_acc, closure suppresses nulls, cold replay
// ---------------------------------------------------------------------------

/// Hand-rolled kernel: boolean `fail_acc`, cold replay attribution.
/// Closure is expected to suppress null-lane fails by returning `Some(...)`;
/// `fail_acc` only fires for real valid-lane overflows.
#[inline]
fn handrolled_boolean<F>(
    lhs: &[u32],
    rhs: &[u32],
    mask: &BitBuffer,
    out: &mut [MaybeUninit<u32>],
    mut f: F,
) -> Result<(), usize>
where
    F: FnMut(u32, u32, bool) -> Option<u32>,
{
    let len = lhs.len();
    assert_eq!(len, rhs.len());
    assert_eq!(len, mask.len());
    assert_eq!(len, out.len());
    let chunks = mask.chunks();
    let chunks_count = len / 64;
    let remainder = len % 64;

    for (chunk_idx, src_chunk) in chunks.iter().enumerate() {
        let base = chunk_idx * 64;
        let mut fail_acc: u64 = 0;
        for bit_idx in 0..64 {
            let i = base + bit_idx;
            let bit = (src_chunk >> bit_idx) & 1 == 1;
            // SAFETY: i < len.
            let a = unsafe { *lhs.get_unchecked(i) };
            let b = unsafe { *rhs.get_unchecked(i) };
            let opt = f(a, b, bit);
            fail_acc |= opt.is_none() as u64;
            unsafe { out.get_unchecked_mut(i).write(opt.unwrap_or_default()) };
        }
        if fail_acc != 0 {
            // Cold: find first failing lane (closure already suppressed nulls).
            for bit_idx in 0..64 {
                let i = base + bit_idx;
                let bit = (src_chunk >> bit_idx) & 1 == 1;
                let a = unsafe { *lhs.get_unchecked(i) };
                let b = unsafe { *rhs.get_unchecked(i) };
                if f(a, b, bit).is_none() {
                    return Err(i);
                }
            }
        }
    }

    if remainder != 0 {
        let src_chunk = chunks.remainder_bits();
        let base = chunks_count * 64;
        let mut fail_acc: u64 = 0;
        for bit_idx in 0..remainder {
            let i = base + bit_idx;
            let bit = (src_chunk >> bit_idx) & 1 == 1;
            let a = unsafe { *lhs.get_unchecked(i) };
            let b = unsafe { *rhs.get_unchecked(i) };
            let opt = f(a, b, bit);
            fail_acc |= opt.is_none() as u64;
            unsafe { out.get_unchecked_mut(i).write(opt.unwrap_or_default()) };
        }
        if fail_acc != 0 {
            for bit_idx in 0..remainder {
                let i = base + bit_idx;
                let bit = (src_chunk >> bit_idx) & 1 == 1;
                let a = unsafe { *lhs.get_unchecked(i) };
                let b = unsafe { *rhs.get_unchecked(i) };
                if f(a, b, bit).is_none() {
                    return Err(i);
                }
            }
        }
    }
    Ok(())
}

#[divan::bench(args = SIZES)]
fn boolean_closure_suppresses(bencher: Bencher, n: usize) {
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
            handrolled_boolean(
                lhs.as_slice(),
                rhs.as_slice(),
                &combined,
                out.as_mut_slice(),
                |a, b, valid| {
                    if valid { a.checked_add(b) } else { Some(0) }
                },
            )
            .unwrap();
            (combined, out)
        });
}

// ---------------------------------------------------------------------------
// Variant 6: hand-rolled pre-mask. Kernel zeros null-lane values via bit
// broadcast, then unconditional add + overflow detect. Boolean fail_acc.
// ---------------------------------------------------------------------------

#[inline]
fn handrolled_premask(
    lhs: &[u32],
    rhs: &[u32],
    mask: &BitBuffer,
    out: &mut [MaybeUninit<u32>],
) -> Result<(), usize> {
    let len = lhs.len();
    assert_eq!(len, rhs.len());
    assert_eq!(len, mask.len());
    assert_eq!(len, out.len());
    let chunks = mask.chunks();
    let chunks_count = len / 64;
    let remainder = len % 64;

    for (chunk_idx, src_chunk) in chunks.iter().enumerate() {
        let base = chunk_idx * 64;
        let mut fail_acc: u64 = 0;
        for bit_idx in 0..64 {
            // bit-broadcast: 0 → 0x00000000, 1 → 0xFFFFFFFF
            let lane_mask = (((src_chunk >> bit_idx) & 1) as u32).wrapping_neg();
            let i = base + bit_idx;
            // SAFETY: i < len.
            let a = unsafe { *lhs.get_unchecked(i) } & lane_mask;
            let b = unsafe { *rhs.get_unchecked(i) } & lane_mask;
            let (sum, overflow) = a.overflowing_add(b);
            fail_acc |= overflow as u64;
            unsafe { out.get_unchecked_mut(i).write(sum) };
        }
        if fail_acc != 0 {
            // Cold: walk chunk to find first valid lane that actually overflows on
            // the unmasked inputs. Null lanes were premasked to 0+0, can't overflow.
            for bit_idx in 0..64 {
                let i = base + bit_idx;
                let bit = (src_chunk >> bit_idx) & 1 == 1;
                if !bit {
                    continue;
                }
                let a = unsafe { *lhs.get_unchecked(i) };
                let b = unsafe { *rhs.get_unchecked(i) };
                if a.checked_add(b).is_none() {
                    return Err(i);
                }
            }
        }
    }

    if remainder != 0 {
        let src_chunk = chunks.remainder_bits();
        let base = chunks_count * 64;
        let mut fail_acc: u64 = 0;
        for bit_idx in 0..remainder {
            let lane_mask = (((src_chunk >> bit_idx) & 1) as u32).wrapping_neg();
            let i = base + bit_idx;
            let a = unsafe { *lhs.get_unchecked(i) } & lane_mask;
            let b = unsafe { *rhs.get_unchecked(i) } & lane_mask;
            let (sum, overflow) = a.overflowing_add(b);
            fail_acc |= overflow as u64;
            unsafe { out.get_unchecked_mut(i).write(sum) };
        }
        if fail_acc != 0 {
            for bit_idx in 0..remainder {
                let i = base + bit_idx;
                let bit = (src_chunk >> bit_idx) & 1 == 1;
                if !bit {
                    continue;
                }
                let a = unsafe { *lhs.get_unchecked(i) };
                let b = unsafe { *rhs.get_unchecked(i) };
                if a.checked_add(b).is_none() {
                    return Err(i);
                }
            }
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
// Variant 7: pure SIMD, no mask awareness — CEILING REFERENCE ONLY.
// Incorrect for arrays where null lanes might overflow; benchmarked just to
// show the theoretical floor for nullable add.
// ---------------------------------------------------------------------------

#[inline]
fn handrolled_no_validity(
    lhs: &[u32],
    rhs: &[u32],
    out: &mut [MaybeUninit<u32>],
) -> Result<(), usize> {
    assert_eq!(lhs.len(), rhs.len());
    assert_eq!(lhs.len(), out.len());
    let mut fail = false;
    for i in 0..lhs.len() {
        let a = unsafe { *lhs.get_unchecked(i) };
        let b = unsafe { *rhs.get_unchecked(i) };
        let (sum, overflow) = a.overflowing_add(b);
        fail |= overflow;
        unsafe { out.get_unchecked_mut(i).write(sum) };
    }
    if fail { Err(0) } else { Ok(()) }
}

/// Pure-SIMD ceiling on **pre-sanitized** input (null lanes pre-zeroed in the
/// fixture, outside the timed region). Cannot run on the realistic
/// `(lhs, rhs)` arrays because their null lanes hold `u32::MAX` and would
/// Err — proven by [`assert_pure_simd_errs_on_realistic_data`].
///
/// Showing the SIMD-only arithmetic floor — what an ideal nullable-add would
/// look like if validity could be free.
#[divan::bench(args = SIZES)]
fn pure_simd_no_validity_sanitized(bencher: Bencher, n: usize) {
    let f = fixture(n);
    bencher
        .with_inputs(|| (f.lhs_sanitized.clone(), f.rhs_sanitized.clone()))
        .bench_refs(|(lhs, rhs)| {
            let mut out = alloc_out(n);
            handrolled_no_validity(lhs.as_slice(), rhs.as_slice(), out.as_mut_slice()).unwrap();
            out
        });
}

// ---------------------------------------------------------------------------
// Parity assertions — must pass before divan runs benches.
// ---------------------------------------------------------------------------

/// Both arrow and our kernel must Err on overflow at a valid lane.
fn assert_overflow_parity() {
    let lhs: Vec<u32> = vec![1, 2, u32::MAX, 4];
    let rhs: Vec<u32> = vec![10, 20, 1, 40];
    let valid = vec![true; 4];

    let lhs_arrow = UInt32Array::new(
        ScalarBuffer::from(lhs.clone()),
        Some(NullBuffer::from(valid.clone())),
    );
    let rhs_arrow = UInt32Array::new(
        ScalarBuffer::from(rhs.clone()),
        Some(NullBuffer::from(valid.clone())),
    );
    let arrow_result =
        arrow_arith::numeric::add(&lhs_arrow as &dyn Datum, &rhs_arrow as &dyn Datum);
    assert!(arrow_result.is_err(), "arrow should Err on overflow");

    let mask = {
        let mut m = BitBufferMut::with_capacity(4);
        for &v in &valid {
            m.append(v);
        }
        m.freeze()
    };
    let mut out: Vec<MaybeUninit<u32>> = (0..4).map(|_| MaybeUninit::uninit()).collect();
    let ours = try_map_with_mask(
        LaneZip::new(lhs.as_slice(), rhs.as_slice()),
        &mask,
        out.as_mut_slice(),
        |(a, b), _| a.checked_add(b),
    );
    assert!(ours.is_err(), "bitpack should Err on overflow");

    let mut out2: Vec<MaybeUninit<u32>> = (0..4).map(|_| MaybeUninit::uninit()).collect();
    let boolean = handrolled_boolean(&lhs, &rhs, &mask, &mut out2, |a, b, valid| {
        if valid { a.checked_add(b) } else { Some(0) }
    });
    assert!(boolean.is_err(), "boolean should Err on overflow");

    let mut out3: Vec<MaybeUninit<u32>> = (0..4).map(|_| MaybeUninit::uninit()).collect();
    let prem = handrolled_premask(&lhs, &rhs, &mask, &mut out3);
    assert!(prem.is_err(), "premask should Err on overflow");
}

/// All correctness-preserving variants must NOT Err when only null lanes
/// would overflow. (Pure-SIMD variant is excluded — it doesn't see validity.)
fn assert_null_overflow_suppressed() {
    // Lane 2 is null and contains overflowing values; valid lanes are safe.
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

    // Bit-pack with value-only closure — kernel filters null-lane fails.
    let mut out = alloc_out(4);
    let r = try_map_with_mask(
        LaneZip::new(lhs.as_slice(), rhs.as_slice()),
        &mask,
        out.as_mut_slice(),
        |(a, b), _| a.checked_add(b),
    );
    assert!(r.is_ok(), "bitpack_value_only: null-lane overflow leaked");

    // Boolean with closure that suppresses nulls.
    let mut out = alloc_out(4);
    let r = handrolled_boolean(&lhs, &rhs, &mask, &mut out, |a, b, valid| {
        if valid { a.checked_add(b) } else { Some(0) }
    });
    assert!(r.is_ok(), "boolean_closure_suppresses: null-lane leaked");

    // Pre-mask: kernel zeroes null-lane values.
    let mut out = alloc_out(4);
    let r = handrolled_premask(&lhs, &rhs, &mask, &mut out);
    assert!(r.is_ok(), "premask_then_simd: null-lane overflow leaked");
}

/// Demonstrates that `pure_simd_no_validity` is **incorrect** on realistic
/// fixture inputs — i.e., when null lanes contain values that overflow on add.
/// This is what justifies excluding pure_simd from the realistic bench and
/// running it only on the sanitized inputs. Without this, the "ignore the
/// mask" approach would look too fast because the test data lets it cheat.
fn assert_pure_simd_errs_on_realistic_data() {
    // Lane 2 is a "null lane" in arrow-style storage: bitmap says null, but
    // the data buffer still holds an overflowing value. The realistic
    // `fixture` does exactly this.
    let lhs: Vec<u32> = vec![1, 2, u32::MAX, 4];
    let rhs: Vec<u32> = vec![10, 20, 1, 40];
    let mut out: Vec<MaybeUninit<u32>> = (0..4).map(|_| MaybeUninit::uninit()).collect();

    let r = handrolled_no_validity(&lhs, &rhs, &mut out);
    assert!(
        r.is_err(),
        "pure_simd_no_validity should Err on realistic data (null lane has \
         u32::MAX). If this passes, the bench fixture isn't exercising the \
         unsafe-null-storage case and the pure_simd ceiling number is \
         misleading — it's running on data the kernel happens to handle even \
         without a mask."
    );
}
