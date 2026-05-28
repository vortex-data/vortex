// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Coverage benchmark for the lane-kernel variants used by primitive casts,
//! bit-packing paths, and `LaneZip` binary kernels.
//!
//! `add_checked` parity assertions (run at startup) verify that the bit-packed
//! fail-tracking scheme:
//!   - propagates valid-lane overflow as `Err`, and
//!   - suppresses null-lane overflow without the closure ever inspecting `valid`.
//!
//! Each Vortex kernel bench has a sibling `arrow_*` baseline bench using the
//! equivalent arrow-rs kernel over the same data shape, so the divan report
//! lines up side-by-side.

#![expect(clippy::unwrap_used)]
#![expect(clippy::clone_on_ref_ptr)]

use std::mem::MaybeUninit;
use std::sync::Arc;

use arrow_arith::numeric::add;
use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::Int32Array;
use arrow_array::UInt16Array;
use arrow_array::UInt32Array;
use arrow_array::UInt64Array;
use arrow_buffer::NullBuffer;
use arrow_cast::CastOptions;
use arrow_cast::cast_with_options;
use arrow_schema::DataType;
use divan::Bencher;
use num_traits::AsPrimitive;
use num_traits::NumCast;
use rand::SeedableRng;
use rand::prelude::*;
use rand::rngs::StdRng;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_buffer::Buffer;
use vortex_compute::lane_kernels::IndexedSinkExt;
use vortex_compute::lane_kernels::IndexedSourceExt;
use vortex_compute::lane_kernels::LaneZip;
use vortex_compute::lane_kernels::ReinterpretSink;

fn main() {
    assert_overflow_parity();
    assert_null_overflow_suppressed();
    divan::main();
}

const SIZES: &[usize] = &[65_536];

// -----------------------------------------------------------------------------
// Cast fixture (u64/u16/i32 lanes + a single validity mask).
// -----------------------------------------------------------------------------

struct CastFixture {
    values_u64: Buffer<u64>,
    values_u16: Buffer<u16>,
    /// Positive `i32` values (always representable as `u32`). Used by the
    /// in-place-vs-out-of-place cast bench.
    values_i32: Buffer<i32>,
    mask: BitBuffer,
    /// Validity as a plain `Vec<bool>` — the source of truth used to build both
    /// the Vortex `BitBuffer` mask and the arrow `NullBuffer`.
    valid: Vec<bool>,
}

fn cast_fixture(n: usize) -> CastFixture {
    let mut rng = StdRng::seed_from_u64(0xC457_1D3E);

    let raw_values: Vec<u64> = (0..n)
        .map(|_| rng.random_range(0..(u32::MAX as u64)))
        .collect();
    let raw_valid: Vec<bool> = (0..n).map(|_| rng.random_bool(0.8)).collect();

    #[expect(clippy::cast_possible_truncation)]
    let values_u16 = raw_values
        .iter()
        .copied()
        .map(|v| v as u16)
        .collect::<Buffer<u16>>();

    // Positive i32 values (top bit cleared) — every value fits in u32.
    #[expect(clippy::cast_possible_truncation)]
    let values_i32 = raw_values
        .iter()
        .copied()
        .map(|v| (v as i32) & i32::MAX)
        .collect::<Buffer<i32>>();

    CastFixture {
        values_u64: raw_values.into(),
        values_u16,
        values_i32,
        mask: BitBufferMut::from_iter(raw_valid.iter().copied()).freeze(),
        valid: raw_valid,
    }
}

fn uninit_out<T>(n: usize) -> Vec<MaybeUninit<T>> {
    let mut out = Vec::with_capacity(n);
    // SAFETY: A `MaybeUninit<T>` does not require initialization.
    unsafe {
        out.set_len(n);
    }
    out
}

// -----------------------------------------------------------------------------
// Cast benches (single-input, source -> output).
// -----------------------------------------------------------------------------

#[divan::bench(args = SIZES)]
fn try_map_into_narrow_u64_u32(bencher: Bencher, n: usize) {
    let f = cast_fixture(n);

    bencher
        .with_inputs(|| (f.values_u64.clone(), uninit_out::<u32>(n)))
        .bench_values(|(values, mut out)| {
            values
                .as_slice()
                .try_map_into(out.as_mut_slice(), <u32 as NumCast>::from)
                .unwrap();
            out
        });
}

#[divan::bench(args = SIZES)]
fn arrow_narrow_u64_u32(bencher: Bencher, n: usize) {
    let f = cast_fixture(n);
    let arr: ArrowArrayRef = Arc::new(UInt64Array::from(f.values_u64.as_slice().to_vec()));
    let opts = CastOptions {
        safe: false,
        ..CastOptions::default()
    };

    bencher
        .with_inputs(|| arr.clone())
        .bench_values(|arr| cast_with_options(&arr, &DataType::UInt32, &opts).unwrap());
}

#[divan::bench(args = SIZES)]
fn map_with_mask_narrow_u64_u32(bencher: Bencher, n: usize) {
    let f = cast_fixture(n);

    bencher
        .with_inputs(|| (f.values_u64.clone(), uninit_out::<u32>(n)))
        .bench_values(|(values, mut out)| {
            values.as_slice().map_into(&mut out, |v| v.as_());
            out
        });
}

/// `try_map_masked_into_widen_u16_u32` and `map_with_mask_widen_u16_u32` have the
/// same runtime — for always-true map operations `try_map_masked_into` is
/// sufficient.
#[divan::bench(args = SIZES)]
fn try_map_masked_into_widen_u16_u32(bencher: Bencher, n: usize) {
    let f = cast_fixture(n);

    bencher
        .with_inputs(|| (f.values_u16.clone(), f.mask.clone(), uninit_out::<u32>(n)))
        .bench_values(|(values, mask, mut out)| {
            values
                .as_slice()
                .try_map_masked_into(&mask, out.as_mut_slice(), <u32 as NumCast>::from)
                .unwrap();
            out
        });
}

#[divan::bench(args = SIZES)]
fn map_with_mask_widen_u16_u32(bencher: Bencher, n: usize) {
    let f = cast_fixture(n);

    bencher
        .with_inputs(|| (f.values_u16.clone(), uninit_out::<u32>(n)))
        .bench_values(|(values, mut out)| {
            values.as_slice().map_into(out.as_mut_slice(), |v| v.as_());
        });
}

#[divan::bench(args = SIZES)]
fn arrow_widen_u16_u32(bencher: Bencher, n: usize) {
    let f = cast_fixture(n);
    let nulls = NullBuffer::from(f.valid.clone());
    let arr: ArrowArrayRef = Arc::new(UInt16Array::new(
        f.values_u16.as_slice().to_vec().into(),
        Some(nulls),
    ));

    bencher.with_inputs(|| arr.clone()).bench_values(|arr| {
        cast_with_options(&arr, &DataType::UInt32, &CastOptions::default()).unwrap()
    });
}

// -----------------------------------------------------------------------------
// In-place vs out-of-place fallible cast i32 → u32 (same byte width).
//
// `try_map_masked_in_place` mutates the input via `ReinterpretSink` and
// transmutes the wrapper — no output allocation. `try_map_masked_into` allocates
// a fresh `BufferMut<u32>` and writes through it. Input values are all positive
// `i32` so every lane succeeds; the two kernels do the same arithmetic, so any
// delta is allocation + memory-traffic overhead.
// -----------------------------------------------------------------------------

#[divan::bench(args = SIZES)]
fn try_map_masked_into_narrow_i32_u32(bencher: Bencher, n: usize) {
    let f = cast_fixture(n);

    bencher
        .with_inputs(|| (f.values_i32.clone(), f.mask.clone(), uninit_out::<u32>(n)))
        .bench_values(|(values, mask, mut out)| {
            values
                .as_slice()
                .try_map_masked_into(&mask, out.as_mut_slice(), <u32 as NumCast>::from)
                .unwrap();
            out
        });
}

#[divan::bench(args = SIZES)]
fn try_map_masked_in_place_narrow_i32_u32(bencher: Bencher, n: usize) {
    let f = cast_fixture(n);

    bencher
        .with_inputs(|| (f.values_i32.as_slice().to_vec(), f.mask.clone()))
        .bench_values(|(mut values, mask)| {
            ReinterpretSink::<i32, u32>::new(values.as_mut_slice())
                .try_map_masked_in_place(&mask, <u32 as NumCast>::from)
                .unwrap();
            values
        });
}

#[divan::bench(args = SIZES)]
fn arrow_narrow_i32_u32(bencher: Bencher, n: usize) {
    let f = cast_fixture(n);
    let nulls = NullBuffer::from(f.valid.clone());
    let arr: ArrowArrayRef = Arc::new(Int32Array::new(
        f.values_i32.as_slice().to_vec().into(),
        Some(nulls),
    ));
    let opts = CastOptions {
        safe: false,
        ..CastOptions::default()
    };

    bencher
        .with_inputs(|| arr.clone())
        .bench_values(|arr| cast_with_options(&arr, &DataType::UInt32, &opts).unwrap());
}

// -----------------------------------------------------------------------------
// LaneZip binary kernel: checked `u32 + u32 -> u32` over two nullable columns.
//
// Per-lane `is_none()` flags are bit-packed and AND-ed with the chunk validity
// word, so null-lane overflow is filtered without the closure inspecting `valid`.
// Verified at startup via parity assertions (`assert_overflow_parity` and
// `assert_null_overflow_suppressed`).
// -----------------------------------------------------------------------------

const ADD_LHS_VALID_RATE: f64 = 0.7;
const ADD_RHS_VALID_RATE: f64 = 0.8;

struct AddFixture {
    /// Valid lanes carry bounded values; null lanes hold `u32::MAX` so a kernel
    /// that ignores validity would `Err` on them. The implementation under test
    /// must suppress that.
    lhs: Buffer<u32>,
    rhs: Buffer<u32>,
    lhs_mask: BitBuffer,
    rhs_mask: BitBuffer,
    /// Plain `Vec<bool>` mirrors of the validity masks — used to build the arrow
    /// `NullBuffer`s for the baseline bench.
    lhs_valid: Vec<bool>,
    rhs_valid: Vec<bool>,
}

fn add_fixture(n: usize) -> AddFixture {
    let mut lhs_rng = StdRng::seed_from_u64(0);
    let mut rhs_rng = StdRng::seed_from_u64(1);
    let mut lvr = StdRng::seed_from_u64(2);
    let mut rvr = StdRng::seed_from_u64(3);

    let lhs_valid: Vec<bool> = (0..n)
        .map(|_| lvr.random_bool(ADD_LHS_VALID_RATE))
        .collect();
    let rhs_valid: Vec<bool> = (0..n)
        .map(|_| rvr.random_bool(ADD_RHS_VALID_RATE))
        .collect();

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

    let lhs_mask = BitBufferMut::from_iter(lhs_valid.iter().copied()).freeze();
    let rhs_mask = BitBufferMut::from_iter(rhs_valid.iter().copied()).freeze();

    AddFixture {
        lhs,
        rhs,
        lhs_mask,
        rhs_mask,
        lhs_valid,
        rhs_valid,
    }
}

#[divan::bench(args = SIZES)]
fn lanezip_checked_add_u32(bencher: Bencher, n: usize) {
    let f = add_fixture(n);
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
            let mut out = uninit_out::<u32>(n);
            LaneZip::new(lhs.as_slice(), rhs.as_slice())
                .try_map_masked_into(&combined, out.as_mut_slice(), |(a, b)| a.checked_add(b))
                .unwrap();
            (combined, out)
        });
}

#[divan::bench(args = SIZES)]
fn arrow_checked_add_u32(bencher: Bencher, n: usize) {
    let f = add_fixture(n);
    let lhs_arr: ArrowArrayRef = Arc::new(UInt32Array::new(
        f.lhs.as_slice().to_vec().into(),
        Some(NullBuffer::from(f.lhs_valid.clone())),
    ));
    let rhs_arr: ArrowArrayRef = Arc::new(UInt32Array::new(
        f.rhs.as_slice().to_vec().into(),
        Some(NullBuffer::from(f.rhs_valid.clone())),
    ));

    bencher
        .with_inputs(|| (lhs_arr.clone(), rhs_arr.clone()))
        .bench_values(|(lhs, rhs)| add(&lhs, &rhs).unwrap());
}

// -----------------------------------------------------------------------------
// Parity assertions — must pass before divan runs benches.
// -----------------------------------------------------------------------------

/// Overflow at a valid lane must propagate as `Err`.
fn assert_overflow_parity() {
    let lhs: Vec<u32> = vec![1, 2, u32::MAX, 4];
    let rhs: Vec<u32> = vec![10, 20, 1, 40];
    let valid = vec![true; 4];

    let mask = BitBufferMut::from_iter(valid).freeze();
    let mut out: Vec<MaybeUninit<u32>> = (0..4).map(|_| MaybeUninit::uninit()).collect();
    let r = LaneZip::new(lhs.as_slice(), rhs.as_slice()).try_map_masked_into(
        &mask,
        out.as_mut_slice(),
        |(a, b)| a.checked_add(b),
    );
    assert!(r.is_err(), "bitpack should Err on overflow");
}

/// Overflow at a null lane must NOT propagate.
fn assert_null_overflow_suppressed() {
    // Lane 2 is null and holds an overflowing value; valid lanes are safe.
    let lhs: Vec<u32> = vec![1, 2, u32::MAX, 4];
    let rhs: Vec<u32> = vec![10, 20, 1, 40];
    let valid = vec![true, true, false, true];

    let mask = BitBufferMut::from_iter(valid).freeze();
    let mut out = uninit_out::<u32>(4);
    let r = LaneZip::new(lhs.as_slice(), rhs.as_slice()).try_map_masked_into(
        &mask,
        out.as_mut_slice(),
        |(a, b)| a.checked_add(b),
    );
    assert!(r.is_ok(), "bitpack: null-lane overflow leaked");
}
