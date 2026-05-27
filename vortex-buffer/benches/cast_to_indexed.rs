// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Coverage benchmark for the indexed lane-op variants used by primitive casts
//! and bit-packing paths.

#![expect(clippy::unwrap_used)]

use std::mem::MaybeUninit;

use arrow_array::UInt16Array;
use arrow_array::UInt64Array;
use arrow_buffer::NullBuffer;
use arrow_buffer::ScalarBuffer;
use arrow_cast::CastOptions;
use divan::Bencher;
use num_traits::AsPrimitive;
use num_traits::NumCast;
use rand::SeedableRng;
use rand::prelude::*;
use rand::rngs::StdRng;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_buffer::Buffer;
use vortex_buffer::lane_ops_indexed::IndexedSinkExt;
use vortex_buffer::lane_ops_indexed::IndexedSourceExt;
use vortex_buffer::lane_ops_indexed::ReinterpretSink;

fn main() {
    divan::main();
}

const SIZES: &[usize] = &[65_536];

struct Fixture {
    values_u64: Buffer<u64>,
    values_u16: Buffer<u16>,
    /// Positive `i32` values (always representable as `u32`). Used by the
    /// in-place-vs-out-of-place cast bench.
    values_i32: Buffer<i32>,
    mask: BitBuffer,
}

fn fixture(n: usize) -> Fixture {
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
    #[expect(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    let values_i32 = raw_values
        .iter()
        .copied()
        .map(|v| (v as i32) & i32::MAX)
        .collect::<Buffer<i32>>();

    #[expect(clippy::cast_possible_truncation)]
    let values_u32 = raw_values
        .iter()
        .copied()
        .map(|v| v as u32)
        .collect::<Buffer<u32>>();

    #[expect(clippy::cast_possible_truncation)]
    let values_u32_small = raw_values
        .iter()
        .copied()
        .map(|v| (v % ((u32::MAX as u64) / 2)) as u32)
        .collect::<Buffer<u32>>();

    let values_u64_invalid_overflows = raw_values
        .iter()
        .copied()
        .zip(raw_valid.iter().copied())
        .map(|(v, valid)| if valid { v } else { u64::MAX })
        .collect::<Buffer<u64>>();

    let arrow_u64 = UInt64Array::new(
        ScalarBuffer::from(raw_values.clone()),
        Some(NullBuffer::from(raw_valid.clone())),
    );
    #[expect(clippy::cast_possible_truncation)]
    let raw_u16: Vec<u16> = raw_values.iter().map(|&v| v as u16).collect();
    let arrow_u16 = UInt16Array::new(
        ScalarBuffer::from(raw_u16),
        Some(NullBuffer::from(raw_valid.clone())),
    );

    Fixture {
        values_u64: raw_values.into(),
        values_u16,
        values_i32,
        mask: BitBufferMut::from_iter(raw_valid).freeze(),
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

#[divan::bench(args = SIZES)]
fn try_map_into_narrow_u64_u32(bencher: Bencher, n: usize) {
    let f = fixture(n);

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
fn map_with_mask_narrow_u64_u32(bencher: Bencher, n: usize) {
    let f = fixture(n);

    bencher
        .with_inputs(|| (f.values_u64.clone(), uninit_out::<u32>(n)))
        .bench_values(|(values, mut out)| {
            values.as_slice().map_into(&mut out, |v| v.as_());
            out
        });
}

/// `try_map_masked_into_widen_u16_u32` and `map_with_mask_widen_u16_u32` have the same runtime
/// and showing for always true map operations `try_map_masked_into` is sufficient.
#[divan::bench(args = SIZES)]
fn try_map_masked_into_widen_u16_u32(bencher: Bencher, n: usize) {
    let f = fixture(n);

    bencher
        .with_inputs(|| (f.values_u16.clone(), f.mask.clone(), uninit_out::<u32>(n)))
        .bench_values(|(values, mask, mut out)| {
            values
                .as_slice()
                .try_map_masked_into(&mask, out.as_mut_slice(), |v| <u32 as NumCast>::from(v))
                .unwrap();
            out
        });
}

#[divan::bench(args = SIZES)]
fn map_with_mask_widen_u16_u32(bencher: Bencher, n: usize) {
    let f = fixture(n);

    bencher
        .with_inputs(|| (f.values_u16.clone(), uninit_out::<u32>(n)))
        .bench_values(|(values, mut out)| {
            values.as_slice().map_into(out.as_mut_slice(), |v| v.as_());
            out
        });
}

// -----------------------------------------------------------------------------
// In-place vs out-of-place fallible cast i32 → u32 (same byte width).
//
// `try_map_masked_into_in_place` mutates the input via `ReinterpretSink` and
// transmutes the wrapper — no output allocation. `try_map_masked_into` allocates
// a fresh `BufferMut<u32>` and writes through it. Input values are all positive
// `i32` so every lane succeeds; the two kernels do the same arithmetic, so any
// delta is pure allocation + memory-traffic overhead.
// -----------------------------------------------------------------------------

#[divan::bench(args = SIZES)]
fn try_map_masked_into_narrow_i32_u32(bencher: Bencher, n: usize) {
    let f = fixture(n);

    bencher
        .with_inputs(|| (f.values_i32.clone(), f.mask.clone(), uninit_out::<u32>(n)))
        .bench_values(|(values, mask, mut out)| {
            values
                .as_slice()
                .try_map_masked_into(&mask, out.as_mut_slice(), |v| <u32 as NumCast>::from(v))
                .unwrap();
            out
        });
}

#[divan::bench(args = SIZES)]
fn try_map_masked_into_in_place_narrow_i32_u32(bencher: Bencher, n: usize) {
    let f = fixture(n);

    bencher
        .with_inputs(|| (f.values_i32.as_slice().to_vec(), f.mask.clone()))
        .bench_values(|(mut values, mask)| {
            ReinterpretSink::<i32, u32>::new(values.as_mut_slice())
                .try_map_masked_in_place(&mask, |v| <u32 as NumCast>::from(v))
                .unwrap();
            values
        });
}
