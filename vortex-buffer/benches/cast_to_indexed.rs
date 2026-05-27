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
use arrow_cast::cast_with_options;
use arrow_schema::DataType;
use divan::Bencher;
use num_traits::NumCast;
use rand::SeedableRng;
use rand::prelude::*;
use rand::rngs::StdRng;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_buffer::Buffer;
use vortex_buffer::lane_ops_indexed::map_no_validity;
use vortex_buffer::lane_ops_indexed::map_to_bits;
use vortex_buffer::lane_ops_indexed::map_with_mask;
use vortex_buffer::lane_ops_indexed::map_with_mask_in_place;
use vortex_buffer::lane_ops_indexed::map_with_mask_to_bits;
use vortex_buffer::lane_ops_indexed::try_map_no_validity;
use vortex_buffer::lane_ops_indexed::try_map_validity_filtered;
use vortex_buffer::lane_ops_indexed::try_map_with_mask;
use vortex_buffer::lane_ops_indexed::try_map_with_mask_in_place;

fn main() {
    divan::main();
}

const SIZES: &[usize] = &[4_096, 65_536, 1_048_576];
const U32_THRESHOLD: u32 = u32::MAX / 2;

struct Fixture {
    values_u64: Buffer<u64>,
    values_u64_invalid_overflows: Buffer<u64>,
    values_u32: Buffer<u32>,
    values_u32_small: Buffer<u32>,
    values_u16: Buffer<u16>,
    mask: BitBuffer,
    /// `UInt64Array` baseline for arrow casts. Same values + validity as `values_u64` / `mask`.
    arrow_u64: UInt64Array,
    /// `UInt16Array` baseline. Same as `values_u16` / `mask`.
    arrow_u16: UInt16Array,
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
        values_u64_invalid_overflows,
        values_u32,
        values_u32_small,
        values_u16,
        mask: BitBufferMut::from_iter(raw_valid).freeze(),
        arrow_u64,
        arrow_u16,
    }
}

const CAST_OPTS_CHECKED: CastOptions<'static> = CastOptions {
    safe: false,
    format_options: arrow_cast::display::FormatOptions::new(),
};

fn uninit_out<T>(n: usize) -> Vec<MaybeUninit<T>> {
    let mut out = Vec::with_capacity(n);
    // SAFETY: A `MaybeUninit<T>` does not require initialization.
    unsafe {
        out.set_len(n);
    }
    out
}

#[divan::bench(args = SIZES)]
fn map_no_validity_widen_u16_u32(bencher: Bencher, n: usize) {
    let f = fixture(n);

    bencher
        .with_inputs(|| (f.values_u16.clone(), uninit_out::<u32>(n)))
        .bench_values(|(values, mut out)| {
            map_no_validity(
                values.as_slice(),
                out.as_mut_slice(),
                <u32 as From<u16>>::from,
            );
            out
        });
}

#[divan::bench(args = SIZES)]
fn map_with_mask_widen_u16_u32_zero_nulls(bencher: Bencher, n: usize) {
    let f = fixture(n);

    bencher
        .with_inputs(|| (f.values_u16.clone(), f.mask.clone(), uninit_out::<u32>(n)))
        .bench_values(|(values, mask, mut out)| {
            map_with_mask(values.as_slice(), &mask, out.as_mut_slice(), |v, valid| {
                <u32 as From<u16>>::from(v) * valid as u32
            });
            out
        });
}

#[divan::bench(args = SIZES)]
fn try_map_no_validity_narrow_u64_u32(bencher: Bencher, n: usize) {
    let f = fixture(n);

    bencher
        .with_inputs(|| (f.values_u64.clone(), uninit_out::<u32>(n)))
        .bench_values(|(values, mut out)| {
            try_map_no_validity(values.as_slice(), out.as_mut_slice(), |v| {
                <u32 as NumCast>::from(v)
            })
            .unwrap();
            out
        });
}

/// `try_map_with_mask` with a closure that **ignores `valid`**. Tests whether
/// LLVM DCEs the per-lane `(src_chunk >> bit_idx) & 1` mask extract. Uses
/// non-overflowing `values_u64` so the closure-ignores-valid spurious-failure
/// case never triggers (would otherwise err on null-lane overflow).
#[divan::bench(args = SIZES)]
fn try_map_with_mask_narrow_u64_u32_ignoring_valid(bencher: Bencher, n: usize) {
    let f = fixture(n);

    bencher
        .with_inputs(|| (f.values_u64.clone(), f.mask.clone(), uninit_out::<u32>(n)))
        .bench_values(|(values, mask, mut out)| {
            try_map_with_mask(values.as_slice(), &mask, out.as_mut_slice(), |v, _valid| {
                <u32 as NumCast>::from(v)
            })
            .unwrap();
            out
        });
}

#[divan::bench(args = SIZES)]
fn try_map_with_mask_narrow_u64_u32_lazy_validity(bencher: Bencher, n: usize) {
    let f = fixture(n);

    bencher
        .with_inputs(|| (f.values_u64.clone(), f.mask.clone(), uninit_out::<u32>(n)))
        .bench_values(|(values, mask, mut out)| {
            try_map_with_mask(values.as_slice(), &mask, out.as_mut_slice(), |v, valid| {
                <u32 as NumCast>::from(v).or_else(|| (!valid).then(u32::default))
            })
            .unwrap();
            out
        });
}

#[divan::bench(args = SIZES)]
fn try_map_validity_filtered_narrow_u64_u32(bencher: Bencher, n: usize) {
    let f = fixture(n);

    bencher
        .with_inputs(|| {
            (
                f.values_u64_invalid_overflows.clone(),
                f.mask.clone(),
                uninit_out::<u32>(n),
            )
        })
        .bench_values(|(values, mask, mut out)| {
            try_map_validity_filtered(values.as_slice(), &mask, out.as_mut_slice(), |v| {
                <u32 as NumCast>::from(v)
            })
            .unwrap();
            out
        });
}

#[divan::bench(args = SIZES)]
fn try_map_with_mask_widen_u16_u32_or_else(bencher: Bencher, n: usize) {
    let f = fixture(n);

    bencher
        .with_inputs(|| (f.values_u16.clone(), f.mask.clone(), uninit_out::<u32>(n)))
        .bench_values(|(values, mask, mut out)| {
            try_map_with_mask(values.as_slice(), &mask, out.as_mut_slice(), |v, valid| {
                Some(<u32 as From<u16>>::from(v)).or_else(|| (!valid).then(u32::default))
            })
            .unwrap();
            out
        });
}

#[divan::bench(args = SIZES)]
fn try_map_with_mask_widen_u16_u32_maskless(bencher: Bencher, n: usize) {
    let f = fixture(n);

    bencher
        .with_inputs(|| (f.values_u16.clone(), f.mask.clone(), uninit_out::<u32>(n)))
        .bench_values(|(values, mask, mut out)| {
            try_map_with_mask(values.as_slice(), &mask, out.as_mut_slice(), |v, _valid| {
                Some(<u32 as From<u16>>::from(v))
            })
            .unwrap();
            out
        });
}

#[divan::bench(args = SIZES)]
fn map_with_mask_in_place_u32_zero_nulls(bencher: Bencher, n: usize) {
    let f = fixture(n);

    bencher
        .with_inputs(|| (f.values_u32.as_slice().to_vec(), f.mask.clone()))
        .bench_values(|(mut values, mask)| {
            map_with_mask_in_place(values.as_mut_slice(), &mask, |v, valid| v * valid as u32);
            values
        });
}

#[divan::bench(args = SIZES)]
fn try_map_with_mask_in_place_u32_checked_mul(bencher: Bencher, n: usize) {
    let f = fixture(n);

    bencher
        .with_inputs(|| (f.values_u32_small.as_slice().to_vec(), f.mask.clone()))
        .bench_values(|(mut values, mask)| {
            try_map_with_mask_in_place(values.as_mut_slice(), &mask, |v, _valid| v.checked_mul(2))
                .unwrap();
            values
        });
}

#[divan::bench(args = SIZES)]
fn map_to_bits_u32_threshold(bencher: Bencher, n: usize) {
    let f = fixture(n);

    bencher
        .with_inputs(|| (f.values_u32.clone(), vec![0; n.div_ceil(64)]))
        .bench_values(|(values, mut out)| {
            map_to_bits(values.as_slice(), out.as_mut_slice(), |v| {
                v >= U32_THRESHOLD
            });
            out
        });
}

#[divan::bench(args = SIZES)]
fn map_with_mask_to_bits_u32_threshold(bencher: Bencher, n: usize) {
    let f = fixture(n);

    bencher
        .with_inputs(|| {
            (
                f.values_u32.clone(),
                f.mask.clone(),
                vec![0; n.div_ceil(64)],
            )
        })
        .bench_values(|(values, mask, mut out)| {
            map_with_mask_to_bits(values.as_slice(), &mask, out.as_mut_slice(), |v, valid| {
                valid && v >= U32_THRESHOLD
            });
            out
        });
}

// -----------------------------------------------------------------------------
// Arrow-rs baselines. Two: one widening (u16 → u32, always succeeds) and one
// narrowing (u64 → u32, can fail). Each pairs with the cast variants above of
// matching direction.
// -----------------------------------------------------------------------------

#[divan::bench(args = SIZES)]
fn arrow_cast_widen_u16_u32(bencher: Bencher, _n: usize) {
    let f = fixture(_n);
    bencher
        .with_inputs(|| f.arrow_u16.clone())
        .bench_refs(|arr| cast_with_options(arr, &DataType::UInt32, &CAST_OPTS_CHECKED).unwrap());
}

#[divan::bench(args = SIZES)]
fn arrow_cast_narrow_u64_u32(bencher: Bencher, _n: usize) {
    let f = fixture(_n);
    bencher
        .with_inputs(|| f.arrow_u64.clone())
        .bench_refs(|arr| cast_with_options(arr, &DataType::UInt32, &CAST_OPTS_CHECKED).unwrap());
}
