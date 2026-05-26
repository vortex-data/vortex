// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Cast `u64 -> u32` over a nullable column, three ways:
//!
//! 1. `kernel_map_with_mask` — uses `map_with_mask`. Writes truncated values into a
//!    pre-allocated `&mut [MaybeUninit<u32>]`. Null lanes write `0` via the branchless
//!    `v * valid as u64` trick, mirroring `primitive/compute/cast.rs:147`.
//! 2. `iter_zip` — `values.iter().zip(mask.iter())` collected through
//!    `BufferMut::from_trusted_len_iter`. This is the shape the current Vortex cast uses.
//! 3. `arrow_cast` — `arrow_cast::cast` against a `UInt64Array`, allocating a new
//!    `UInt32Array`.
//!
//! Plus two fallible variants that error on overflow:
//!
//! 4. `kernel_try_map_with_mask` — `try_map_with_mask` with `|v, valid| (v <= MAX).then_some(...)`.
//!    Unconditional cast + parallel range check OR-reduced into a u64 fail accumulator.
//! 5. `iter_zip_checked` — `BufferMut::try_from_trusted_len_iter` returning Err on overflow.
//! 6. `arrow_cast_checked` — `arrow_cast::cast` with `safe = false` (errors on overflow).
//!
//! Inputs are bounded to fit in `u32`, so the fallible variants always succeed and we
//! measure the cost of the range check on the success path.

#![expect(clippy::unwrap_used)]

use std::mem::MaybeUninit;

use arrow_array::UInt64Array;
use arrow_buffer::NullBuffer;
use arrow_buffer::ScalarBuffer;
use arrow_cast::CastOptions;
use arrow_cast::cast_with_options;
use arrow_schema::DataType;
use divan::Bencher;
use rand::SeedableRng;
use rand::prelude::*;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_buffer::map_with_mask;
use vortex_buffer::try_map_with_mask;

fn main() {
    divan::main();
}

const SIZES: &[usize] = &[4_096, 65_536, 1_048_576];
const VALID_RATE: f64 = 0.7;
const DATA_SEED: u64 = 0;
const VALID_SEED: u64 = 1;

// Non-byte-aligned bit offset → forces BitChunks::iter() to shift across byte
// boundaries on every chunk it yields.
const SLICE_OFFSET: usize = 5;

struct Fixture {
    values: Buffer<u64>,
    /// `offset() == 0`, underlying byte buffer starts on a byte boundary.
    mask_aligned: BitBuffer,
    /// Same validity bits but sliced so `offset() == SLICE_OFFSET`.
    mask_unaligned: BitBuffer,
    arrow_arr: UInt64Array,
    /// Same as `arrow_arr` but its NullBuffer has a non-byte-aligned bit offset,
    /// constructed by building an oversized array and slicing.
    arrow_arr_unaligned: UInt64Array,
}

fn fixture(n: usize) -> Fixture {
    let mut data_rng = StdRng::seed_from_u64(DATA_SEED);
    let mut valid_rng = StdRng::seed_from_u64(VALID_SEED);
    let raw_values: Vec<u64> = (0..n)
        .map(|_| data_rng.random_range(0..u32::MAX as u64))
        .collect();
    let raw_valid: Vec<bool> = (0..n).map(|_| valid_rng.random_bool(VALID_RATE)).collect();

    let values: Buffer<u64> = raw_values.iter().copied().collect();

    let mask_aligned = {
        let mut m = BitBufferMut::with_capacity(n);
        for &v in &raw_valid {
            m.append(v);
        }
        m.freeze()
    };

    // Build n + SLICE_OFFSET bits then slice off the leading SLICE_OFFSET, so the
    // remaining `n` lanes carry the SAME validity pattern as the aligned mask.
    let mask_unaligned = {
        let mut m = BitBufferMut::with_capacity(n + SLICE_OFFSET);
        for _ in 0..SLICE_OFFSET {
            m.append(false); // filler — sliced away
        }
        for &v in &raw_valid {
            m.append(v);
        }
        m.freeze().slice(SLICE_OFFSET..SLICE_OFFSET + n)
    };
    debug_assert_eq!(mask_unaligned.offset(), SLICE_OFFSET);
    debug_assert_eq!(mask_unaligned.len(), n);

    let arrow_arr = UInt64Array::new(
        ScalarBuffer::from(raw_values.clone()),
        Some(NullBuffer::from(raw_valid.clone())),
    );

    // Oversized array → slice off SLICE_OFFSET lanes so the resulting array's
    // NullBuffer has `offset() == SLICE_OFFSET`. The remaining `n` lanes hold the
    // same validity pattern as `arrow_arr`.
    let arrow_arr_unaligned = {
        let mut padded_values: Vec<u64> = vec![0; SLICE_OFFSET];
        padded_values.extend_from_slice(&raw_values);
        let mut padded_valid: Vec<bool> = vec![false; SLICE_OFFSET];
        padded_valid.extend_from_slice(&raw_valid);
        let oversized = UInt64Array::new(
            ScalarBuffer::from(padded_values),
            Some(NullBuffer::from(padded_valid)),
        );
        use arrow_array::Array;
        let sliced = oversized.slice(SLICE_OFFSET, n);
        debug_assert_eq!(
            sliced.nulls().map(|n| n.offset()).unwrap_or(0) % 8,
            SLICE_OFFSET
        );
        sliced
    };

    Fixture {
        values,
        mask_aligned,
        mask_unaligned,
        arrow_arr,
        arrow_arr_unaligned,
    }
}

const CAST_OPTS: CastOptions<'static> = CastOptions {
    safe: true,
    format_options: arrow_cast::display::FormatOptions::new(),
};

const CAST_OPTS_CHECKED: CastOptions<'static> = CastOptions {
    safe: false,
    format_options: arrow_cast::display::FormatOptions::new(),
};

#[divan::bench(args = SIZES)]
fn kernel_map_with_mask(bencher: Bencher, n: usize) {
    let f = fixture(n);
    bencher
        .with_inputs(|| {
            // Owned uninit-slot vector, sized once outside the timed region.
            let mut out: Vec<MaybeUninit<u32>> = Vec::with_capacity(n);
            // SAFETY: every lane is written before any read inside the kernel.
            unsafe { out.set_len(n) };
            (f.values.clone(), f.mask_aligned.clone(), out)
        })
        .bench_refs(|(values, mask, out)| {
            map_with_mask(values.as_slice(), mask, out.as_mut_slice(), |v, valid| {
                (v * valid as u64) as u32
            });
        });
}

#[divan::bench(args = SIZES)]
fn arrow_cast(bencher: Bencher, n: usize) {
    let _ = n;
    let f = fixture(n);
    bencher
        .with_inputs(|| f.arrow_arr.clone())
        .bench_refs(|arr| cast_with_options(arr, &DataType::UInt32, &CAST_OPTS).unwrap());
}

#[divan::bench(args = SIZES)]
fn arrow_cast_unaligned(bencher: Bencher, n: usize) {
    let _ = n;
    let f = fixture(n);
    bencher
        .with_inputs(|| f.arrow_arr_unaligned.clone())
        .bench_refs(|arr| cast_with_options(arr, &DataType::UInt32, &CAST_OPTS).unwrap());
}

#[divan::bench(args = SIZES)]
fn kernel_try_map_with_mask(bencher: Bencher, n: usize) {
    let f = fixture(n);
    bencher
        .with_inputs(|| {
            let mut out: Vec<MaybeUninit<u32>> = Vec::with_capacity(n);
            // SAFETY: every lane is written before any read inside the kernel.
            unsafe { out.set_len(n) };
            (f.values.clone(), f.mask_aligned.clone(), out)
        })
        .bench_refs(|(values, mask, out)| {
            try_map_with_mask(values.as_slice(), mask, out.as_mut_slice(), |v, valid| {
                let scaled = v * valid as u64;
                (scaled <= u32::MAX as u64).then_some(scaled as u32)
            })
            .unwrap();
        });
}

/// Same kernel, but the mask has `offset() == 5` so `BitChunks::iter()` must shift
/// across byte boundaries on every chunk. Quantifies the cost of unaligned mask access.
#[divan::bench(args = SIZES)]
fn kernel_try_map_with_mask_unaligned(bencher: Bencher, n: usize) {
    let f = fixture(n);
    bencher
        .with_inputs(|| {
            let mut out: Vec<MaybeUninit<u32>> = Vec::with_capacity(n);
            unsafe { out.set_len(n) };
            (f.values.clone(), f.mask_unaligned.clone(), out)
        })
        .bench_refs(|(values, mask, out)| {
            try_map_with_mask(values.as_slice(), mask, out.as_mut_slice(), |v, valid| {
                let scaled = v * valid as u64;
                (scaled <= u32::MAX as u64).then_some(scaled as u32)
            })
            .unwrap();
        });
}

/// Aligned-mask counterpart for `map_with_mask` (infallible). Pair with the
/// `_unaligned` variant below to isolate the mask-iteration cost from the closure.
#[divan::bench(args = SIZES)]
fn kernel_map_with_mask_unaligned(bencher: Bencher, n: usize) {
    let f = fixture(n);
    bencher
        .with_inputs(|| {
            let mut out: Vec<MaybeUninit<u32>> = Vec::with_capacity(n);
            unsafe { out.set_len(n) };
            (f.values.clone(), f.mask_unaligned.clone(), out)
        })
        .bench_refs(|(values, mask, out)| {
            map_with_mask(values.as_slice(), mask, out.as_mut_slice(), |v, valid| {
                (v * valid as u64) as u32
            });
        });
}

/// As above but with the branchful idiomatic form. Tests whether autovectorization
/// survives a per-lane `if valid { ... } else { ... }` shape.
#[divan::bench(args = SIZES)]
fn kernel_try_from_branchful(bencher: Bencher, n: usize) {
    let f = fixture(n);
    bencher
        .with_inputs(|| {
            let mut out: Vec<MaybeUninit<u32>> = Vec::with_capacity(n);
            unsafe { out.set_len(n) };
            (f.values.clone(), f.mask_aligned.clone(), out)
        })
        .bench_refs(|(values, mask, out)| {
            try_map_with_mask(values.as_slice(), mask, out.as_mut_slice(), |v, valid| {
                if valid {
                    u32::try_from(v).ok()
                } else {
                    Some(0_u32)
                }
            })
            .unwrap();
        });
}

#[divan::bench(args = SIZES)]
fn iter_zip_checked(bencher: Bencher, n: usize) {
    let f = fixture(n);
    bencher
        .with_inputs(|| (f.values.clone(), f.mask_aligned.clone()))
        .bench_refs(|(values, mask)| {
            let buf: Buffer<u32> = BufferMut::try_from_trusted_len_iter(
                values.iter().zip(mask.iter()).map(|(&v, valid)| {
                    let scaled = v * valid as u64;
                    if scaled <= u32::MAX as u64 {
                        Ok(scaled as u32)
                    } else {
                        Err(())
                    }
                }),
            )
            .unwrap()
            .freeze();
            buf
        });
}

#[divan::bench(args = SIZES)]
fn iter_zip_checked_unaligned(bencher: Bencher, n: usize) {
    let f = fixture(n);
    bencher
        .with_inputs(|| (f.values.clone(), f.mask_unaligned.clone()))
        .bench_refs(|(values, mask)| {
            let buf: Buffer<u32> = BufferMut::try_from_trusted_len_iter(
                values.iter().zip(mask.iter()).map(|(&v, valid)| {
                    let scaled = v * valid as u64;
                    if scaled <= u32::MAX as u64 {
                        Ok(scaled as u32)
                    } else {
                        Err(())
                    }
                }),
            )
            .unwrap()
            .freeze();
            buf
        });
}

#[divan::bench(args = SIZES)]
fn arrow_cast_checked(bencher: Bencher, n: usize) {
    let _ = n;
    let f = fixture(n);
    bencher
        .with_inputs(|| f.arrow_arr.clone())
        .bench_refs(|arr| cast_with_options(arr, &DataType::UInt32, &CAST_OPTS_CHECKED).unwrap());
}

#[divan::bench(args = SIZES)]
fn arrow_cast_checked_unaligned(bencher: Bencher, n: usize) {
    let _ = n;
    let f = fixture(n);
    bencher
        .with_inputs(|| f.arrow_arr_unaligned.clone())
        .bench_refs(|arr| cast_with_options(arr, &DataType::UInt32, &CAST_OPTS_CHECKED).unwrap());
}
