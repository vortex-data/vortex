// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Mirror of `cast_to.rs` driving the kernels through [`vortex_buffer::lane_ops_indexed`]
//! (the `IndexedSource` trait) plus isolation benches that decompose the cost of the
//! kernel structure vs. the cast vs. the mask access.
//!
//! See `vortex-buffer/HISTORY.md` for the iterator-API investigation that motivated
//! this design: a stateful `ExactSizeIterator` variant of these kernels was ~+100%
//! slower because per-lane `next()` calls create a 64-deep dependency chain across
//! iterations that blocks vectorization. The `IndexedSource` trait uses
//! `unsafe fn get_unchecked(i)` reads — independent across iterations — and inlines
//! to the same indexed load as the slice kernel.

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
use vortex_buffer::lane_ops_indexed::map_with_mask as indexed_map_with_mask;
use vortex_buffer::lane_ops_indexed::try_map_validity_filtered as indexed_try_map_validity_filtered;
use vortex_buffer::lane_ops_indexed::try_map_with_mask as indexed_try_map_with_mask;

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

    let mask_unaligned = {
        let mut m = BitBufferMut::with_capacity(n + SLICE_OFFSET);
        for _ in 0..SLICE_OFFSET {
            m.append(false);
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

// -----------------------------------------------------------------------------
// Isolation benches: drop the mask, isolate the cast u64 -> u32 to see whether
// the iterator cost is intrinsic or comes from the surrounding kernel structure.
// -----------------------------------------------------------------------------

/// Plain slice indexing, no mask. Upper bound on what the iter variants must beat.
#[divan::bench(args = SIZES)]
fn iso_slice_cast(bencher: Bencher, n: usize) {
    let f = fixture(n);
    bencher
        .with_inputs(|| {
            let mut out: Vec<MaybeUninit<u32>> = Vec::with_capacity(n);
            unsafe { out.set_len(n) };
            (f.values.clone(), out)
        })
        .bench_refs(|(values, out)| {
            let v = values.as_slice();
            let o = out.as_mut_slice();
            assert_eq!(v.len(), o.len());
            for i in 0..v.len() {
                // SAFETY: bounds checked by the assert above.
                unsafe { o.get_unchecked_mut(i).write(*v.get_unchecked(i) as u32) };
            }
        });
}

/// Per-lane iterator zip, no mask. Tests whether `slice::Iter::next` autovectorizes
/// when nothing else is in the way.
#[divan::bench(args = SIZES)]
fn iso_iter_cast(bencher: Bencher, n: usize) {
    let f = fixture(n);
    bencher
        .with_inputs(|| {
            let mut out: Vec<MaybeUninit<u32>> = Vec::with_capacity(n);
            unsafe { out.set_len(n) };
            (f.values.clone(), out)
        })
        .bench_refs(|(values, out)| {
            for (slot, &v) in out.iter_mut().zip(values.iter()) {
                slot.write(v as u32);
            }
        });
}

/// `chunks_exact(64)` + `try_into::<&[u64; 64]>` so the outer iter advances once per
/// 64 lanes and the inner loop indexes a fixed-size array. Tests whether moving the
/// iterator state from per-lane to per-chunk fixes vectorization.
#[divan::bench(args = SIZES)]
fn iso_iter_chunks_64(bencher: Bencher, n: usize) {
    let f = fixture(n);
    bencher
        .with_inputs(|| {
            let mut out: Vec<MaybeUninit<u32>> = Vec::with_capacity(n);
            unsafe { out.set_len(n) };
            (f.values.clone(), out)
        })
        .bench_refs(|(values, out)| {
            let v = values.as_slice();
            let o = out.as_mut_slice();
            assert_eq!(v.len(), o.len());
            for (v_chunk, o_chunk) in v.chunks_exact(64).zip(o.chunks_exact_mut(64)) {
                let v_arr: &[u64; 64] = v_chunk.try_into().unwrap();
                let o_arr: &mut [MaybeUninit<u32>; 64] = o_chunk.try_into().unwrap();
                for bit_idx in 0..64 {
                    o_arr[bit_idx].write(v_arr[bit_idx] as u32);
                }
            }
            // Ignore the tail — SIZES are all multiples of 64.
        });
}

// -----------------------------------------------------------------------------
// Indexed-source variant (lane_ops_indexed). The kernel takes an `IndexedSource` whose
// `&[T]` impl is `unsafe fn get_unchecked(i) -> T` — same indexed load as the slice
// kernel, but the trait also supports binary inputs via `LaneZip`.
// -----------------------------------------------------------------------------

#[divan::bench(args = SIZES)]
fn indexed_kernel_map_with_mask(bencher: Bencher, n: usize) {
    let f = fixture(n);
    bencher
        .with_inputs(|| {
            let mut out: Vec<MaybeUninit<u32>> = Vec::with_capacity(n);
            unsafe { out.set_len(n) };
            (f.values.clone(), f.mask_aligned.clone(), out)
        })
        .bench_refs(|(values, mask, out)| {
            indexed_map_with_mask(values.as_slice(), mask, out.as_mut_slice(), |v, valid| {
                (v * valid as u64) as u32
            });
        });
}

#[divan::bench(args = SIZES)]
fn indexed_kernel_map_with_mask_unaligned(bencher: Bencher, n: usize) {
    let f = fixture(n);
    bencher
        .with_inputs(|| {
            let mut out: Vec<MaybeUninit<u32>> = Vec::with_capacity(n);
            unsafe { out.set_len(n) };
            (f.values.clone(), f.mask_unaligned.clone(), out)
        })
        .bench_refs(|(values, mask, out)| {
            indexed_map_with_mask(values.as_slice(), mask, out.as_mut_slice(), |v, valid| {
                (v * valid as u64) as u32
            });
        });
}

#[divan::bench(args = SIZES)]
fn indexed_kernel_try_map_with_mask(bencher: Bencher, n: usize) {
    let f = fixture(n);
    bencher
        .with_inputs(|| {
            let mut out: Vec<MaybeUninit<u32>> = Vec::with_capacity(n);
            unsafe { out.set_len(n) };
            (f.values.clone(), f.mask_aligned.clone(), out)
        })
        .bench_refs(|(values, mask, out)| {
            indexed_try_map_with_mask(values.as_slice(), mask, out.as_mut_slice(), |v, valid| {
                let scaled = v * valid as u64;
                (scaled <= u32::MAX as u64).then_some(scaled as u32)
            })
            .unwrap();
        });
}

#[divan::bench(args = SIZES)]
fn indexed_kernel_try_map_with_mask_unaligned(bencher: Bencher, n: usize) {
    let f = fixture(n);
    bencher
        .with_inputs(|| {
            let mut out: Vec<MaybeUninit<u32>> = Vec::with_capacity(n);
            unsafe { out.set_len(n) };
            (f.values.clone(), f.mask_unaligned.clone(), out)
        })
        .bench_refs(|(values, mask, out)| {
            indexed_try_map_with_mask(values.as_slice(), mask, out.as_mut_slice(), |v, valid| {
                let scaled = v * valid as u64;
                (scaled <= u32::MAX as u64).then_some(scaled as u32)
            })
            .unwrap();
        });
}

#[divan::bench(args = SIZES)]
fn indexed_kernel_try_from_branchful(bencher: Bencher, n: usize) {
    let f = fixture(n);
    bencher
        .with_inputs(|| {
            let mut out: Vec<MaybeUninit<u32>> = Vec::with_capacity(n);
            unsafe { out.set_len(n) };
            (f.values.clone(), f.mask_aligned.clone(), out)
        })
        .bench_refs(|(values, mask, out)| {
            indexed_try_map_with_mask(values.as_slice(), mask, out.as_mut_slice(), |v, valid| {
                if valid {
                    u32::try_from(v).ok()
                } else {
                    Some(0_u32)
                }
            })
            .unwrap();
        });
}

// -----------------------------------------------------------------------------
// Decoupled-design variant with CORRECT validity semantics: closure is `|v|`
// (no per-lane mask threading), but the mask filters out null-lane failures at
// the chunk boundary. A null row whose stored value would overflow does NOT
// cause Err — this matches the existing `try_map_with_mask` semantics while
// keeping the lighter inner loop.
// -----------------------------------------------------------------------------

#[divan::bench(args = SIZES)]
fn indexed_decoupled_kernel_try_map_with_mask(bencher: Bencher, n: usize) {
    let f = fixture(n);
    bencher
        .with_inputs(|| {
            let mut out: Vec<MaybeUninit<u32>> = Vec::with_capacity(n);
            // SAFETY: every lane is written before any read inside the kernel.
            unsafe { out.set_len(n) };
            (f.values.clone(), f.mask_aligned.clone(), out)
        })
        .bench_refs(|(values, mask, out)| {
            indexed_try_map_validity_filtered(values.as_slice(), mask, out.as_mut_slice(), |v| {
                (v <= u32::MAX as u64).then_some(v as u32)
            })
            .unwrap();
        });
}

#[divan::bench(args = SIZES)]
fn indexed_decoupled_kernel_try_from_branchful(bencher: Bencher, n: usize) {
    let f = fixture(n);
    bencher
        .with_inputs(|| {
            let mut out: Vec<MaybeUninit<u32>> = Vec::with_capacity(n);
            unsafe { out.set_len(n) };
            (f.values.clone(), f.mask_aligned.clone(), out)
        })
        .bench_refs(|(values, mask, out)| {
            indexed_try_map_validity_filtered(values.as_slice(), mask, out.as_mut_slice(), |v| {
                u32::try_from(v).ok()
            })
            .unwrap();
        });
}

/// Full checked-cast kernel using `chunks_exact(64)` + fixed-size array refs, with
/// the mask. If this matches the slice kernel, the cost is in the per-lane iterator
/// state, not the iter pattern in general.
#[divan::bench(args = SIZES)]
fn kernel_iter_chunks_64(bencher: Bencher, n: usize) {
    let f = fixture(n);
    bencher
        .with_inputs(|| {
            let mut out: Vec<MaybeUninit<u32>> = Vec::with_capacity(n);
            unsafe { out.set_len(n) };
            (f.values.clone(), f.mask_aligned.clone(), out)
        })
        .bench_refs(|(values, mask, out)| {
            let v = values.as_slice();
            let o = out.as_mut_slice();
            let len = v.len();
            assert_eq!(len, mask.len());
            assert_eq!(len, o.len());

            let chunks = mask.chunks();
            let chunks_count = len / 64;
            let full = chunks_count * 64;
            let (v_full, _v_rem) = v.split_at(full);
            let (o_full, _o_rem) = o.split_at_mut(full);

            for ((v_chunk, o_chunk), src_chunk) in v_full
                .chunks_exact(64)
                .zip(o_full.chunks_exact_mut(64))
                .zip(chunks.iter())
            {
                let v_arr: &[u64; 64] = v_chunk.try_into().unwrap();
                let o_arr: &mut [MaybeUninit<u32>; 64] = o_chunk.try_into().unwrap();
                let mut fail_acc: u64 = 0;
                for bit_idx in 0..64 {
                    let bit = (src_chunk >> bit_idx) & 1 == 1;
                    let scaled = v_arr[bit_idx] * bit as u64;
                    let opt = (scaled <= u32::MAX as u64).then_some(scaled as u32);
                    fail_acc |= opt.is_none() as u64;
                    o_arr[bit_idx].write(opt.unwrap_or_default());
                }
                assert_eq!(fail_acc, 0);
            }
            // Ignore the tail — SIZES are all multiples of 64.
        });
}
