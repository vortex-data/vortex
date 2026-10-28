// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Filtering of fixed-width values by a [`Selection`].
//!
//! [`filter_buffer`] filters values that are already in memory, and [`filter_chunked`] filters
//! values that a [`ChunkDecoder`] decodes one chunk at a time. Both work for any `T: Copy` and
//! share one strategy ladder. [`filter_records`] filters records whose byte width is known only
//! at run time, and uses [`filter_buffer`] for power-of-two widths. The first matching row wins:
//!
//! | Priority | Selection or condition                              | Out-of-place   | In-place      |
//! | -------- | --------------------------------------------------- | -------------- | ------------- |
//! | 1        | slices                                              | copy runs      | move runs     |
//! | 2        | indices                                             | gather indices | move indices  |
//! | 3        | bitmap, an eligible architecture kernel             | SIMD compress  | SIMD compress |
//! | 4        | bitmap, density reaches the byte compress threshold | byte compress  | bitmap walk   |
//! | 5        | bitmap, runs average at least 8 values and 96 bytes | copy runs      | bitmap walk   |
//! | 6        | bitmap, otherwise                                   | bitmap walk    | bitmap walk   |
//!
//! [`filter_buffer`] compacts in place only a uniquely owned buffer with density >= 0.50.
//! [`filter_chunked`] applies the ladder to each chunk that it decodes, at the density of that
//! chunk. SIMD eligibility is summarized in the `simd` module.

use std::mem::MaybeUninit;

use crate::BitBuffer;
use crate::Buffer;
use crate::BufferAllocatorRef;
use crate::BufferMut;

mod bits;
mod byte_compress;
mod chunked;
mod records;
mod scalar;
mod simd;
#[cfg(test)]
mod tests;

use bits::MaskBits;
pub use chunked::ChunkDecoder;
pub use chunked::FILTER_CHUNK_LEN;
pub use chunked::filter_chunked;
pub use records::filter_records;

const IN_PLACE_MIN_DENSITY: f64 = 0.5;

/// Bitmaps without a SIMD kernel whose runs of selected values average at least this many bytes,
/// and at least [`MIN_RUN_LEN`] values, copy each run.
const MIN_RUN_BYTES: usize = 96;

/// The smallest average run length, in values, for which a bitmap copies each run. Random masks
/// of wide values reach [`MIN_RUN_BYTES`] with runs of only a few values, and the bitmap walk is
/// faster for those.
const MIN_RUN_LEN: usize = 8;

/// The positions that a filter keeps, out of `len` positions.
///
/// A selection holds one representation of the positions. The caller chooses the
/// representation, because the cost to make one from another is higher than the cost to filter.
#[derive(Clone, Copy)]
pub struct Selection<'a> {
    len: usize,
    true_count: usize,
    repr: Repr<'a>,
}

#[derive(Clone, Copy)]
enum Repr<'a> {
    Bitmap(&'a BitBuffer),
    Indices(&'a [usize]),
    Slices(&'a [(usize, usize)]),
}

impl<'a> Selection<'a> {
    /// Selects the set bits of `bits`.
    ///
    /// # Safety
    ///
    /// `true_count` must equal the number of set bits in `bits`.
    pub unsafe fn from_bitmap_unchecked(bits: &'a BitBuffer, true_count: usize) -> Self {
        debug_assert_eq!(bits.true_count(), true_count);

        Self {
            len: bits.len(),
            true_count,
            repr: Repr::Bitmap(bits),
        }
    }

    /// Selects the positions in `indices`, out of `len` positions.
    ///
    /// # Safety
    ///
    /// `indices` must be strictly increasing, and each index must be less than `len`.
    pub unsafe fn from_indices_unchecked(len: usize, indices: &'a [usize]) -> Self {
        debug_assert!(indices.is_sorted_by(|a, b| a < b));
        debug_assert!(indices.last().is_none_or(|&last| last < len));

        Self {
            len,
            true_count: indices.len(),
            repr: Repr::Indices(indices),
        }
    }

    /// Selects the positions in the `(start, end)` ranges of `slices`, out of `len` positions.
    ///
    /// # Safety
    ///
    /// The ranges must be sorted and disjoint, each `start` must not be more than its `end`, the
    /// last `end` must not be more than `len`, and `true_count` must equal the total length of
    /// the ranges.
    pub unsafe fn from_slices_unchecked(
        len: usize,
        slices: &'a [(usize, usize)],
        true_count: usize,
    ) -> Self {
        debug_assert!(slices.iter().all(|&(start, end)| start <= end));
        debug_assert!(slices.is_sorted_by(|a, b| a.1 <= b.0));
        debug_assert!(slices.last().is_none_or(|&(_, end)| end <= len));
        debug_assert_eq!(
            slices.iter().map(|(start, end)| end - start).sum::<usize>(),
            true_count
        );

        Self {
            len,
            true_count,
            repr: Repr::Slices(slices),
        }
    }

    /// Returns the number of positions that the selection keeps or drops.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns whether the selection has no positions.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the number of positions that the selection keeps.
    pub fn true_count(&self) -> usize {
        self.true_count
    }

    fn density(&self) -> f64 {
        if self.len == 0 {
            return 0.0;
        }

        self.true_count as f64 / self.len as f64
    }
}

/// Filters `buffer` by `selection` and returns the selected values.
///
/// A uniquely owned buffer with density of at least 0.50 is compacted in place. Other buffers
/// are copied to a new allocation from `allocator`.
///
/// # Panics
///
/// Panics if `buffer` and `selection` have different lengths.
pub fn filter_buffer<T: Copy>(
    buffer: Buffer<T>,
    selection: Selection<'_>,
    allocator: &BufferAllocatorRef,
) -> Buffer<T> {
    assert_eq!(
        buffer.len(),
        selection.len(),
        "Selection length must equal the buffer length"
    );

    let buffer = if selection.density() >= IN_PLACE_MIN_DENSITY {
        match buffer.try_into_mut() {
            Ok(mut buffer_mut) => {
                let new_len = filter_slice_in_place(buffer_mut.as_mut_slice(), selection);
                buffer_mut.truncate(new_len);
                return buffer_mut.freeze();
            }
            Err(buffer) => buffer,
        }
    } else {
        buffer
    };

    filter_slice(buffer.as_slice(), selection, allocator)
}

/// Copies the values of `values` that `selection` keeps to a new buffer.
fn filter_slice<T: Copy>(
    values: &[T],
    selection: Selection<'_>,
    allocator: &BufferAllocatorRef,
) -> Buffer<T> {
    debug_assert_eq!(values.len(), selection.len());

    let true_count = selection.true_count();
    // The SIMD kernels can store one vector past the last value that they write.
    let mut out = BufferMut::<T>::with_capacity_in(
        true_count + simd::SLACK_BYTES / size_of::<T>(),
        allocator.clone(),
    );
    let dst = out.spare_capacity_mut();

    let written = match selection.repr {
        Repr::Slices(slices) => scalar::copy_slices(values, slices, dst),
        Repr::Indices(indices) => {
            // SAFETY: the selection guarantees that each index is less than `values.len()`.
            unsafe { scalar::gather_indices(values, indices, dst) };
            indices.len()
        }
        // SAFETY: the bitmap has one bit for each value, and the output has room for every
        // selected value plus the slack.
        Repr::Bitmap(bits) => unsafe {
            compact_bits(values, MaskBits::Bitmap(bits), true_count, dst)
        },
    };

    debug_assert_eq!(written, true_count);
    // SAFETY: the strategy above initialized the first `true_count` values.
    unsafe { out.set_len(true_count) };
    out.freeze()
}

/// Compacts the values of `values` that `selection` keeps to its front, and returns the new
/// length.
fn filter_slice_in_place<T: Copy>(values: &mut [T], selection: Selection<'_>) -> usize {
    debug_assert_eq!(values.len(), selection.len());

    let written = match selection.repr {
        Repr::Slices(slices) => scalar::compact_in_place_by_slices(values, slices),
        // SAFETY: the selection guarantees that each index is less than `values.len()`.
        Repr::Indices(indices) => unsafe { scalar::compact_in_place_by_indices(values, indices) },
        Repr::Bitmap(bits) => {
            let bits = MaskBits::Bitmap(bits);
            // SAFETY: the bitmap has one bit for each value.
            unsafe { simd::compress_in_place(values, bits, selection.density()) }
                .unwrap_or_else(|| scalar::compact_in_place_by_bitmap(values, bits))
        }
    };

    debug_assert_eq!(written, selection.true_count());
    written
}

/// Copies the `selected` values of `src` that `bits` selects to `dst` and returns the number
/// copied.
///
/// This is rows 3 to 6 of the [module](self) ladder: the SIMD kernel for the density, if any.
/// Otherwise, a density at or above the byte compress threshold uses byte compress, long runs
/// of set bits are copied run by run, and shorter runs use the bitmap walk.
///
/// # Safety
///
/// `bits` must select only values of `src`, and `dst` must hold every selected value plus
/// [`SLACK_BYTES`](simd::SLACK_BYTES).
unsafe fn compact_bits<T: Copy>(
    src: &[T],
    bits: MaskBits<'_>,
    selected: usize,
    dst: &mut [MaybeUninit<T>],
) -> usize {
    let density = if src.is_empty() {
        0.0
    } else {
        selected as f64 / src.len() as f64
    };

    // SAFETY: forwarded from the caller contract.
    unsafe {
        if let Some(written) = simd::compress(src, bits, density, dst) {
            written
        } else if density >= byte_compress_density_threshold::<T>() {
            byte_compress::compress(src, bits, dst)
        } else if has_long_runs::<T>(bits, selected) {
            scalar::compact_runs_by_bitmap(src, bits, dst)
        } else {
            scalar::compact_by_bitmap(src, bits, dst)
        }
    }
}

/// Returns whether the runs of the `selected` set bits of `bits` average at least
/// [`MIN_RUN_LEN`] values and [`MIN_RUN_BYTES`] of `T`.
// Inlining this check into `compact_bits` makes the bit walk up to 1.3x slower on x86-64.
#[inline(never)]
fn has_long_runs<T>(bits: MaskBits<'_>, selected: usize) -> bool {
    // The run walk copies at most one word of values at a time, so narrow values never qualify.
    if 64 * size_of::<T>() < MIN_RUN_BYTES {
        return false;
    }

    // A run that crosses a word boundary counts once in each word, as the run walk copies it. The
    // count stops at the first word that takes it past `max_runs`, because a popcount is not a
    // native instruction on baseline x86-64.
    let max_runs = (selected * size_of::<T>() / MIN_RUN_BYTES).min(selected / MIN_RUN_LEN);
    let mut runs = 0;
    bits::all_mask_words(bits, |word| {
        runs += (word & !(word << 1)).count_ones() as usize;
        runs <= max_runs
    })
}

/// Returns the density at or above which byte compress is used after the higher-priority
/// strategies have declined the bitmap.
fn byte_compress_density_threshold<T>() -> f64 {
    let width = size_of::<T>();

    // These crossovers are benchmarked in `vortex-array/benches/filter_fixed_width.rs`.
    //
    // | Target  | 1 byte | 2 bytes | 4 bytes | 8 bytes | other |
    // | ------- | -----: | ------: | ------: | ------: | ----: |
    // | aarch64 |   0.90 |    0.90 |    0.90 |    0.75 |  0.90 |
    // | other   |   0.00 |    0.50 |    0.50 |    0.75 | 0.875 |
    if cfg!(target_arch = "aarch64") {
        return match width {
            8 => 0.75,
            _ => 0.9,
        };
    }

    match width {
        1 => 0.0,
        2 | 4 => 0.5,
        8 => 0.75,
        _ => 0.875,
    }
}
