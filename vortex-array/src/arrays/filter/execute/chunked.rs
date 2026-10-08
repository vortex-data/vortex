// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Filtering of values that decode in fixed-size chunks.
//!
//! The mask is applied one chunk at a time, from the mask's cached slices when
//! [`filter_buffer`](super::buffer::filter_buffer) would use them and otherwise from its bitmap.
//! Each chunk takes the first matching strategy:
//!
//! | Selected values in the chunk       | Strategy                                          |
//! | ---------------------------------- | ------------------------------------------------- |
//! | none                               | skip the chunk                                    |
//! | all                                | decode the chunk straight to the output           |
//! | at most 16 per byte of value width | decode only the selected values                   |
//! | otherwise                          | decode to scratch, then compact the scratch       |
//!
//! The scratch buffer stays in cache. It is compacted by copying the slices, by copying the runs
//! of set bits when they average at least [`MIN_RUN_BYTES`], and otherwise with the same SIMD,
//! byte compress and scalar kernels that `filter_buffer` chooses for the chunk's density.

use std::mem::MaybeUninit;
use std::ptr;

use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_mask::MaskValues;

use super::buffer::byte_compress_density_threshold;
use super::buffer::useful_cached_slices;
use super::byte_compress;
use super::simd_compress;
use super::simd_compress::SLACK_BYTES;
use super::slice::MaskBits;
use super::slice::compact_by_bitmap;
use super::slice::compact_runs_by_bitmap;
use super::slice::low_bits_mask;
use crate::dtype::NativePType;

/// Number of values in each chunk of a [`ChunkDecoder`].
pub const FILTER_CHUNK_LEN: usize = 1024;

/// Number of 64-bit mask words that cover a chunk.
const CHUNK_WORDS: usize = FILTER_CHUNK_LEN / 64;

/// Chunks with at most this many selected values per byte of value width decode only those
/// values rather than the whole chunk. Wider values make a whole chunk more expensive to decode,
/// so the threshold grows with the width.
const SPARSE_VALUES_PER_BYTE: usize = 16;

/// The largest number of selected values for which a chunk decodes only those values.
const MAX_SPARSE_VALUES: usize = SPARSE_VALUES_PER_BYTE * 8;

/// Chunks whose runs of selected values average at least this many bytes copy each run. Shorter
/// runs are faster to compact with the SIMD and byte compress kernels.
const MIN_RUN_BYTES: usize = 96;

/// Number of mask bytes read to assemble the words of a chunk at an arbitrary bit offset.
const CHUNK_WORD_BYTES: usize = FILTER_CHUNK_LEN / 8 + 8;

/// Decodes the values of an array in chunks of [`FILTER_CHUNK_LEN`] values, for
/// [`filter_chunked`].
///
/// # Safety
///
/// Each method must initialize every value of its `dst`.
pub unsafe trait ChunkDecoder<T> {
    /// Decodes all values of chunk `chunk_idx` to `dst`.
    fn decode_chunk(&self, chunk_idx: usize, dst: &mut [MaybeUninit<T>; FILTER_CHUNK_LEN]);

    /// Decodes the values at the sorted in-chunk `indices` of chunk `chunk_idx` to `dst`, which
    /// holds one value for each index.
    fn decode_indices(&self, chunk_idx: usize, indices: &[usize], dst: &mut [MaybeUninit<T>]);
}

/// Filters the values of `decoder` by `mask`, and decodes only the chunks that hold selected
/// values.
///
/// Value `i` of the filtered array is value `offset + i` of the decoded chunks, so `offset` skips
/// the values at the start of the first chunk. See the [module docs](self) for the strategy that
/// each chunk takes.
pub fn filter_chunked<T: NativePType>(
    decoder: &impl ChunkDecoder<T>,
    offset: usize,
    mask: &MaskValues,
) -> Buffer<T> {
    let mut filter = ChunkedFilter::new(decoder, mask.true_count());
    match useful_cached_slices(mask) {
        Some(slices) => filter.push_slices(offset, slices),
        None => filter.push_bitmap(offset, mask),
    }
    filter.finish()
}

/// A chunk of decoded values, aligned to a cache line so that the decoder's vector stores never
/// cross one.
#[repr(align(64))]
struct Scratch<T>([MaybeUninit<T>; FILTER_CHUNK_LEN]);

/// The output and scratch buffers of [`filter_chunked`].
struct ChunkedFilter<'a, T, D> {
    decoder: &'a D,
    values: BufferMut<T>,
    true_count: usize,
    written: usize,
    max_sparse: usize,
    scratch: Scratch<T>,
    indices: [usize; MAX_SPARSE_VALUES],
}

impl<'a, T: NativePType, D: ChunkDecoder<T>> ChunkedFilter<'a, T, D> {
    fn new(decoder: &'a D, true_count: usize) -> Self {
        Self {
            decoder,
            // The SIMD kernels can store one vector past the last value that they write.
            values: BufferMut::with_capacity(true_count + SLACK_BYTES / size_of::<T>()),
            true_count,
            written: 0,
            max_sparse: (SPARSE_VALUES_PER_BYTE * size_of::<T>()).min(MAX_SPARSE_VALUES),
            scratch: Scratch([const { MaybeUninit::uninit() }; FILTER_CHUNK_LEN]),
            indices: [0; MAX_SPARSE_VALUES],
        }
    }

    /// Returns the output position of the next value, with room for every value not yet written
    /// plus [`SLACK_BYTES`].
    fn dst(&mut self) -> *mut MaybeUninit<T> {
        // SAFETY: the output has capacity for every selected value plus the slack.
        unsafe {
            self.values
                .spare_capacity_mut()
                .as_mut_ptr()
                .add(self.written)
        }
    }

    /// Decodes chunk `chunk_idx` straight to the output.
    fn push_full_chunk(&mut self, chunk_idx: usize) {
        // SAFETY: a fully selected chunk has `FILTER_CHUNK_LEN` values of output capacity.
        let dst = unsafe { &mut *self.dst().cast() };
        self.decoder.decode_chunk(chunk_idx, dst);
        self.written += FILTER_CHUNK_LEN;
    }

    /// Decodes the values at the first `len` of `self.indices` in chunk `chunk_idx` to the output.
    fn push_indices(&mut self, chunk_idx: usize, len: usize) {
        // SAFETY: the output has capacity for the `len` selected values.
        let dst = unsafe { std::slice::from_raw_parts_mut(self.dst(), len) };
        self.decoder
            .decode_indices(chunk_idx, &self.indices[..len], dst);
        self.written += len;
    }

    /// Decodes chunk `chunk_idx` to the scratch buffer and returns its values.
    fn decode_scratch(&mut self, chunk_idx: usize) -> &[T] {
        self.decoder.decode_chunk(chunk_idx, &mut self.scratch.0);
        // SAFETY: the decoder initialized every value of the scratch buffer.
        unsafe { std::slice::from_raw_parts(self.scratch.0.as_ptr().cast(), FILTER_CHUNK_LEN) }
    }

    /// Filters chunk by chunk from the bitmap of `mask`.
    fn push_bitmap(&mut self, offset: usize, mask: &MaskValues) {
        let len = mask.len();
        let bits = mask.bit_buffer();
        let bits_bytes = bits.inner().as_slice();
        let mut words = [0u64; CHUNK_WORDS];

        for chunk_idx in 0..(offset + len).div_ceil(FILTER_CHUNK_LEN) {
            // The range `start..end` of the array that falls within this chunk, which begins at
            // index `chunk_offset` within the chunk.
            let chunk_start = chunk_idx * FILTER_CHUNK_LEN;
            let start = chunk_start.saturating_sub(offset);
            let end = (chunk_start + FILTER_CHUNK_LEN - offset).min(len);
            let chunk_offset = start + offset - chunk_start;
            let chunk_len = end - start;

            load_chunk_words(bits_bytes, bits.offset() + start, chunk_len, &mut words);
            let words = &words[..chunk_len.div_ceil(64)];
            let selected = words.iter().map(|w| w.count_ones() as usize).sum::<usize>();

            if selected == 0 {
                continue;
            }

            if selected == FILTER_CHUNK_LEN {
                self.push_full_chunk(chunk_idx);
            } else if selected <= self.max_sparse {
                let mut n = 0;
                for (word_idx, &word) in words.iter().enumerate() {
                    let mut word = word;
                    while word != 0 {
                        self.indices[n] =
                            chunk_offset + word_idx * 64 + word.trailing_zeros() as usize;
                        n += 1;
                        word &= word - 1;
                    }
                }
                self.push_indices(chunk_idx, selected);
            } else {
                let dst = self.dst().cast::<T>();
                let src = &self.decode_scratch(chunk_idx)[chunk_offset..][..chunk_len];
                // SAFETY: the words select only values of `src` and clear any bits past
                // `chunk_len`, and the output has room for every selected value plus the slack.
                let written = unsafe { compact_words(src, words, selected, dst) };
                debug_assert_eq!(written, selected);
                self.written += selected;
            }
        }
    }

    /// Filters chunk by chunk from the sorted, disjoint `slices`.
    fn push_slices(&mut self, offset: usize, slices: &[(usize, usize)]) {
        // `slices[next..]` holds the slices not yet fully written, and `resume` is the start of
        // the first chunk not yet processed, from which `slices[next]` resumes if it began in an
        // earlier chunk. Positions include `offset`.
        let mut next = 0;
        let mut resume = 0;
        while let Some(&(first_start, _)) = slices.get(next) {
            let chunk_idx = (first_start + offset).max(resume) / FILTER_CHUNK_LEN;
            let chunk_start = chunk_idx * FILTER_CHUNK_LEN;
            let chunk_end = chunk_start + FILTER_CHUNK_LEN;

            // The in-chunk ranges selected within this chunk.
            let runs = slices[next..]
                .iter()
                .map(|&(start, end)| (start + offset, end + offset))
                .take_while(|&(start, _)| start < chunk_end)
                .map(|(start, end)| {
                    (
                        start.max(chunk_start) - chunk_start,
                        end.min(chunk_end) - chunk_start,
                    )
                });
            let selected = runs.clone().map(|(start, end)| end - start).sum::<usize>();

            if selected == FILTER_CHUNK_LEN {
                self.push_full_chunk(chunk_idx);
            } else if selected <= self.max_sparse {
                for (index, chunk_index) in self
                    .indices
                    .iter_mut()
                    .zip(runs.flat_map(|(start, end)| start..end))
                {
                    *index = chunk_index;
                }
                self.push_indices(chunk_idx, selected);
            } else {
                let dst = self.dst().cast::<T>();
                let src = self.decode_scratch(chunk_idx).as_ptr();
                let mut written = 0;
                for (start, end) in runs {
                    // SAFETY: each run lies within the chunk, and the output has room for every
                    // selected value.
                    unsafe {
                        ptr::copy_nonoverlapping(src.add(start), dst.add(written), end - start)
                    };
                    written += end - start;
                }
                self.written += written;
            }

            // Skip the slices that end within this chunk; the last one may continue into the
            // next.
            next += slices[next..]
                .iter()
                .take_while(|&&(_, end)| end + offset <= chunk_end)
                .count();
            resume = chunk_end;
        }
    }

    fn finish(mut self) -> Buffer<T> {
        debug_assert_eq!(self.written, self.true_count);
        // SAFETY: the first `true_count` output values were initialized above.
        unsafe { self.values.set_len(self.true_count) };
        self.values.freeze()
    }
}

/// Copies the `selected` values of `src` selected by `words` to `dst` and returns the number
/// copied.
///
/// Runs of set bits that average at least [`MIN_RUN_BYTES`] are copied run by run. Otherwise the kernel is the one that
/// [`filter_buffer`](super::buffer::filter_buffer) chooses for the density.
///
/// # Safety
///
/// `words` must hold `src.len()` bits with any bits past `src.len()` cleared, and `dst` must be
/// valid for writes of every selected value plus [`SLACK_BYTES`].
unsafe fn compact_words<T: Copy>(src: &[T], words: &[u64], selected: usize, dst: *mut T) -> usize {
    let bits = MaskBits::Words {
        words,
        len: src.len(),
    };
    // A run that crosses a word boundary counts once in each word, as the run walk copies it.
    let runs = words
        .iter()
        .map(|&word| (word & !(word << 1)).count_ones() as usize)
        .sum::<usize>();
    let density = selected as f64 / src.len() as f64;

    // SAFETY: forwarded from the caller contract.
    unsafe {
        if selected * size_of::<T>() >= MIN_RUN_BYTES * runs {
            compact_runs_by_bitmap(src, bits, dst)
        } else if let Some(written) = simd_compress::compress_bits(src, bits, density, dst) {
            written
        } else if density >= byte_compress_density_threshold::<T>() {
            byte_compress::compress_words(src, words, dst)
        } else {
            compact_by_bitmap(src, bits, dst)
        }
    }
}

/// Loads the `len <= FILTER_CHUNK_LEN` mask bits starting at bit `start` of `bytes` into
/// `words`, with any bits past `len` cleared.
#[inline]
fn load_chunk_words(bytes: &[u8], start: usize, len: usize, words: &mut [u64; CHUNK_WORDS]) {
    debug_assert!(len <= FILTER_CHUNK_LEN);
    let byte_start = start / 8;
    let shift = start % 8;

    if len == FILTER_CHUNK_LEN {
        if shift == 0
            && let Some(src) = bytes[byte_start..].first_chunk::<{ FILTER_CHUNK_LEN / 8 }>()
        {
            for (word, src) in words.iter_mut().zip(src.as_chunks::<8>().0) {
                *word = u64::from_le_bytes(*src);
            }
            return;
        }
        if let Some(src) = bytes[byte_start..].first_chunk::<CHUNK_WORD_BYTES>() {
            assemble_words(src, shift, words);
            return;
        }
    }

    // Near the end of the bitmap, copy into a zero-padded buffer so the words can still be
    // assembled from fixed-size reads.
    let byte_len = (shift + len).div_ceil(8);
    let mut buf = [0u8; CHUNK_WORD_BYTES];
    buf[..byte_len].copy_from_slice(&bytes[byte_start..][..byte_len]);
    assemble_words(&buf, shift, words);
    for (i, word) in words.iter_mut().enumerate() {
        *word &= low_bits_mask(len.saturating_sub(i * 64).min(64));
    }
}

/// Assembles 64-bit words from `src`, starting `shift < 8` bits into its first byte.
#[inline]
fn assemble_words(src: &[u8; CHUNK_WORD_BYTES], shift: usize, words: &mut [u64; CHUNK_WORDS]) {
    for (i, word) in words.iter_mut().enumerate() {
        let (lo, rest) = src[i * 8..]
            .split_first_chunk::<8>()
            .unwrap_or_else(|| unreachable!());
        let lo = u64::from_le_bytes(*lo);
        let hi = u64::from(rest[0]);
        // Shifting `hi` in two steps keeps the shift amount below 64 when `shift == 0`.
        *word = (lo >> shift) | ((hi << 1) << (63 - shift));
    }
}

#[cfg(test)]
mod tests {
    use std::mem::MaybeUninit;

    use rand::RngExt;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rstest::rstest;
    use vortex_buffer::BitBuffer;
    use vortex_mask::Mask;

    use super::ChunkDecoder;
    use super::FILTER_CHUNK_LEN;
    use super::filter_chunked;
    use crate::dtype::NativePType;

    /// Decodes chunks of a plain slice, which holds whole chunks.
    struct SliceDecoder<'a, T>(&'a [T]);

    // SAFETY: both methods initialize every value of `dst`.
    unsafe impl<T: Copy> ChunkDecoder<T> for SliceDecoder<'_, T> {
        fn decode_chunk(&self, chunk_idx: usize, dst: &mut [MaybeUninit<T>; FILTER_CHUNK_LEN]) {
            dst.write_copy_of_slice(&self.0[chunk_idx * FILTER_CHUNK_LEN..][..FILTER_CHUNK_LEN]);
        }

        fn decode_indices(&self, chunk_idx: usize, indices: &[usize], dst: &mut [MaybeUninit<T>]) {
            let chunk = &self.0[chunk_idx * FILTER_CHUNK_LEN..][..FILTER_CHUNK_LEN];
            for (dst, &index) in dst.iter_mut().zip(indices) {
                dst.write(chunk[index]);
            }
        }
    }

    fn check<T: NativePType + From<u8>>(bits: &BitBuffer, offset: usize, slices: bool) {
        let padded_len = (offset + bits.len()).next_multiple_of(FILTER_CHUNK_LEN);
        let values = (0..padded_len)
            .map(|i| {
                <T as From<u8>>::from(u8::try_from(i % 251).unwrap_or_else(|_| unreachable!()))
            })
            .collect::<Vec<T>>();
        let expected = bits
            .iter()
            .enumerate()
            .filter_map(|(i, selected)| selected.then_some(values[offset + i]))
            .collect::<Vec<T>>();

        let mask = if slices {
            Mask::from_slices(bits.len(), bits.set_slices().collect())
        } else {
            Mask::from_buffer(bits.clone())
        };
        let Mask::Values(mask) = mask else {
            unreachable!("the patterns select some but not all values")
        };
        let filtered = filter_chunked(&SliceDecoder(&values), offset, &mask);
        assert_eq!(filtered.as_slice(), expected);
    }

    #[rstest]
    fn filter_chunked_matches_expected(
        #[values(0.001, 0.01, 0.05, 0.2, 0.5, 0.9, 0.999)] density: f64,
        #[values(0, 3, 1000)] offset: usize,
        #[values(0, 5)] bit_offset: usize,
    ) {
        let mut rng = StdRng::seed_from_u64(42);
        let bits = (0..5000 + bit_offset)
            .map(|_| rng.random_bool(density))
            .collect::<BitBuffer>()
            .slice(bit_offset..5000 + bit_offset);

        check::<u8>(&bits, offset, false);
        check::<u16>(&bits, offset, false);
        check::<u32>(&bits, offset, false);
        check::<u64>(&bits, offset, false);
    }

    /// Runs long enough that the cached slices are used rather than the bitmap.
    #[rstest]
    fn filter_chunked_runs_matches_expected(
        #[values(9, 12, 200, 3000)] run_len: usize,
        #[values(1, 16, 64, 1500)] gap: usize,
        #[values(0, 3, 1000)] offset: usize,
        #[values(false, true)] slices: bool,
    ) {
        let bits = (0..5000)
            .map(|i| i % (run_len + gap) < run_len)
            .collect::<BitBuffer>();

        check::<u8>(&bits, offset, slices);
        check::<u16>(&bits, offset, slices);
        check::<u32>(&bits, offset, slices);
        check::<u64>(&bits, offset, slices);
    }

    #[rstest]
    fn filter_chunked_full_and_empty_chunks(#[values(0, 3)] offset: usize) {
        // Whole chunks of selected values, alternating with whole chunks of unselected values.
        let bits = (0..5000)
            .map(|i| ((i + offset) / FILTER_CHUNK_LEN).is_multiple_of(2))
            .collect::<BitBuffer>();

        for slices in [false, true] {
            check::<u8>(&bits, offset, slices);
            check::<u64>(&bits, offset, slices);
        }
    }
}
