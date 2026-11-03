// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Filtering of values that decode in fixed-size chunks. See [`filter_chunked`].

use std::mem::MaybeUninit;

use super::Repr;
use super::Selection;
use super::bits::MaskBits;
use super::bits::low_bits_mask;
use super::compact_bits;
use super::simd::SLACK_BYTES;
use crate::BitBuffer;
use crate::Buffer;
use crate::BufferAllocatorRef;
use crate::BufferMut;

/// Number of values in each chunk of a [`ChunkDecoder`].
pub const FILTER_CHUNK_LEN: usize = 1024;

/// Number of 64-bit mask words that cover a chunk.
const CHUNK_WORDS: usize = FILTER_CHUNK_LEN / 64;

/// The largest number of selected values for which a chunk decodes only those values.
const MAX_SPARSE_VALUES: usize = 160;

/// Chunks with at most this many selected values decode only those values rather than the whole
/// chunk. Wider values make a whole chunk more expensive to decode, so the threshold grows with
/// the width. Benchmarks of FastLanes and Vortex set these conservative crossovers.
const fn max_sparse_values<T>() -> usize {
    match size_of::<T>() {
        1 => 16,
        2 => 32,
        4 => 64,
        _ => MAX_SPARSE_VALUES,
    }
}

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

/// Filters the values of `decoder` by `selection`, and decodes only the chunks that hold
/// selected values.
///
/// Value `i` of the filtered array is value `offset + i` of the decoded chunks, so `offset` skips
/// the values at the start of the first chunk. The decoded chunks must hold at least
/// `offset + selection.len()` values.
///
/// The selection is applied one chunk at a time, in the representation that `selection` holds.
/// Each chunk takes the first strategy that matches:
///
/// | Selected values in the chunk | Strategy                                    |
/// | ---------------------------- | ------------------------------------------- |
/// | none                         | skip the chunk                              |
/// | all                          | decode the chunk straight to the output     |
/// | at most the sparse limit     | decode only the selected values             |
/// | otherwise                    | decode to scratch, then compact the scratch |
///
/// The sparse limit is 16, 32, 64 or 160 for values of 1, 2, 4 or 8 or more bytes, because a
/// wider chunk costs more to decode.
///
/// The scratch buffer stays in cache. The filter compacts it by copying the slices or gathering
/// the indices, or from a bitmap with the same strategy ladder as [`filter_buffer`].
///
/// [`filter_buffer`]: super::filter_buffer
pub fn filter_chunked<T: Copy>(
    decoder: &impl ChunkDecoder<T>,
    offset: usize,
    selection: Selection<'_>,
    allocator: &BufferAllocatorRef,
) -> Buffer<T> {
    let mut filter = ChunkedFilter::new(decoder, selection.true_count, allocator);

    match selection.repr {
        Repr::Bitmap(bits) => filter.push_bitmap(offset, bits),
        Repr::Indices(indices) => filter.push_mask_indices(offset, indices),
        Repr::Slices(slices) => filter.push_slices(offset, slices),
    }

    filter.finish()
}

/// A chunk of decoded values, aligned to a cache line so that the decoder's vector stores never
/// cross one.
#[repr(align(64))]
struct Scratch<T>([MaybeUninit<T>; FILTER_CHUNK_LEN]);

impl<T> Scratch<T> {
    /// Decodes chunk `chunk_idx` of `decoder` and returns its values.
    fn decode(&mut self, decoder: &impl ChunkDecoder<T>, chunk_idx: usize) -> &[T] {
        decoder.decode_chunk(chunk_idx, &mut self.0);
        // SAFETY: the decoder initialized every value.
        unsafe { self.0.assume_init_ref() }
    }
}

/// The output and scratch buffers of [`filter_chunked`].
struct ChunkedFilter<'a, T, D> {
    decoder: &'a D,
    values: BufferMut<T>,
    true_count: usize,
    written: usize,
    scratch: Scratch<T>,
}

impl<'a, T: Copy, D: ChunkDecoder<T>> ChunkedFilter<'a, T, D> {
    fn new(decoder: &'a D, true_count: usize, allocator: &BufferAllocatorRef) -> Self {
        Self {
            decoder,
            // The SIMD kernels can store one vector past the last value that they write.
            values: BufferMut::with_capacity_in(
                true_count + SLACK_BYTES / size_of::<T>(),
                allocator.clone(),
            ),
            true_count,
            written: 0,
            scratch: Scratch([const { MaybeUninit::uninit() }; FILTER_CHUNK_LEN]),
        }
    }

    /// Decodes chunk `chunk_idx` straight to the output.
    fn push_full_chunk(&mut self, chunk_idx: usize) {
        let dst = self.values.spare_capacity_mut()[self.written..]
            .first_chunk_mut::<FILTER_CHUNK_LEN>()
            .unwrap_or_else(|| unreachable!("a fully selected chunk has output capacity"));
        self.decoder.decode_chunk(chunk_idx, dst);
        self.written += FILTER_CHUNK_LEN;
    }

    /// Decodes the values at the sorted in-chunk `indices` of chunk `chunk_idx` to the output.
    fn push_indices(&mut self, chunk_idx: usize, indices: &[usize]) {
        let dst = &mut self.values.spare_capacity_mut()[self.written..][..indices.len()];
        self.decoder.decode_indices(chunk_idx, indices, dst);
        self.written += indices.len();
    }

    /// Filters chunk by chunk from the bitmap `bits`.
    fn push_bitmap(&mut self, offset: usize, bits: &BitBuffer) {
        let len = bits.len();
        let bits_bytes = bits.inner().as_slice();
        let mut words = [0u64; CHUNK_WORDS];
        let mut indices = [0usize; MAX_SPARSE_VALUES];

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

            if words.iter().all(|&w| w == 0) {
                continue;
            }

            let selected = words.iter().map(|w| w.count_ones() as usize).sum::<usize>();

            if selected == FILTER_CHUNK_LEN {
                self.push_full_chunk(chunk_idx);
            } else if selected <= max_sparse_values::<T>() {
                let mut n = 0;
                for (word_idx, &word) in words.iter().enumerate() {
                    let mut word = word;
                    while word != 0 {
                        indices[n] = chunk_offset + word_idx * 64 + word.trailing_zeros() as usize;
                        n += 1;
                        word &= word - 1;
                    }
                }
                self.push_indices(chunk_idx, &indices[..selected]);
            } else {
                let src =
                    &self.scratch.decode(self.decoder, chunk_idx)[chunk_offset..][..chunk_len];
                let dst = &mut self.values.spare_capacity_mut()[self.written..];
                let bits = MaskBits::Words {
                    words,
                    len: chunk_len,
                };
                // SAFETY: the words select only values of `src` and clear any bits past
                // `chunk_len`, and the output has room for every selected value plus the slack.
                let written = unsafe { compact_bits(src, bits, selected, dst) };
                debug_assert_eq!(written, selected);
                self.written += selected;
            }
        }
    }

    /// Filters chunk by chunk from the sorted, unique `mask_indices`.
    fn push_mask_indices(&mut self, offset: usize, mask_indices: &[usize]) {
        let mut indices = [0usize; MAX_SPARSE_VALUES];
        let mut rest = mask_indices;

        while let Some(&first) = rest.first() {
            let chunk_idx = (first + offset) / FILTER_CHUNK_LEN;
            let chunk_start = chunk_idx * FILTER_CHUNK_LEN;
            let chunk_end = chunk_start + FILTER_CHUNK_LEN - offset;
            let (chunk_indices, tail) = rest.split_at(rest.partition_point(|&i| i < chunk_end));
            rest = tail;

            // The in-chunk position of each selected value.
            let positions = chunk_indices.iter().map(|&i| i + offset - chunk_start);
            let selected = chunk_indices.len();

            if selected == FILTER_CHUNK_LEN {
                self.push_full_chunk(chunk_idx);
            } else if selected <= max_sparse_values::<T>() {
                for (index, position) in indices.iter_mut().zip(positions) {
                    *index = position;
                }
                self.push_indices(chunk_idx, &indices[..selected]);
            } else {
                let src = self.scratch.decode(self.decoder, chunk_idx);
                let dst = &mut self.values.spare_capacity_mut()[self.written..][..selected];
                for (dst, position) in dst.iter_mut().zip(positions) {
                    dst.write(src[position]);
                }
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
        let mut indices = [0usize; MAX_SPARSE_VALUES];

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
            } else if selected <= max_sparse_values::<T>() {
                for (index, chunk_index) in indices
                    .iter_mut()
                    .zip(runs.flat_map(|(start, end)| start..end))
                {
                    *index = chunk_index;
                }
                self.push_indices(chunk_idx, &indices[..selected]);
            } else {
                let src = self.scratch.decode(self.decoder, chunk_idx);
                let dst = &mut self.values.spare_capacity_mut()[self.written..];
                let mut written = 0;
                for (start, end) in runs {
                    dst[written..][..end - start].write_copy_of_slice(&src[start..end]);
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

    use super::BufferAllocatorRef;
    use super::ChunkDecoder;
    use super::FILTER_CHUNK_LEN;
    use super::filter_chunked;
    use crate::BitBuffer;
    use crate::filter::Selection;

    /// The representation of the test selection.
    #[derive(Clone, Copy, Debug)]
    enum Cached {
        Bitmap,
        Slices,
        Indices,
    }

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

    fn check<T: Copy + PartialEq + std::fmt::Debug + From<u8>>(
        bits: &BitBuffer,
        offset: usize,
        cached: Cached,
    ) {
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

        let slices = bits.set_slices().collect::<Vec<_>>();
        let indices = bits.set_indices().collect::<Vec<_>>();
        // SAFETY: all three representations come from `bits`.
        let selection = unsafe {
            match cached {
                Cached::Bitmap => Selection::from_bitmap_unchecked(bits, bits.true_count()),
                Cached::Slices => {
                    Selection::from_slices_unchecked(bits.len(), &slices, bits.true_count())
                }
                Cached::Indices => Selection::from_indices_unchecked(bits.len(), &indices),
            }
        };

        let filtered = filter_chunked(
            &SliceDecoder(&values),
            offset,
            selection,
            BufferAllocatorRef::static_ref(),
        );
        assert_eq!(filtered.as_slice(), expected);
    }

    #[rstest]
    fn filter_chunked_matches_expected(
        #[values(0.001, 0.01, 0.05, 0.2, 0.5, 0.9, 0.999)] density: f64,
        #[values(0, 3, 1000)] offset: usize,
        #[values(0, 5)] bit_offset: usize,
        #[values(Cached::Bitmap, Cached::Slices, Cached::Indices)] cached: Cached,
    ) {
        let mut rng = StdRng::seed_from_u64(42);
        let bits = (0..5000 + bit_offset)
            .map(|_| rng.random_bool(density))
            .collect::<BitBuffer>()
            .slice(bit_offset..5000 + bit_offset);

        check::<u8>(&bits, offset, cached);
        check::<u16>(&bits, offset, cached);
        check::<u32>(&bits, offset, cached);
        check::<u64>(&bits, offset, cached);
        check::<u128>(&bits, offset, cached);
    }

    /// Runs of selected values, some of which cross chunk boundaries.
    #[rstest]
    fn filter_chunked_runs_matches_expected(
        #[values(9, 12, 200, 3000)] run_len: usize,
        #[values(1, 16, 64, 1500)] gap: usize,
        #[values(0, 3, 1000)] offset: usize,
        #[values(Cached::Bitmap, Cached::Slices, Cached::Indices)] cached: Cached,
    ) {
        let bits = (0..5000)
            .map(|i| i % (run_len + gap) < run_len)
            .collect::<BitBuffer>();

        check::<u8>(&bits, offset, cached);
        check::<u16>(&bits, offset, cached);
        check::<u32>(&bits, offset, cached);
        check::<u64>(&bits, offset, cached);
        check::<u128>(&bits, offset, cached);
    }

    #[rstest]
    fn filter_chunked_full_and_empty_chunks(#[values(0, 3)] offset: usize) {
        // One whole chunk of selected values in every three chunks.
        let bits = (0..5000)
            .map(|i| ((i + offset) / FILTER_CHUNK_LEN).is_multiple_of(3))
            .collect::<BitBuffer>();

        for cached in [Cached::Bitmap, Cached::Slices, Cached::Indices] {
            check::<u8>(&bits, offset, cached);
            check::<u64>(&bits, offset, cached);
        }
    }
}
