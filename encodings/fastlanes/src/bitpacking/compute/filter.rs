// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem::MaybeUninit;

use fastlanes::BitPacking;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::filter::FilterKernel;
use vortex_array::arrays::filter::uses_simd_compress;
use vortex_array::dtype::NativePType;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_mask::MaskValues;

use crate::BitPacked;
use crate::BitPackedArrayExt;
use crate::BitPackedData;
use crate::BitWidthsView;

/// Number of values in a FastLanes chunk.
const CHUNK_LEN: usize = 1024;
/// Number of 64-bit mask words that cover a FastLanes chunk.
const CHUNK_WORDS: usize = CHUNK_LEN / 64;
/// Chunks with at most this many selected values per byte of value width unpack only those
/// values rather than the whole chunk. Wider values make a whole chunk more expensive to unpack,
/// so the threshold grows with the width, see [`sparse_chunk_threshold`].
const SPARSE_VALUES_PER_BYTE: usize = 16;
/// The largest [`sparse_chunk_threshold`], for 8-byte values.
const MAX_SPARSE_CHUNK_THRESHOLD: usize = SPARSE_VALUES_PER_BYTE * 8;
/// Masks at most this dense are always filtered chunk by chunk, since most chunks are skipped or
/// only partly unpacked.
const MAX_SPARSE_DENSITY: f64 = 0.02;
/// Selections whose runs of selected values average at least this length are compacted by
/// copying each run.
const MIN_COPIED_RUN_LEN: u32 = 8;

/// Kernel to execute filtering directly on a bit-packed array.
///
/// The selection is applied one FastLanes chunk at a time, from the mask's cached slices if they
/// form long runs and otherwise from its bitmap, without materializing selected indices. Chunks
/// without selected values are never unpacked, fully selected chunks are unpacked straight into
/// the output, sparsely selected chunks unpack only their selected values, and the remaining
/// chunks are unpacked into a cache-resident scratch buffer and then compacted into the output.
///
/// Masks that the canonical filter compacts with SIMD decline the kernel unless they are very
/// sparse or form long runs, see [`prefer_chunked_filter`].
impl FilterKernel for BitPacked {
    fn filter(
        array: ArrayView<'_, Self>,
        mask: &Mask,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let BitWidthsView::Global(bit_width) = array.bit_widths() else {
            return Ok(None);
        };
        let values = match mask {
            Mask::AllTrue(_) | Mask::AllFalse(_) => {
                return Ok(None);
            }
            Mask::Values(values) => values,
        };

        // FastLanes only unpacks unsigned types, so filter as unsigned and reinterpret the
        // resulting buffer with the array's (possibly signed) ptype.
        let ptype = array.dtype().as_ptype();
        if !match_each_unsigned_integer_ptype!(ptype.to_unsigned(), |U| {
            prefer_chunked_filter::<U>(values)
        }) {
            return Ok(None);
        }

        let validity = array.validity()?.filter(mask)?;
        let buffer = match_each_unsigned_integer_ptype!(ptype.to_unsigned(), |U| {
            match long_run_slices(values) {
                Some(slices) => filter_values_by_slices::<U>(
                    array.data(),
                    bit_width,
                    slices,
                    values.true_count(),
                ),
                None => filter_values::<U>(array.data(), bit_width, values),
            }
            .into_byte_buffer()
        });
        let primitive = PrimitiveArray::from_byte_buffer(buffer, ptype, validity);

        let patches = array
            .patches()
            .map(|patches| patches.filter(mask, ctx))
            .transpose()?
            .flatten();

        Ok(Some(match patches {
            Some(patches) => primitive.patch(&patches, ctx)?.into_array(),
            None => primitive.into_array(),
        }))
    }
}

/// Returns whether to filter `T` values chunk by chunk rather than unpack the whole array and
/// filter the unpacked values with the canonical filter.
///
/// Above the sparsest masks, the canonical filter's SIMD compress, where the target has one for
/// `T`'s width and the mask's density, is faster than compacting each chunk unless the selected
/// values form long runs. Without SIMD compress, filtering chunk by chunk is faster, because it
/// skips unselected chunks and never materializes the whole unpacked array.
fn prefer_chunked_filter<T>(mask: &MaskValues) -> bool {
    long_run_slices(mask).is_some()
        || mask.density() <= MAX_SPARSE_DENSITY
        || !uses_simd_compress::<T>(mask)
}

/// Returns the mask's cached slices if its runs of selected values average at least
/// [`MIN_COPIED_RUN_LEN`] values.
///
/// Shorter runs are filtered faster from the bitmap than by walking the slices one by one.
fn long_run_slices(mask: &MaskValues) -> Option<&[(usize, usize)]> {
    mask.cached_slices()
        .filter(|slices| mask.true_count() >= MIN_COPIED_RUN_LEN as usize * slices.len())
}

/// Returns the largest number of selected `T` values for which a chunk unpacks only those values
/// rather than the whole chunk.
const fn sparse_chunk_threshold<T>() -> usize {
    SPARSE_VALUES_PER_BYTE * size_of::<T>()
}

/// Unpacks the values of `array` selected by `mask`, ignoring patches and validity.
///
/// Because the FastLanes bit-packing kernels are only implemented for unsigned types, `T` must
/// be the unsigned variant of the array's ptype.
fn filter_values<T: NativePType + BitPacking>(
    array: &BitPackedData,
    bit_width: u8,
    mask: &MaskValues,
) -> Buffer<T> {
    let len = mask.len();
    let offset = array.offset() as usize;
    let bit_width = bit_width as usize;
    let packed = array.packed_slice::<T>();
    let packed_chunk_len = 128 * bit_width / size_of::<T>();

    let bits = mask.bit_buffer();
    let bits_bytes = bits.inner().as_slice();

    let true_count = mask.true_count();
    let mut values = BufferMut::<T>::with_capacity(true_count);
    let out = values.spare_capacity_mut().as_mut_ptr().cast::<T>();
    let mut written = 0;

    let mut unpacked = [const { MaybeUninit::<T>::uninit() }; CHUNK_LEN];
    let mut words = [0u64; CHUNK_WORDS];
    let mut indices = [0usize; MAX_SPARSE_CHUNK_THRESHOLD];
    let indices = &mut indices[..sparse_chunk_threshold::<T>()];

    for chunk_idx in 0..(offset + len).div_ceil(CHUNK_LEN) {
        // The logical range `start..end` of the array that falls within this chunk, which begins
        // at index `chunk_offset` within the chunk.
        let chunk_start = chunk_idx * CHUNK_LEN;
        let start = chunk_start.saturating_sub(offset);
        let end = (chunk_start + CHUNK_LEN - offset).min(len);
        let chunk_offset = start + offset - chunk_start;

        load_chunk_words(bits_bytes, bits.offset() + start, end - start, &mut words);
        // Classify the chunk with bitwise reductions and a bounded scan rather than a popcount,
        // which is not a native instruction on baseline x86-64.
        if words.iter().all(|&w| w == 0) {
            continue;
        }
        let words = &words[..(end - start).div_ceil(64)];

        let packed = &packed[chunk_idx * packed_chunk_len..][..packed_chunk_len];

        if words.len() == CHUNK_WORDS && words.iter().all(|&w| w == u64::MAX) {
            // SAFETY: the output has capacity for every selected value, including this chunk.
            unsafe {
                let dst = std::slice::from_raw_parts_mut(out.add(written), CHUNK_LEN);
                BitPacking::unchecked_unpack(bit_width, packed, dst);
            }
            written += CHUNK_LEN;
        } else if let Some(indices) = sparse_indices(words, chunk_offset, indices) {
            // SAFETY: every index is below `CHUNK_LEN` and the output has capacity for every
            // selected value.
            unsafe {
                let dst = std::slice::from_raw_parts_mut(
                    out.add(written).cast::<MaybeUninit<T>>(),
                    indices.len(),
                );
                BitPacking::unchecked_unpack_indices(bit_width, packed, indices, dst);
            }
            written += indices.len();
        } else {
            // SAFETY: `MaybeUninit<T>` has the same layout as `T`, and the unpack initializes all
            // `CHUNK_LEN` values.
            let unpacked = unsafe {
                let dst = std::mem::transmute::<&mut [MaybeUninit<T>], &mut [T]>(&mut unpacked);
                BitPacking::unchecked_unpack(bit_width, packed, dst);
                &*dst
            };
            for (word_idx, &word) in words.iter().enumerate() {
                let word_offset = chunk_offset + word_idx * 64;
                let src = &unpacked[word_offset..(word_offset + 64).min(CHUNK_LEN)];
                // SAFETY: the output has capacity for every selected value.
                written += unsafe { compact_word(word, src, out.add(written)) };
            }
        }
    }

    debug_assert_eq!(written, true_count);
    // SAFETY: the first `true_count` output values were initialized above.
    unsafe { values.set_len(true_count) };
    values.freeze()
}

/// Unpacks the values of `array` within the sorted, disjoint `slices`, ignoring patches and
/// validity.
///
/// Walking the ranges directly avoids scanning the whole mask bitmap when the selection is
/// already known to be grouped into runs.
fn filter_values_by_slices<T: NativePType + BitPacking>(
    array: &BitPackedData,
    bit_width: u8,
    slices: &[(usize, usize)],
    true_count: usize,
) -> Buffer<T> {
    let offset = array.offset() as usize;
    let bit_width = bit_width as usize;
    let packed = array.packed_slice::<T>();
    let packed_chunk_len = 128 * bit_width / size_of::<T>();

    let mut values = BufferMut::<T>::with_capacity(true_count);
    let out = values.spare_capacity_mut().as_mut_ptr().cast::<T>();
    let mut written = 0;

    let mut unpacked = [const { MaybeUninit::<T>::uninit() }; CHUNK_LEN];
    let mut indices = [0usize; MAX_SPARSE_CHUNK_THRESHOLD];

    // Slices are processed one chunk at a time. `slices[next..]` holds the slices that are not yet
    // fully processed, and `resume` is the start of the first chunk not yet processed, from which
    // `slices[next]` resumes if it began in an earlier chunk. Positions are relative to the start
    // of the first chunk.
    let mut next = 0;
    let mut resume = 0;
    while let Some(&(first_start, _)) = slices.get(next) {
        let chunk_idx = (first_start + offset).max(resume) / CHUNK_LEN;
        let chunk_start = chunk_idx * CHUNK_LEN;
        let chunk_end = chunk_start + CHUNK_LEN;

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
        let packed = &packed[chunk_idx * packed_chunk_len..][..packed_chunk_len];

        let mut chunk_true_count = 0;
        for (start, end) in runs.clone() {
            chunk_true_count += end - start;
        }

        if chunk_true_count == CHUNK_LEN {
            // SAFETY: the output has capacity for every selected value, including this chunk.
            unsafe {
                let dst = std::slice::from_raw_parts_mut(out.add(written), CHUNK_LEN);
                BitPacking::unchecked_unpack(bit_width, packed, dst);
            }
        } else if chunk_true_count <= sparse_chunk_threshold::<T>() {
            for (index, chunk_index) in indices
                .iter_mut()
                .zip(runs.clone().flat_map(|(start, end)| start..end))
            {
                *index = chunk_index;
            }
            // SAFETY: every index is below `CHUNK_LEN` and the output has capacity for every
            // selected value.
            unsafe {
                let dst = std::slice::from_raw_parts_mut(
                    out.add(written).cast::<MaybeUninit<T>>(),
                    chunk_true_count,
                );
                BitPacking::unchecked_unpack_indices(
                    bit_width,
                    packed,
                    &indices[..chunk_true_count],
                    dst,
                );
            }
        } else {
            // SAFETY: `MaybeUninit<T>` has the same layout as `T` and the unpack initializes all
            // `CHUNK_LEN` values.
            let unpacked = unsafe {
                let dst = std::mem::transmute::<&mut [MaybeUninit<T>], &mut [T]>(&mut unpacked);
                BitPacking::unchecked_unpack(bit_width, packed, dst);
                &*dst
            };
            let mut chunk_written = 0;
            for (start, end) in runs.clone() {
                // SAFETY: the output has capacity for every selected value.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        unpacked.as_ptr().add(start),
                        out.add(written + chunk_written),
                        end - start,
                    );
                }
                chunk_written += end - start;
            }
        }
        written += chunk_true_count;

        // Skip the slices that end within this chunk; the last one may continue into the next.
        next += slices[next..]
            .iter()
            .take_while(|&&(_, end)| end + offset <= chunk_end)
            .count();
        resume = chunk_end;
    }

    debug_assert_eq!(written, true_count);
    // SAFETY: the first `true_count` output values were initialized above.
    unsafe { values.set_len(true_count) };
    values.freeze()
}

/// Collects the in-chunk indices of the bits set in `words`, whose first bit is chunk index
/// `chunk_offset`, or returns `None` if more bits are set than `indices` can hold.
#[inline]
fn sparse_indices<'a>(
    words: &[u64],
    chunk_offset: usize,
    indices: &'a mut [usize],
) -> Option<&'a [usize]> {
    let mut n = 0;
    for (word_idx, &word) in words.iter().enumerate() {
        let word_offset = chunk_offset + word_idx * 64;
        let mut word = word;
        while word != 0 {
            *indices.get_mut(n)? = word_offset + word.trailing_zeros() as usize;
            n += 1;
            word &= word - 1;
        }
    }
    Some(&indices[..n])
}

/// Number of mask bytes read to assemble the words of a chunk at an arbitrary bit offset.
const CHUNK_WORD_BYTES: usize = CHUNK_LEN / 8 + 8;

/// Loads the `len <= CHUNK_LEN` mask bits starting at bit `start` of `bytes` into `words`, with
/// any bits past `len` cleared.
#[inline]
fn load_chunk_words(bytes: &[u8], start: usize, len: usize, words: &mut [u64; CHUNK_WORDS]) {
    debug_assert!(len <= CHUNK_LEN);
    let byte_start = start / 8;
    let shift = start % 8;

    if len == CHUNK_LEN {
        if shift == 0
            && let Some(src) = bytes[byte_start..].first_chunk::<{ CHUNK_LEN / 8 }>()
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
        let valid = len.saturating_sub(i * 64).min(64);
        *word &= u64::MAX.checked_shr((64 - valid) as u32).unwrap_or(0);
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

/// Returns the number of runs of set bits in `word`.
#[inline]
fn runs(word: u64) -> u32 {
    (word & !(word << 1)).count_ones()
}

/// Copies the values of `src` whose bit is set in `word` to `dst`, returning the number copied.
///
/// # Safety
///
/// `dst` must be valid for writes of `word.count_ones()` values, and every set bit of `word` must
/// index into `src`.
#[inline]
unsafe fn compact_word<T: Copy>(word: u64, src: &[T], dst: *mut T) -> usize {
    if word == u64::MAX {
        // SAFETY: a full word selects 64 values of `src`, all of which fit in `dst`.
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), dst, 64) };
        return 64;
    }

    let selected = word.count_ones();
    let mut written = 0;
    let mut word = word;

    if selected >= MIN_COPIED_RUN_LEN && selected >= MIN_COPIED_RUN_LEN * runs(word) {
        // Long runs of selected values, e.g. from a list-level selection, are copied whole.
        while word != 0 {
            let run_start = word.trailing_zeros();
            let run_len = (!(word >> run_start)).trailing_zeros() as usize;
            // SAFETY: the run lies within the selected bits of `src` and fits in `dst`.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src.as_ptr().add(run_start as usize),
                    dst.add(written),
                    run_len,
                )
            };
            written += run_len;
            // Adding the run's lowest bit carries through the run, clearing it.
            word &= word.wrapping_add(1 << run_start);
        }
    } else {
        while word != 0 {
            // SAFETY: set bits index into `src` and `dst` has room for every selected value.
            unsafe {
                dst.add(written)
                    .write(*src.get_unchecked(word.trailing_zeros() as usize));
            }
            written += 1;
            word &= word - 1;
        }
    }

    written
}

#[cfg(test)]
mod tests {
    use std::ops::Range;
    use std::sync::LazyLock;

    use fastlanes::BitPacking;
    use rand::RngExt;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray as _;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::slice::SliceKernel;
    use vortex_array::assert_arrays_eq;
    use vortex_array::compute::conformance::filter::test_filter_conformance;
    use vortex_array::dtype::NativePType;
    use vortex_array::validity::Validity;
    use vortex_buffer::BitBuffer;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_mask::Mask;
    use vortex_session::VortexSession;

    use super::filter_values;
    use super::filter_values_by_slices;
    use crate::BitPacked;
    use crate::BitPackedData;
    use crate::bitpacking::array::BitPackedArrayExt;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        session
    });

    #[test]
    fn take_indices() {
        let mut ctx = SESSION.create_execution_ctx();
        // Create a u8 array modulo 63.
        let unpacked = PrimitiveArray::from_iter((0..4096).map(|i| (i % 63) as u8));
        let bitpacked = BitPackedData::encode(&unpacked.into_array(), 6, &mut ctx).unwrap();

        let mask = Mask::from_indices(bitpacked.len(), vec![0, 125, 2047, 2049, 2151, 2790]);

        let primitive_result = bitpacked.filter(mask).unwrap();
        assert_arrays_eq!(
            primitive_result,
            PrimitiveArray::from_iter([0u8, 62, 31, 33, 9, 18]),
            &mut ctx
        );
    }

    #[test]
    fn take_sliced_indices() {
        let mut ctx = SESSION.create_execution_ctx();
        // Create a u8 array modulo 63.
        let unpacked = PrimitiveArray::from_iter((0..4096).map(|i| (i % 63) as u8));
        let bitpacked = BitPackedData::encode(&unpacked.into_array(), 6, &mut ctx).unwrap();
        let sliced = bitpacked.slice(128..2050).unwrap();

        let mask = Mask::from_indices(sliced.len(), vec![1919, 1921]);

        let primitive_result = sliced.filter(mask).unwrap();
        assert_arrays_eq!(
            primitive_result,
            PrimitiveArray::from_iter([31u8, 33]),
            &mut ctx
        );
    }

    #[test]
    fn filter_bitpacked() {
        let mut ctx = SESSION.create_execution_ctx();
        let unpacked = PrimitiveArray::from_iter((0..4096).map(|i| (i % 63) as u8));
        let bitpacked = BitPackedData::encode(&unpacked.into_array(), 6, &mut ctx).unwrap();
        let filtered = bitpacked.filter(Mask::from_indices(4096, 0..1024)).unwrap();
        let filtered_prim = filtered.execute::<PrimitiveArray>(&mut ctx).unwrap();
        assert_arrays_eq!(
            filtered_prim,
            PrimitiveArray::from_iter((0..1024).map(|i| (i % 63) as u8)),
            &mut ctx
        );
    }

    #[test]
    fn filter_bitpacked_signed() {
        let mut ctx = SESSION.create_execution_ctx();
        let values: Buffer<i64> = (0..500).collect();
        let unpacked = PrimitiveArray::new(values.clone(), Validity::NonNullable);
        let bitpacked = BitPackedData::encode(&unpacked.into_array(), 9, &mut ctx).unwrap();
        let filtered = bitpacked
            .filter(Mask::from_indices(values.len(), 0..250))
            .unwrap()
            .execute::<PrimitiveArray>(&mut ctx)
            .unwrap();

        assert_arrays_eq!(
            filtered,
            PrimitiveArray::from_iter(values[0..250].iter().copied()),
            &mut ctx
        );
    }

    #[test]
    fn test_filter_bitpacked_conformance() {
        let mut ctx = SESSION.create_execution_ctx();
        // Test with u8 values
        let unpacked = buffer![1u8, 2, 3, 4, 5].into_array();
        let bitpacked = BitPackedData::encode(&unpacked, 3, &mut ctx).unwrap();
        test_filter_conformance(&bitpacked.into_array(), &mut ctx);

        // Test with u32 values
        let unpacked = buffer![100u32, 200, 300, 400, 500].into_array();
        let bitpacked = BitPackedData::encode(&unpacked, 9, &mut ctx).unwrap();
        test_filter_conformance(&bitpacked.into_array(), &mut ctx);

        // Test with nullable values
        let unpacked = PrimitiveArray::from_option_iter([Some(1u16), None, Some(3), Some(4), None]);
        let bitpacked = BitPackedData::encode(&unpacked.into_array(), 3, &mut ctx).unwrap();
        test_filter_conformance(&bitpacked.into_array(), &mut ctx);
    }

    /// Regression test for signed integers with patches.
    ///
    /// When filtering signed integers that have patches (exceptions), the patches
    /// are stored with the signed type but FastLanes uses unsigned types internally.
    /// This test ensures that the type handling is correct.
    #[test]
    fn filter_bitpacked_signed_with_patches() {
        let mut ctx = SESSION.create_execution_ctx();
        // Create signed integer values where some exceed the bit width (causing patches).
        // Values 0-127 fit in 7 bits, but 1000 and 2000 do not.
        let values: Vec<i32> = vec![0, 10, 1000, 20, 30, 2000, 40, 50, 60, 70];
        let unpacked = PrimitiveArray::from_iter(values.clone());
        let bitpacked = BitPackedData::encode(&unpacked.into_array(), 7, &mut ctx).unwrap();
        assert!(
            bitpacked.patches().is_some(),
            "Expected patches for values exceeding bit width"
        );

        // Filter to include some patched and some non-patched values.
        let filtered = bitpacked
            .filter(Mask::from_indices(values.len(), vec![0, 2, 5, 9]))
            .unwrap()
            .execute::<PrimitiveArray>(&mut ctx)
            .unwrap();

        assert_arrays_eq!(
            filtered,
            PrimitiveArray::from_iter([0i32, 1000, 2000, 70]),
            &mut ctx
        );
    }

    /// Regression test for signed integers with patches using low selectivity.
    ///
    /// This test uses a low selectivity filter which takes a different code path
    /// that doesn't fully decompress the array first.
    #[test]
    fn filter_bitpacked_signed_with_patches_low_selectivity() {
        let mut ctx = SESSION.create_execution_ctx();
        // Create a larger array with signed integers and some patches.
        let values: Vec<i32> = (0..1000)
            .map(|i| {
                if i % 100 == 0 {
                    10000 + i // These will be patches (exceed 7 bits)
                } else {
                    i % 128 // These fit in 7 bits
                }
            })
            .collect();
        let unpacked = PrimitiveArray::from_iter(values.clone());
        let bitpacked = BitPackedData::encode(&unpacked.into_array(), 7, &mut ctx).unwrap();
        assert!(
            bitpacked.patches().is_some(),
            "Expected patches for values exceeding bit width"
        );

        // Use low selectivity (only select 2% of values) to avoid full decompression.
        let indices: Vec<usize> = (0..20).collect();
        let filtered = bitpacked
            .filter(Mask::from_indices(values.len(), indices))
            .unwrap()
            .execute::<PrimitiveArray>(&mut ctx)
            .unwrap();

        let expected: Vec<i32> = values[0..20].to_vec();
        assert_arrays_eq!(filtered, PrimitiveArray::from_iter(expected), &mut ctx);
    }

    /// Selection patterns that exercise every chunk strategy of the kernel.
    #[derive(Clone, Copy, Debug)]
    enum Pattern {
        /// Each value is selected independently with the given probability.
        Random(f64),
        /// Runs of 12 values every 64 values, which are compacted by copying runs.
        Runs,
        /// Half of the values in the first 400 of every 1024 positions, which are too dense
        /// within a chunk to unpack individually but sparse enough overall to filter by chunk.
        Clustered,
        /// Everything except a single value, so most chunks are fully selected.
        AllButOne,
    }

    fn selection(pattern: Pattern, len: usize) -> BitBuffer {
        let mut rng = StdRng::seed_from_u64(42);
        BitBuffer::from_iter((0..len).map(|i| match pattern {
            Pattern::Random(density) => rng.random_bool(density),
            Pattern::Runs => i % 64 < 12,
            Pattern::Clustered => i % 1024 < 400 && rng.random_bool(0.5),
            Pattern::AllButOne => i != len / 2,
        }))
    }

    /// A bit-packed array of `len + 2000` values with some patches, sliced to `range`.
    fn sliced_bitpacked(len: usize, range: Range<usize>) -> VortexResult<(ArrayRef, ArrayRef)> {
        let mut ctx = SESSION.create_execution_ctx();
        let values =
            PrimitiveArray::from_option_iter((0..len as u32).map(|i| {
                (i % 97 != 0).then_some(if i % 501 == 0 { 100_000 + i } else { i % 1000 })
            }));
        let bitpacked = BitPackedData::encode(&values.clone().into_array(), 10, &mut ctx)?;
        assert!(bitpacked.patches().is_some());
        // Slicing patches needs execution, so slice with the kernel rather than lazily.
        let sliced =
            <BitPacked as SliceKernel>::slice(bitpacked.as_view(), range.clone(), &mut ctx)?
                .unwrap();
        Ok((values.into_array().slice(range)?, sliced))
    }

    #[rstest]
    fn filter_matches_canonical(
        #[values(
            Pattern::Random(0.0005),
            Pattern::Random(0.01),
            Pattern::Random(0.05),
            Pattern::Random(0.15),
            Pattern::Random(0.5),
            Pattern::Runs,
            Pattern::Clustered,
            Pattern::AllButOne
        )]
        pattern: Pattern,
        #[values(0..5000, 3..5000, 1000..4099, 1024..3072)] range: Range<usize>,
        #[values(false, true)] cached_slices: bool,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let (values, bitpacked) = sliced_bitpacked(7000, range)?;
        assert!(bitpacked.is::<BitPacked>());

        let bits = selection(pattern, values.len());
        let mask = if cached_slices {
            Mask::from_slices(bits.len(), bits.set_slices().collect())
        } else {
            Mask::from_buffer(bits)
        };

        let expected = values
            .filter(mask.clone())?
            .execute::<PrimitiveArray>(&mut ctx)?;
        let actual = bitpacked
            .filter(mask)?
            .execute::<PrimitiveArray>(&mut ctx)?;
        assert_arrays_eq!(actual, expected, &mut ctx);
        Ok(())
    }

    /// Checks both unpacking strategies directly, including on masks for which the kernel would
    /// rather unpack the whole array.
    #[rstest]
    fn filter_values_matches_canonical(
        #[values(
            Pattern::Random(0.0005),
            Pattern::Random(0.05),
            Pattern::Random(0.15),
            Pattern::Random(0.5),
            Pattern::Random(0.95),
            Pattern::Runs,
            Pattern::Clustered,
            Pattern::AllButOne
        )]
        pattern: Pattern,
        #[values(0..5000, 3..5000, 1000..4099, 1024..3072)] range: Range<usize>,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let len = range.len();
        let unpacked = PrimitiveArray::from_iter((0..7000u32).map(|i| i % 1000));
        let bitpacked = BitPackedData::encode(&unpacked.into_array(), 10, &mut ctx)?
            .into_array()
            .slice(range.clone())?;
        let bitpacked = bitpacked.as_::<BitPacked>();

        let bits = selection(pattern, len);
        let expected: Vec<u32> = (range.start as u32..range.end as u32)
            .zip(bits.iter())
            .filter_map(|(i, selected)| selected.then_some(i % 1000))
            .collect();

        let Mask::Values(mask) = Mask::from_buffer(bits.clone()) else {
            unreachable!("patterns select some but not all values")
        };
        assert_eq!(
            filter_values::<u32>(bitpacked.data(), 10, &mask).as_slice(),
            expected
        );

        let slices: Vec<_> = bits.set_slices().collect();
        assert_eq!(
            filter_values_by_slices::<u32>(bitpacked.data(), 10, &slices, expected.len())
                .as_slice(),
            expected
        );
        Ok(())
    }

    /// Checks both unpacking strategies for every value width, since the number of selected
    /// values for which a chunk unpacks only those values depends on the width.
    #[rstest]
    fn filter_values_matches_expected_for_every_width(
        #[values(
            Pattern::Random(0.01),
            Pattern::Random(0.03),
            Pattern::Random(0.08),
            Pattern::Random(0.15),
            Pattern::Runs,
            Pattern::Clustered
        )]
        pattern: Pattern,
    ) -> VortexResult<()> {
        check_filter_values::<u8>(pattern)?;
        check_filter_values::<u16>(pattern)?;
        check_filter_values::<u32>(pattern)?;
        check_filter_values::<u64>(pattern)
    }

    fn check_filter_values<T: NativePType + BitPacking + From<u8>>(
        pattern: Pattern,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let range = 3..5000;

        let unpacked = PrimitiveArray::from_iter((0..5000u32).map(|i| T::from((i % 100) as u8)));
        let bitpacked = BitPackedData::encode(&unpacked.into_array(), 7, &mut ctx)?
            .into_array()
            .slice(range.clone())?;
        let bitpacked = bitpacked.as_::<BitPacked>();

        let bits = selection(pattern, range.len());
        let expected: Vec<T> = range
            .zip(bits.iter())
            .filter_map(|(i, selected)| selected.then_some(T::from((i % 100) as u8)))
            .collect();

        let Mask::Values(mask) = Mask::from_buffer(bits.clone()) else {
            unreachable!("patterns select some but not all values")
        };
        assert_eq!(
            filter_values::<T>(bitpacked.data(), 7, &mask).as_slice(),
            expected
        );

        let slices: Vec<_> = bits.set_slices().collect();
        assert_eq!(
            filter_values_by_slices::<T>(bitpacked.data(), 7, &slices, expected.len()).as_slice(),
            expected
        );
        Ok(())
    }
}
