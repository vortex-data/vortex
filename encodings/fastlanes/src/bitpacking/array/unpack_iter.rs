// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem::MaybeUninit;
use std::ops::Range;

use fastlanes::BitPacking;
use vortex_array::dtype::PhysicalPType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;

use crate::BitPackedData;
use crate::FL_CHUNK_SIZE;

const CHUNK_SIZE: usize = FL_CHUNK_SIZE;

/// Borrow uninitialized integer storage as its unsigned FastLanes physical type.
pub(crate) fn physical_output<T: PhysicalPType>(
    output: &mut [MaybeUninit<T>],
) -> &mut [MaybeUninit<T::Physical>] {
    const {
        assert!(size_of::<T>() == size_of::<T::Physical>());
        assert!(align_of::<T>() == align_of::<T::Physical>());
    }
    // SAFETY: the element layouts agree, and both views permit uninitialized bytes.
    unsafe { std::slice::from_raw_parts_mut(output.as_mut_ptr().cast(), output.len()) }
}

/// Strategy trait for fastlanes unpacking operations
pub trait UnpackStrategy<T: PhysicalPType> {
    /// Unpack a chunk of packed data into the destination buffer
    ///
    /// # Safety
    /// - `chunk` must contain exactly `elems_per_chunk` elements
    /// - `dst` must have exactly CHUNK_SIZE capacity
    unsafe fn unpack_chunk(&self, bit_width: usize, chunk: &[T::Physical], dst: &mut [T::Physical]);

    /// Unpack into fresh storage and return the initialized destination.
    ///
    /// # Safety
    /// The same packed and unpacked lengths are required as for [`Self::unpack_chunk`].
    unsafe fn unpack_chunk_uninit<'a>(
        &self,
        bit_width: usize,
        chunk: &[T::Physical],
        dst: &'a mut [MaybeUninit<T::Physical>],
    ) -> &'a mut [T::Physical] {
        dst.fill(MaybeUninit::new(T::Physical::default()));
        // SAFETY: every slot was initialized above.
        let dst = unsafe { dst.assume_init_mut() };
        // SAFETY: the caller guarantees both chunk lengths.
        unsafe { self.unpack_chunk(bit_width, chunk, dst) };
        dst
    }
}

/// BitPacking strategy - uses plain bitpacking without reference value
pub struct BitPackingStrategy;

impl<T: PhysicalPType<Physical: BitPacking>> UnpackStrategy<T> for BitPackingStrategy {
    #[allow(clippy::inline_always)]
    #[inline(always)]
    unsafe fn unpack_chunk(
        &self,
        bit_width: usize,
        chunk: &[T::Physical],
        dst: &mut [T::Physical],
    ) {
        // SAFETY: Caller must ensure [`BitPacking::unchecked_unpack`] safety requirements hold.
        unsafe {
            BitPacking::unchecked_unpack(bit_width, chunk, dst);
        }
    }

    unsafe fn unpack_chunk_uninit<'a>(
        &self,
        bit_width: usize,
        chunk: &[T::Physical],
        dst: &'a mut [MaybeUninit<T::Physical>],
    ) -> &'a mut [T::Physical] {
        // SAFETY: the caller guarantees both chunk lengths, and the unpack initializes every slot.
        unsafe {
            BitPacking::unchecked_unpack_uninit(bit_width, chunk, dst);
            dst.assume_init_mut()
        }
    }
}

/// Unpack a validated full chunk into logical integer storage.
fn unpack_chunk<'a, T: PhysicalPType, S: UnpackStrategy<T>>(
    strategy: &S,
    width: usize,
    chunk: &[T::Physical],
    dst: &'a mut [MaybeUninit<T>],
) -> &'a mut [T] {
    let expected = dst.as_mut_ptr_range();
    // SAFETY: callers pass a full destination and a validated chunk at this width.
    let initialized = unsafe { strategy.unpack_chunk_uninit(width, chunk, physical_output(dst)) };
    // UnpackStrategy is publicly implementable. Check that the initialized slice actually
    // covers the requested destination before the caller assumes it initialized.
    let actual = initialized.as_mut_ptr_range();
    assert!(actual.start.cast() == expected.start && actual.end.cast() == expected.end);
    // SAFETY: the initialized output covers every slot, and every bit pattern is valid for T.
    unsafe { dst.assume_init_mut() }
}

/// Accessor to unpacked chunks of bitpacked arrays
///
/// The usual pattern of usage should follow
/// ```
/// use std::mem::MaybeUninit;
///
/// use vortex_array::IntoArray;
/// use vortex_session::VortexSession;
/// use vortex_array::VortexSessionExecute;
/// use vortex_buffer::buffer;
/// use vortex_fastlanes::BitPackedData;
/// use vortex_fastlanes::BitPackedArrayExt;
/// use vortex_fastlanes::unpack_iter::BitUnpackedChunks;
///
/// let mut ctx = VortexSession::empty().create_execution_ctx();
/// let array = BitPackedData::encode(&buffer![2, 3, 4, 5].into_array(), 2, &mut ctx).unwrap();
/// let mut scratch = [const { MaybeUninit::<i32>::uninit() }; 1024];
/// let mut unpacked_chunks: BitUnpackedChunks<i32> =
///     array.unpacked_chunks(&mut scratch).unwrap();
///
/// if let Some(header) = unpacked_chunks.initial() {
///    // handle partial initial chunk
/// }
///
/// {
///     let mut chunks_iter = unpacked_chunks.full_chunks();
///     while let Some(chunk) = chunks_iter.next_chunk() {
///         // handle full bitpacked chunks of 1024 elements
///     }
/// }
///
/// if let Some(trailer) = unpacked_chunks.trailer() {
///     // handle partial trailing chunk
/// }
/// ```
pub struct UnpackedChunks<'a, T: PhysicalPType, S: UnpackStrategy<T>> {
    strategy: S,
    bit_width: usize,
    offset: usize,
    len: usize,
    num_chunks: usize,
    // 0 indicates full chunk of CHUNK_SIZE
    last_chunk_length: usize,
    packed: &'a [T::Physical],
    scratch: &'a mut [MaybeUninit<T>; CHUNK_SIZE],
}

pub type BitUnpackedChunks<'a, T> = UnpackedChunks<'a, T, BitPackingStrategy>;

impl<'a, T: BitPacked> BitUnpackedChunks<'a, T> {
    pub fn try_new(
        array: &'a BitPackedData,
        len: usize,
        scratch: &'a mut [MaybeUninit<T>; CHUNK_SIZE],
    ) -> VortexResult<Self> {
        let Some(bit_width) = array.global_bit_width else {
            vortex_bail!("BitPacked array has per-block bit widths");
        };
        Self::try_new_with_strategy(
            BitPackingStrategy,
            array.packed_slice::<T::Physical>(),
            bit_width as usize,
            array.offset() as usize,
            len,
            scratch,
        )
    }

    pub fn full_chunks(&mut self) -> BitUnpackIterator<'_, T> {
        let elems_per_chunk = self.elems_per_chunk();
        let last_chunk_is_sliced = self.last_chunk_is_sliced() as usize;
        let first_chunk_is_sliced = self.first_chunk_is_sliced();
        BitUnpackIterator::new(
            self.packed,
            self.scratch,
            self.bit_width,
            elems_per_chunk,
            self.num_chunks - last_chunk_is_sliced,
            first_chunk_is_sliced,
        )
    }
}

impl<'a, T: PhysicalPType, S: UnpackStrategy<T>> UnpackedChunks<'a, T, S> {
    pub fn try_new_with_strategy(
        strategy: S,
        packed: &'a [T::Physical],
        bit_width: usize,
        offset: usize,
        len: usize,
        scratch: &'a mut [MaybeUninit<T>; CHUNK_SIZE],
    ) -> VortexResult<Self> {
        let (num_chunks, last_chunk_length) =
            validate_packed::<T>(packed.len(), bit_width, offset, len)?;
        Ok(Self {
            strategy,
            bit_width,
            offset,
            len,
            num_chunks,
            last_chunk_length,
            packed,
            scratch,
        })
    }

    #[allow(clippy::inline_always)]
    #[inline(always)]
    const fn elems_per_chunk(&self) -> usize {
        128 * self.bit_width / size_of::<T>()
    }

    /// Access the first chunk of the array if it has fewer than 1024 elements.
    pub fn initial(&mut self) -> Option<&mut [T]> {
        (self.len > 0
            && (self.first_chunk_is_sliced()
                || (self.num_chunks == 1 && self.last_chunk_is_sliced())))
        .then(|| {
            let chunk: &[T::Physical] = &self.packed[..self.elems_per_chunk()];

            let header_end_slice = if self.num_chunks == 1 {
                self.len
            } else {
                CHUNK_SIZE - self.offset
            };
            let values = unpack_chunk(&self.strategy, self.bit_width, chunk, self.scratch);
            &mut values[self.offset..][..header_end_slice]
        })
    }

    /// Decode all chunks (initial, full, and trailer) directly into the output range.
    pub fn decode_into(&mut self, output: &mut [MaybeUninit<T>]) {
        debug_assert_eq!(output.len(), self.len);
        let mut local_idx = 0;

        if let Some(initial) = self.initial() {
            local_idx = initial.len();

            output[..local_idx].write_copy_of_slice(initial);
        }

        local_idx = self.decode_full_chunks_into_at(output, local_idx);

        if let Some(trailer) = self.trailer() {
            output[local_idx..][..trailer.len()].write_copy_of_slice(trailer);
            local_idx += trailer.len();
        }

        debug_assert_eq!(local_idx, self.len);
    }

    /// Decode all chunks (initial, full, and trailer), mapping each unpacked value through f.
    pub(crate) fn decode_map_into<U>(
        &mut self,
        output: &mut [MaybeUninit<U>],
        mut f: impl FnMut(T) -> U,
    ) {
        debug_assert_eq!(output.len(), self.len);

        self.for_each_unpacked_chunk(|chunk, range| {
            write_map(chunk, &mut output[range], &mut f);
        });
    }

    /// Walk every unpacked chunk in array order, reusing the internal scratch buffer.
    pub(crate) fn for_each_unpacked_chunk<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut [T], Range<usize>),
    {
        let mut local_idx = 0;

        if let Some(initial) = self.initial() {
            let chunk_len = initial.len();
            f(initial, local_idx..local_idx + chunk_len);
            local_idx += chunk_len;
        }

        let packed_slice = self.packed;
        let elems_per_chunk = self.elems_per_chunk();
        for i in self.full_chunks_range() {
            let chunk = &packed_slice[i * elems_per_chunk..][..elems_per_chunk];
            let values = unpack_chunk(&self.strategy, self.bit_width, chunk, self.scratch);
            f(values, local_idx..local_idx + CHUNK_SIZE);
            local_idx += CHUNK_SIZE;
        }

        if let Some(trailer) = self.trailer() {
            let chunk_len = trailer.len();
            f(trailer, local_idx..local_idx + chunk_len);
            local_idx += chunk_len;
        }

        debug_assert_eq!(local_idx, self.len);
    }

    /// Unpack full chunks into output range starting at the given index.
    fn decode_full_chunks_into_at(
        &mut self,
        output: &mut [MaybeUninit<T>],
        start_idx: usize,
    ) -> usize {
        let mut local_idx = start_idx;

        let packed_slice = self.packed;
        let elems_per_chunk = self.elems_per_chunk();
        for i in self.full_chunks_range() {
            let chunk = &packed_slice[i * elems_per_chunk..][..elems_per_chunk];

            let dst = &mut output[local_idx..local_idx + CHUNK_SIZE];
            unpack_chunk(&self.strategy, self.bit_width, chunk, dst);
            local_idx += CHUNK_SIZE;
        }
        local_idx
    }

    fn full_chunks_range(&self) -> Range<usize> {
        (self.first_chunk_is_sliced() as usize)
            ..(self.num_chunks - self.last_chunk_is_sliced() as usize)
    }

    /// Access last chunk of the array if the last chunk has fewer than 1024 due to slicing
    pub fn trailer(&mut self) -> Option<&mut [T]> {
        (self.last_chunk_is_sliced() && self.num_chunks > 1).then(|| {
            let chunk: &[T::Physical] = &self.packed
                [(self.num_chunks - 1) * self.elems_per_chunk()..][..self.elems_per_chunk()];
            let values = unpack_chunk(&self.strategy, self.bit_width, chunk, self.scratch);
            &mut values[..self.last_chunk_length]
        })
    }

    fn last_chunk_is_sliced(&self) -> bool {
        self.last_chunk_length != 0
    }

    fn first_chunk_is_sliced(&self) -> bool {
        self.offset != 0
    }
}

/// Walk every packed chunk in array order without allocating an unpack scratch buffer.
pub(crate) fn for_each_packed_chunk<T, F>(
    packed: &[T::Physical],
    bit_width: usize,
    offset: usize,
    len: usize,
    mut f: F,
) -> VortexResult<()>
where
    T: PhysicalPType,
    F: FnMut(&[T::Physical], Range<usize>),
{
    let (num_chunks, _) = validate_packed::<T>(packed.len(), bit_width, offset, len)?;
    let elems_per_chunk = 128 * bit_width / size_of::<T>();
    let padded_len = offset + len;
    for chunk in 0..num_chunks {
        let packed_chunk = &packed[chunk * elems_per_chunk..][..elems_per_chunk];
        let start = chunk * CHUNK_SIZE;
        let end = (start + CHUNK_SIZE).min(padded_len);
        f(packed_chunk, start..end);
    }
    Ok(())
}

/// The number of chunks `packed` holds for `len` values from `offset`, and the length of its last
/// chunk, or 0 if that is full. Errors if `packed` does not hold exactly those chunks.
pub(crate) fn validate_packed<T: PhysicalPType>(
    packed_len: usize,
    bit_width: usize,
    offset: usize,
    len: usize,
) -> VortexResult<(usize, usize)> {
    vortex_ensure!(
        offset < CHUNK_SIZE,
        "Invalid bit-packed offset {offset}, expected < {CHUNK_SIZE}"
    );
    let elems_per_chunk = 128 * bit_width / size_of::<T>();
    let num_chunks = (offset + len).div_ceil(CHUNK_SIZE);
    vortex_ensure_eq!(packed_len, num_chunks * elems_per_chunk);
    Ok((num_chunks, (offset + len) % CHUNK_SIZE))
}

/// Iterator over full chunks of bitpacked array that yields unpacked chunks one at a time
pub struct BitUnpackIterator<'a, T: BitPacked + 'a> {
    packed: &'a [T::Physical],
    buffer: &'a mut [MaybeUninit<T>; CHUNK_SIZE],
    bit_width: usize,
    elems_per_chunk: usize,
    num_chunks: usize,
    idx: usize,
}

impl<'a, T: BitPacked> BitUnpackIterator<'a, T> {
    pub fn new(
        packed: &'a [T::Physical],
        buffer: &'a mut [MaybeUninit<T>; CHUNK_SIZE],
        bit_width: usize,
        elems_per_chunk: usize,
        num_chunks: usize,
        first_chunk_is_sliced: bool,
    ) -> Self {
        Self {
            packed,
            buffer,
            bit_width,
            elems_per_chunk,
            num_chunks,
            idx: if first_chunk_is_sliced { 1 } else { 0 },
        }
    }

    /// Unpack the next full chunk into the scratch buffer and return it.
    ///
    /// The returned slice borrows the iterator, so it must be dropped before the next call.
    pub fn next_chunk(&mut self) -> Option<&mut [T; CHUNK_SIZE]> {
        if self.idx >= self.num_chunks {
            return None;
        }

        let chunk = &self.packed[self.idx * self.elems_per_chunk..][..self.elems_per_chunk];

        // SAFETY: the packed slice holds one full chunk, and the buffer has CHUNK_SIZE slots.
        unsafe {
            BitPacking::unchecked_unpack_uninit(self.bit_width, chunk, physical_output(self.buffer))
        };
        self.idx += 1;
        // SAFETY: the unpack initialized every slot in the array.
        unsafe { self.buffer.assume_init_mut() }.first_chunk_mut::<CHUNK_SIZE>()
    }
}

fn write_map<T: Copy, U>(src: &[T], dst: &mut [MaybeUninit<U>], f: &mut impl FnMut(T) -> U) {
    for (dst, &src) in dst.iter_mut().zip(src.iter()) {
        dst.write(f(src));
    }
}

pub trait BitPacked: PhysicalPType<Physical: BitPacking> {}

impl BitPacked for i8 {}
impl BitPacked for i16 {}
impl BitPacked for i32 {}
impl BitPacked for i64 {}
impl BitPacked for u8 {}
impl BitPacked for u16 {}
impl BitPacked for u32 {}
impl BitPacked for u64 {}

#[cfg(test)]
mod tests {
    use std::mem::MaybeUninit;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_error::VortexResult;

    use super::*;
    use crate::BitPackedArrayExt;
    use crate::BitPackedData;
    use crate::bitpacking::array::bitpack_compress::bitpack_primitive;
    use crate::test::SESSION;

    #[rstest]
    #[case::empty(0, 0, None, 0, None)]
    #[case::empty_with_offset(10, 0, None, 0, None)]
    #[case::single_partial(0, 500, Some(500), 0, None)]
    #[case::single_full_chunk(0, 1024, None, 1, None)]
    #[case::single_chunk_offset_partial(100, 500, Some(500), 0, None)]
    #[case::single_chunk_offset_to_end(100, 924, Some(924), 0, None)]
    #[case::two_full_chunks(0, 2048, None, 2, None)]
    #[case::full_and_trailer(0, 1500, None, 1, Some(476))]
    #[case::header_and_trailer(100, 1000, Some(924), 0, Some(76))]
    #[case::header_full_and_trailer(100, 2048, Some(924), 1, Some(100))]
    fn test_unpacked_chunks_partitions_without_overlap(
        #[case] offset: usize,
        #[case] len: usize,
        #[case] expected_initial: Option<usize>,
        #[case] expected_full_chunks: usize,
        #[case] expected_trailer: Option<usize>,
    ) -> VortexResult<()> {
        let total_chunks = (offset + len).div_ceil(CHUNK_SIZE);
        let source: Vec<u32> = (0..total_chunks * CHUNK_SIZE).map(|i| i as u32).collect();
        let packed = bitpack_primitive(&source, 12);
        let expected = &source[offset..offset + len];

        let mut scratch = [const { MaybeUninit::<u32>::uninit() }; CHUNK_SIZE];
        let mut chunks = BitUnpackedChunks::try_new_with_strategy(
            BitPackingStrategy,
            packed.as_slice(),
            12,
            offset,
            len,
            &mut scratch,
        )?;

        let mut collected = Vec::with_capacity(len);
        let initial_len = chunks.initial().map(|slice| {
            collected.extend_from_slice(slice);
            slice.len()
        });
        assert_eq!(initial_len, expected_initial);

        let mut full_count = 0;
        {
            let mut full = chunks.full_chunks();
            while let Some(chunk) = full.next_chunk() {
                collected.extend_from_slice(chunk);
                full_count += 1;
            }
        }
        assert_eq!(full_count, expected_full_chunks);

        let trailer_len = chunks.trailer().map(|slice| {
            collected.extend_from_slice(slice);
            slice.len()
        });
        assert_eq!(trailer_len, expected_trailer);
        assert_eq!(collected.as_slice(), expected);

        // Verify `for_each_unpacked_chunk` and `decode_into` produce the same elements.
        let mut visited = Vec::with_capacity(len);
        chunks.for_each_unpacked_chunk(|slice, range| {
            assert_eq!(range, visited.len()..visited.len() + slice.len());
            visited.extend_from_slice(slice);
        });
        assert_eq!(visited.as_slice(), expected);

        let mut decoded = vec![MaybeUninit::<u32>::uninit(); len];
        chunks.decode_into(&mut decoded);
        // SAFETY: `decode_into` initialized all `len` elements.
        let decoded: Vec<u32> = decoded
            .into_iter()
            .map(|v| unsafe { v.assume_init() })
            .collect();
        assert_eq!(decoded.as_slice(), expected);

        Ok(())
    }

    #[test]
    fn test_single_full_chunk_array_unpacked_chunks() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values: Vec<u32> = (0..1024).map(|i| i % 4).collect();
        let array = BitPackedData::encode(
            &PrimitiveArray::from_iter(values.clone()).into_array(),
            2,
            &mut ctx,
        )?;
        let mut scratch = [const { MaybeUninit::<u32>::uninit() }; CHUNK_SIZE];
        let mut chunks = array.unpacked_chunks::<u32>(&mut scratch)?;

        assert!(chunks.initial().is_none());
        let mut full = chunks.full_chunks();
        assert_eq!(
            full.next_chunk().map(|c| c.as_slice()),
            Some(values.as_slice())
        );
        assert!(full.next_chunk().is_none());
        assert!(chunks.trailer().is_none());
        Ok(())
    }
}
