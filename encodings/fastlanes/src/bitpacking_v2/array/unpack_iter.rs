// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem;
use std::mem::MaybeUninit;
use std::ops::Range;

use fastlanes::BitPacking;
use lending_iterator::gat;
use lending_iterator::prelude::Item;
#[gat(Item)]
use lending_iterator::prelude::LendingIterator;
use vortex_array::dtype::PhysicalPType;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::BitPackedV2Data;
use crate::FL_CHUNK_SIZE;
use crate::bitpacking_v2::array::ChunkWidths;
use crate::bitpacking_v2::array::chunk_packed_bytes;

const CHUNK_SIZE: usize = FL_CHUNK_SIZE;

pub use crate::unpack_iter::BitPacked as BitPackedV2;
pub use crate::unpack_iter::BitPackingStrategy;
pub use crate::unpack_iter::UnpackStrategy;

/// The packed FastLanes block of `chunk` and its bit width.
#[allow(clippy::inline_always)]
#[inline(always)]
fn packed_chunk<'a, P>(packed: &'a [P], widths: &ChunkWidths, chunk: usize) -> (&'a [P], usize) {
    let bit_width = widths.width(chunk);
    let start = widths.byte_offset(chunk) / size_of::<P>();
    let len = chunk_packed_bytes(bit_width) / size_of::<P>();
    (&packed[start..][..len], bit_width as usize)
}

/// Accessor to unpacked chunks of bitpacked arrays
///
/// The usual pattern of usage should follow
/// ```
/// use std::mem::MaybeUninit;
///
/// use lending_iterator::gat;
/// use lending_iterator::prelude::Item;
/// #[gat(Item)]
/// use lending_iterator::prelude::LendingIterator;
/// use vortex_array::IntoArray;
/// use vortex_array::VortexSessionExecute;
/// use vortex_buffer::buffer;
/// use vortex_fastlanes::BitPackedV2Data;
/// use vortex_fastlanes::BitPackedV2ArrayExt;
/// use vortex_fastlanes::FL_CHUNK_SIZE;
/// use vortex_fastlanes::bitpacking_v2::unpack_iter::BitUnpackedChunks;
///
/// let mut ctx = vortex_array::array_session().create_execution_ctx();
/// let array = BitPackedV2Data::encode(&buffer![2, 3, 4, 5].into_array(), 2, &mut ctx).unwrap();
/// let mut scratch = [const { MaybeUninit::<i32>::uninit() }; FL_CHUNK_SIZE];
/// let mut unpacked_chunks: BitUnpackedChunks<i32> = array.unpacked_chunks(&mut scratch).unwrap();
///
/// if let Some(header) = unpacked_chunks.initial() {
///    // handle partial initial chunk
/// }
///
/// let mut chunks_iter = unpacked_chunks.full_chunks();
/// while let Some(chunk) = chunks_iter.next() {
///     // handle full bitpacked chunks of 1024 elements
/// }
///
/// if let Some(trailer) = unpacked_chunks.trailer() {
///     // handle partial trailing chunk
/// }
/// ```
pub struct UnpackedChunks<'a, T: PhysicalPType, S: UnpackStrategy<T>> {
    strategy: S,
    widths: &'a ChunkWidths,
    offset: usize,
    len: usize,
    num_chunks: usize,
    // 0 indicates full chunk of CHUNK_SIZE
    last_chunk_length: usize,
    packed: &'a [T::Physical],
    scratch: &'a mut [MaybeUninit<T>; CHUNK_SIZE],
}

pub type BitUnpackedChunks<'a, T> = UnpackedChunks<'a, T, BitPackingStrategy>;

impl<'a, T: BitPackedV2> BitUnpackedChunks<'a, T> {
    pub fn try_new(
        array: &'a BitPackedV2Data,
        len: usize,
        scratch: &'a mut [MaybeUninit<T>; CHUNK_SIZE],
    ) -> VortexResult<Self> {
        Self::try_new_with_strategy(
            BitPackingStrategy,
            array.packed_slice::<T::Physical>(),
            array.chunk_widths(),
            array.offset() as usize,
            len,
            scratch,
        )
    }

    pub fn full_chunks(&mut self) -> BitUnpackIterator<'_, T> {
        let last_chunk_is_sliced = self.last_chunk_is_sliced() as usize;
        let first_chunk_is_sliced = self.first_chunk_is_sliced();
        BitUnpackIterator::new(
            self.packed,
            self.widths,
            self.scratch,
            self.num_chunks - last_chunk_is_sliced,
            first_chunk_is_sliced,
        )
    }
}

impl<'a, T: PhysicalPType, S: UnpackStrategy<T>> UnpackedChunks<'a, T, S> {
    pub fn try_new_with_strategy(
        strategy: S,
        packed: &'a [T::Physical],
        widths: &'a ChunkWidths,
        offset: usize,
        len: usize,
        scratch: &'a mut [MaybeUninit<T>; CHUNK_SIZE],
    ) -> VortexResult<Self> {
        let (num_chunks, last_chunk_length) =
            validate_packed::<T>(packed.len(), widths, offset, len)?;
        Ok(Self {
            strategy,
            widths,
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
    fn chunk(&self, chunk: usize) -> (&'a [T::Physical], usize) {
        packed_chunk(self.packed, self.widths, chunk)
    }

    /// Access first chunk of the array if the last chunk has fewer than 1024 due to slicing
    pub fn initial(&mut self) -> Option<&mut [T]> {
        (self.first_chunk_is_sliced() || self.num_chunks == 1).then(|| {
            let (chunk, bit_width) = self.chunk(0);
            let dst: &mut [MaybeUninit<T>] = self.scratch;
            let dst: &mut [T::Physical] = unsafe { mem::transmute(dst) };

            let header_end_slice = if self.num_chunks == 1 {
                self.len
            } else {
                CHUNK_SIZE - self.offset
            };
            // SAFETY:
            // 1. chunk holds exactly one packed block at bit_width.
            // 2. buffer is exactly CHUNK_SIZE.
            unsafe {
                self.strategy.unpack_chunk(bit_width, chunk, dst);
                mem::transmute(&mut self.scratch[self.offset..][..header_end_slice])
            }
        })
    }

    /// Decode all chunks (initial, full, and trailer) directly into the output range.
    pub fn decode_into(&mut self, output: &mut [MaybeUninit<T>]) {
        debug_assert_eq!(output.len(), self.len);
        let mut local_idx = 0;

        if let Some(initial) = self.initial() {
            local_idx = initial.len();

            // TODO(connor): use maybe_uninit_write_slice when it gets stabilized.
            // SAFETY: &[T] and &[MaybeUninit<T>] have the same layout.
            let init_initial: &[MaybeUninit<T>] = unsafe { mem::transmute(initial) };
            output[..local_idx].copy_from_slice(init_initial);
        }

        local_idx = self.decode_full_chunks_into_at(output, local_idx);

        if let Some(trailer) = self.trailer() {
            // TODO(connor): use maybe_uninit_write_slice when it gets stabilized.
            // SAFETY: &[T] and &[MaybeUninit<T>] have the same layout.
            let init_trailer: &[MaybeUninit<T>] = unsafe { mem::transmute(trailer) };
            output[local_idx..][..init_trailer.len()].copy_from_slice(init_trailer);
            local_idx += init_trailer.len();
        }

        debug_assert_eq!(local_idx, self.len);
    }

    /// Unpack full chunks into output range starting at the given index.
    fn decode_full_chunks_into_at(
        &mut self,
        output: &mut [MaybeUninit<T>],
        start_idx: usize,
    ) -> usize {
        if self.num_chunks == 1 {
            return start_idx;
        }

        let mut local_idx = start_idx;

        let range = self.full_chunks_range();
        let mut start = self.widths.byte_offset(range.start) / size_of::<T::Physical>();
        for &bit_width in &self.widths.as_slice()[range] {
            let len = chunk_packed_bytes(bit_width) / size_of::<T::Physical>();
            let chunk = &self.packed[start..start + len];
            start += len;

            unsafe {
                let uninit_dst = &mut output[local_idx..local_idx + CHUNK_SIZE];
                // SAFETY: &[T] and &[MaybeUninit<T>] have the same layout.
                let dst: &mut [T::Physical] = mem::transmute(uninit_dst);
                self.strategy.unpack_chunk(bit_width as usize, chunk, dst);
            }
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
            let (chunk, bit_width) = self.chunk(self.num_chunks - 1);
            let dst: &mut [MaybeUninit<T>] = self.scratch;
            let dst: &mut [T::Physical] = unsafe { mem::transmute(dst) };
            // SAFETY:
            // 1. chunk holds exactly one packed block at bit_width.
            // 2. buffer is exactly CHUNK_SIZE.
            unsafe {
                self.strategy.unpack_chunk(bit_width, chunk, dst);
                mem::transmute(&mut self.scratch[..self.last_chunk_length])
            }
        })
    }

    fn last_chunk_is_sliced(&self) -> bool {
        self.last_chunk_length != 0
    }

    fn first_chunk_is_sliced(&self) -> bool {
        self.offset != 0
    }
}

/// Check that `packed_len` words of `T::Physical` hold exactly the chunks described by `widths`
/// for `offset + len` padded elements, returning the chunk count and the trailing chunk's length.
fn validate_packed<T: PhysicalPType>(
    packed_len: usize,
    widths: &ChunkWidths,
    offset: usize,
    len: usize,
) -> VortexResult<(usize, usize)> {
    vortex_ensure!(
        offset < CHUNK_SIZE,
        "Invalid bit-packed offset {offset}, expected < {CHUNK_SIZE}"
    );
    let num_chunks = (offset + len).div_ceil(CHUNK_SIZE);
    vortex_ensure!(
        widths.len() == num_chunks,
        "Invalid chunk widths: got {}, expected {num_chunks}",
        widths.len()
    );
    let expected = widths.packed_bytes() / size_of::<T::Physical>();
    vortex_ensure!(
        packed_len == expected,
        "Invalid packed length: got {packed_len}, expected {expected}"
    );
    Ok((num_chunks, (offset + len) % CHUNK_SIZE))
}

/// Iterator over full chunks of bitpacked array that yields unpacked chunks one at a time
pub struct BitUnpackIterator<'a, T: BitPackedV2 + 'a> {
    packed: &'a [T::Physical],
    widths: &'a ChunkWidths,
    buffer: &'a mut [MaybeUninit<T>; CHUNK_SIZE],
    num_chunks: usize,
    idx: usize,
    /// Word offset of chunk `idx` within `packed`.
    start: usize,
}

impl<'a, T: BitPackedV2> BitUnpackIterator<'a, T> {
    pub fn new(
        packed: &'a [T::Physical],
        widths: &'a ChunkWidths,
        buffer: &'a mut [MaybeUninit<T>; CHUNK_SIZE],
        num_chunks: usize,
        first_chunk_is_sliced: bool,
    ) -> Self {
        let idx = if first_chunk_is_sliced { 1 } else { 0 };
        Self {
            packed,
            widths,
            buffer,
            num_chunks,
            idx,
            start: widths.byte_offset(idx) / size_of::<T::Physical>(),
        }
    }
}

#[gat]
impl<'a, T: BitPackedV2 + 'a> LendingIterator for BitUnpackIterator<'a, T> {
    type Item<'next>
    where
        Self: 'next,
    = &'next mut [T; CHUNK_SIZE];

    fn next(&'_ mut self) -> Option<Item<'_, Self>> {
        if self.idx >= self.num_chunks {
            return None;
        }

        let bit_width = self.widths.width(self.idx);
        let len = chunk_packed_bytes(bit_width) / size_of::<T::Physical>();
        let chunk = &self.packed[self.start..self.start + len];

        let dst: &mut [MaybeUninit<T>] = self.buffer;
        unsafe {
            let dst: &mut [T::Physical] = mem::transmute(dst);

            BitPacking::unchecked_unpack(bit_width as usize, chunk, dst);
        }
        self.idx += 1;
        self.start += len;
        // SAFETY: The buffer has the appropriate lifetime, the iterator signature doesn't account for it
        Some(unsafe { mem::transmute::<&mut [MaybeUninit<T>; 1024], &mut [T; 1024]>(self.buffer) })
    }
}
