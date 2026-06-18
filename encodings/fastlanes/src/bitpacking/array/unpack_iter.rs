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
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::BitPackedData;

const CHUNK_SIZE: usize = 1024;

/// Strategy trait for fastlanes unpacking operations
pub trait UnpackStrategy<T: PhysicalPType> {
    /// Unpack a chunk of packed data into the destination buffer
    ///
    /// # Safety
    /// - `chunk` must contain exactly `elems_per_chunk` elements
    /// - `dst` must have exactly CHUNK_SIZE capacity
    unsafe fn unpack_chunk(&self, bit_width: usize, chunk: &[T::Physical], dst: &mut [T::Physical]);
}

/// BitPacking strategy - uses plain bitpacking without reference value
pub struct BitPackingStrategy;

impl<T: PhysicalPType<Physical: BitPacking>> UnpackStrategy<T> for BitPackingStrategy {
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
}

/// Accessor to unpacked chunks of bitpacked arrays
///
/// The usual pattern of usage should follow
/// ```
/// use lending_iterator::gat;
/// use lending_iterator::prelude::Item;
/// #[gat(Item)]
/// use lending_iterator::prelude::LendingIterator;
/// use vortex_array::IntoArray;
/// use vortex_array::VortexSessionExecute;
/// use vortex_buffer::buffer;
/// use vortex_fastlanes::BitPackedData;
/// use vortex_fastlanes::BitPackedArrayExt;
/// use vortex_fastlanes::unpack_iter::BitUnpackedChunks;
///
/// let mut ctx = vortex_array::LEGACY_SESSION.create_execution_ctx();
/// let array = BitPackedData::encode(&buffer![2, 3, 4, 5].into_array(), 2, &mut ctx).unwrap();
/// let mut unpacked_chunks: BitUnpackedChunks<i32> = array.unpacked_chunks().unwrap();
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
pub struct UnpackedChunks<T: PhysicalPType, S: UnpackStrategy<T>> {
    strategy: S,
    bit_width: usize,
    offset: usize,
    len: usize,
    num_chunks: usize,
    // 0 indicates full chunk of CHUNK_SIZE
    last_chunk_length: usize,
    packed: ByteBuffer,
    buffer: [MaybeUninit<T>; CHUNK_SIZE],
}

pub type BitUnpackedChunks<T> = UnpackedChunks<T, BitPackingStrategy>;

impl<T: BitPacked> BitUnpackedChunks<T> {
    pub fn try_new(array: &BitPackedData, len: usize) -> VortexResult<Self> {
        Self::try_new_with_strategy(
            BitPackingStrategy,
            array.packed().clone().unwrap_host(),
            array.bit_width() as usize,
            array.offset() as usize,
            len,
        )
    }

    pub fn full_chunks(&mut self) -> BitUnpackIterator<'_, T> {
        let elems_per_chunk = self.elems_per_chunk();
        let last_chunk_is_sliced = self.last_chunk_is_sliced() as usize;
        let first_chunk_is_sliced = self.first_chunk_is_sliced();
        BitUnpackIterator::new(
            buffer_as_slice(&self.packed),
            &mut self.buffer,
            self.bit_width,
            elems_per_chunk,
            self.num_chunks - last_chunk_is_sliced,
            first_chunk_is_sliced,
        )
    }
}

impl<T: PhysicalPType, S: UnpackStrategy<T>> UnpackedChunks<T, S> {
    pub fn try_new_with_strategy(
        strategy: S,
        packed: ByteBuffer,
        bit_width: usize,
        offset: usize,
        len: usize,
    ) -> VortexResult<Self> {
        vortex_ensure!(
            offset < CHUNK_SIZE,
            "Invalid bit-packed offset {offset}, expected < {CHUNK_SIZE}"
        );
        let elems_per_chunk = 128 * bit_width / size_of::<T>();
        let num_chunks = (offset + len).div_ceil(CHUNK_SIZE);

        vortex_ensure!(
            packed.len() / size_of::<T>() == num_chunks * elems_per_chunk,
            "Invalid packed length: got {}, expected {}",
            packed.len() / size_of::<T>(),
            num_chunks * elems_per_chunk
        );

        let last_chunk_length = (offset + len) % CHUNK_SIZE;
        Ok(Self {
            strategy,
            bit_width,
            offset,
            len,
            packed,
            buffer: [const { MaybeUninit::<T>::uninit() }; CHUNK_SIZE],
            num_chunks,
            last_chunk_length,
        })
    }

    #[inline(always)]
    const fn elems_per_chunk(&self) -> usize {
        128 * self.bit_width / size_of::<T>()
    }

    /// Access first chunk of the array if the last chunk has fewer than 1024 due to slicing
    pub fn initial(&mut self) -> Option<&mut [T]> {
        (self.first_chunk_is_sliced() || self.num_chunks == 1).then(|| {
            let chunk: &[T::Physical] = &buffer_as_slice(&self.packed)[..self.elems_per_chunk()];
            let dst: &mut [MaybeUninit<T>] = &mut self.buffer;
            let dst: &mut [T::Physical] = unsafe { mem::transmute(dst) };

            let header_end_slice = if self.num_chunks == 1 {
                self.len
            } else {
                CHUNK_SIZE - self.offset
            };
            // SAFETY:
            // 1. chunk is elems_per_chunk.
            // 2. buffer is exactly CHUNK_SIZE.
            unsafe {
                self.strategy.unpack_chunk(self.bit_width, chunk, dst);
                mem::transmute(&mut self.buffer[self.offset..][..header_end_slice])
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

        if self.num_chunks > 1 {
            let packed_slice: &[T::Physical] = buffer_as_slice(&self.packed);
            let elems_per_chunk = self.elems_per_chunk();
            for i in self.full_chunks_range() {
                let chunk = &packed_slice[i * elems_per_chunk..][..elems_per_chunk];
                unsafe {
                    let dst: &mut [T::Physical] = mem::transmute(&mut self.buffer[..]);
                    self.strategy.unpack_chunk(self.bit_width, chunk, dst);
                    let unpacked: &mut [T] = mem::transmute(&mut self.buffer[..]);
                    f(unpacked, local_idx..local_idx + CHUNK_SIZE);
                }
                local_idx += CHUNK_SIZE;
            }
        }

        if let Some(trailer) = self.trailer() {
            let chunk_len = trailer.len();
            f(trailer, local_idx..local_idx + chunk_len);
            local_idx += chunk_len;
        }

        debug_assert_eq!(local_idx, self.len);
    }

    /// Walk every *packed* chunk in array order, yielding the raw packed FastLanes block and the
    /// padded bit range it covers, without unpacking it.
    ///
    /// Unlike [`Self::for_each_unpacked_chunk`], this does not fill the scratch buffer: it hands
    /// the still-packed block to the callback so fused kernels (e.g. compare) can unpack and
    /// consume it in a single pass. Each yielded block holds exactly `elems_per_chunk` packed
    /// values (the buffer is zero-padded out to a whole final chunk).
    ///
    /// The yielded range is in *padded* coordinates: block `c` covers
    /// `[c * 1024, min((c + 1) * 1024, offset + len))`, so it includes the leading `offset` rows
    /// that slicing skips. Block starts are therefore always 1024-aligned regardless of `offset`.
    /// Callers must account for the array's `offset` when mapping a block's rows back to logical
    /// output positions (e.g. by viewing the output buffer at a bit offset of `offset`).
    pub(crate) fn for_each_packed_chunk<F>(&self, mut f: F)
    where
        F: FnMut(&[T::Physical], Range<usize>),
    {
        let packed_slice: &[T::Physical] = buffer_as_slice(&self.packed);
        let elems_per_chunk = self.elems_per_chunk();
        let padded_len = self.offset + self.len;
        for chunk in 0..self.num_chunks {
            let packed_chunk = &packed_slice[chunk * elems_per_chunk..][..elems_per_chunk];
            let start = chunk * CHUNK_SIZE;
            let end = (start + CHUNK_SIZE).min(padded_len);
            f(packed_chunk, start..end);
        }
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

        let packed_slice: &[T::Physical] = buffer_as_slice(&self.packed);
        let elems_per_chunk = self.elems_per_chunk();
        for i in self.full_chunks_range() {
            let chunk = &packed_slice[i * elems_per_chunk..][..elems_per_chunk];

            unsafe {
                let uninit_dst = &mut output[local_idx..local_idx + CHUNK_SIZE];
                // SAFETY: &[T] and &[MaybeUninit<T>] have the same layout.
                let dst: &mut [T::Physical] = mem::transmute(uninit_dst);
                self.strategy.unpack_chunk(self.bit_width, chunk, dst);
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
            let chunk: &[T::Physical] = &buffer_as_slice(&self.packed)
                [(self.num_chunks - 1) * self.elems_per_chunk()..][..self.elems_per_chunk()];
            let dst: &mut [MaybeUninit<T>] = &mut self.buffer;
            let dst: &mut [T::Physical] = unsafe { mem::transmute(dst) };
            // SAFETY:
            // 1. chunk is elems_per_chunk.
            // 2. buffer is exactly CHUNK_SIZE.
            unsafe {
                self.strategy.unpack_chunk(self.bit_width, chunk, dst);
                mem::transmute(&mut self.buffer[..self.last_chunk_length])
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
}

#[gat]
impl<'a, T: BitPacked + 'a> LendingIterator for BitUnpackIterator<'a, T> {
    type Item<'next>
    where
        Self: 'next,
    = &'next mut [T; CHUNK_SIZE];

    fn next(&'_ mut self) -> Option<Item<'_, Self>> {
        if self.idx >= self.num_chunks {
            return None;
        }

        let chunk = &self.packed[self.idx * self.elems_per_chunk..][..self.elems_per_chunk];

        let dst: &mut [MaybeUninit<T>] = self.buffer;
        unsafe {
            let dst: &mut [T::Physical] = mem::transmute(dst);

            BitPacking::unchecked_unpack(self.bit_width, chunk, dst);
        }
        self.idx += 1;
        // SAFETY: The buffer has the appropriate lifetime, the iterator signature doesn't account for it
        Some(unsafe { mem::transmute::<&mut [MaybeUninit<T>; 1024], &mut [T; 1024]>(self.buffer) })
    }
}

fn buffer_as_slice<T>(buffer: &ByteBuffer) -> &[T] {
    let packed_ptr: *const T = buffer.as_ptr().cast();
    // Return number of elements of type `T` packed in the buffer
    let packed_len = buffer.len() / size_of::<T>();

    // SAFETY: as_slice points to buffer memory that outlives the lifetime of `self`.
    //  Unfortunately Rust cannot understand this, so we reconstruct the slice from raw parts
    //  to get it to reinterpret the lifetime.
    unsafe { std::slice::from_raw_parts(packed_ptr, packed_len) }
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
