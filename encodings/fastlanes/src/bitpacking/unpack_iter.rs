use std::mem;
use std::mem::MaybeUninit;

use fastlanes::BitPacking;
use lending_iterator::gat;
use lending_iterator::prelude::Item;
#[gat(Item)]
use lending_iterator::prelude::LendingIterator;
use vortex_array::Array;
use vortex_array::builders::UninitRange;
use vortex_buffer::ByteBuffer;
use vortex_dtype::NativePType;

use crate::BitPackedArray;

const CHUNK_SIZE: usize = 1024;

/// Accessor to unpacked chunks of bitpacked arrays
///
/// The usual pattern of usage should follow
/// ```
/// use lending_iterator::gat;
/// use lending_iterator::prelude::Item;
/// #[gat(Item)]
/// use lending_iterator::prelude::LendingIterator;
/// use vortex_array::IntoArray;
/// use vortex_buffer::buffer;
/// use vortex_fastlanes::BitPackedArray;
///
/// let array = BitPackedArray::encode(&buffer![2, 3, 4, 5].into_array(), 2).unwrap();
/// let mut unpacked_chunks = array.unpacked_chunks();
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
///
pub struct BitUnpackedChunks<T: BitPacked> {
    bit_width: usize,
    offset: usize,
    len: usize,
    num_chunks: usize,
    // 0 indicates full chunk of CHUNK_SIZE
    last_chunk_length: usize,
    packed: ByteBuffer,
    buffer: [MaybeUninit<T>; CHUNK_SIZE],
}

impl<T: BitPacked> BitUnpackedChunks<T> {
    pub fn new(array: &BitPackedArray) -> Self {
        let offset = array.offset() as usize;
        let len = array.len();
        let bit_width = array.bit_width() as usize;
        let elems_per_chunk = 128 * bit_width / size_of::<T>();
        let num_chunks = (offset + len).div_ceil(CHUNK_SIZE);

        assert_eq!(
            array.packed().len() / size_of::<T>(),
            num_chunks * elems_per_chunk,
            "Invalid packed length: got {}, expected {}",
            array.packed().len() / size_of::<T>(),
            num_chunks * elems_per_chunk
        );

        let last_chunk_length = (offset + len) % CHUNK_SIZE;
        Self {
            bit_width,
            offset,
            len,
            packed: array.packed().clone(),
            buffer: [const { MaybeUninit::<T>::uninit() }; CHUNK_SIZE],
            num_chunks,
            last_chunk_length,
        }
    }

    #[inline(always)]
    const fn elems_per_chunk(&self) -> usize {
        128 * self.bit_width / size_of::<T>()
    }

    /// Access first chunk of the array if the last chunk has fewer than 1024 due to slicing
    pub fn initial(&mut self) -> Option<&mut [T]> {
        (self.first_chunk_is_sliced() || self.num_chunks == 1).then(|| {
            let chunk: &[T::UnsignedT] = &buffer_as_slice(&self.packed)[..self.elems_per_chunk()];
            let dst: &mut [MaybeUninit<T>] = &mut self.buffer;
            let dst: &mut [T::UnsignedT] = unsafe { mem::transmute(dst) };

            let header_end_slice = if self.num_chunks == 1 {
                self.len
            } else {
                CHUNK_SIZE - self.offset
            };
            // SAFETY:
            // 1. chunk is elems_per_chunk.
            // 2. buffer is exactly CHUNK_SIZE.
            unsafe {
                BitPacking::unchecked_unpack(self.bit_width, chunk, dst);
                mem::transmute(&mut self.buffer[self.offset..][..header_end_slice])
            }
        })
    }

    /// Iterator over complete chunks of this array
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

    /// Unpack full chunks into output range and return the last index that was written to
    pub fn decode_full_chunks_into(&mut self, output: &mut UninitRange<T>) -> usize {
        let first_chunk_is_sliced = self.first_chunk_is_sliced();
        // If there's only one chunk and that chunk is sliced it has been handled already by `header` method
        if first_chunk_is_sliced && self.num_chunks == 1 {
            return self.len;
        }

        let last_chunk_is_sliced = self.last_chunk_is_sliced();
        let full_chunks_range =
            (first_chunk_is_sliced as usize)..(self.num_chunks - last_chunk_is_sliced as usize);

        let mut out_idx = if first_chunk_is_sliced {
            CHUNK_SIZE - self.offset
        } else {
            0
        };

        let packed_slice: &[T::UnsignedT] = buffer_as_slice(&self.packed);
        let elems_per_chunk = self.elems_per_chunk();
        for i in full_chunks_range {
            let chunk = &packed_slice[i * elems_per_chunk..][..elems_per_chunk];

            unsafe {
                // SAFETY: &[T] and &[MaybeUninit<T>] have the same layout
                let dst: &mut [T::UnsignedT] = mem::transmute(&mut output[out_idx..][..CHUNK_SIZE]);
                BitPacking::unchecked_unpack(self.bit_width, chunk, dst);
            }
            out_idx += CHUNK_SIZE;
        }
        out_idx
    }

    /// Access last chunk of the array if the last chunk has fewer than 1024 due to slicing
    pub fn trailer(&mut self) -> Option<&mut [T]> {
        (self.last_chunk_is_sliced() && self.num_chunks > 1).then(|| {
            let chunk: &[T::UnsignedT] = &buffer_as_slice(&self.packed)
                [(self.num_chunks - 1) * self.elems_per_chunk()..][..self.elems_per_chunk()];
            let dst: &mut [MaybeUninit<T>] = &mut self.buffer;
            let dst: &mut [T::UnsignedT] = unsafe { mem::transmute(dst) };
            // SAFETY:
            // 1. chunk is elems_per_chunk.
            // 2. buffer is exactly CHUNK_SIZE.
            unsafe {
                BitPacking::unchecked_unpack(self.bit_width, chunk, dst);
                mem::transmute(&mut self.buffer[..self.last_chunk_length])
            }
        })
    }

    #[inline]
    fn last_chunk_is_sliced(&self) -> bool {
        self.last_chunk_length != 0
    }

    #[inline]
    fn first_chunk_is_sliced(&self) -> bool {
        self.offset != 0
    }
}

/// Iterator over full chunks of bitpacked array that yields unpacked chunks one at a time
pub struct BitUnpackIterator<'a, T: BitPacked + 'a> {
    packed: &'a [T::UnsignedT],
    buffer: &'a mut [MaybeUninit<T>; CHUNK_SIZE],
    bit_width: usize,
    elems_per_chunk: usize,
    num_chunks: usize,
    idx: usize,
}

impl<'a, T: BitPacked> BitUnpackIterator<'a, T> {
    pub fn new(
        packed: &'a [T::UnsignedT],
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
            let dst: &mut [T::UnsignedT] = mem::transmute(dst);

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

pub trait BitPacked: NativePType {
    type UnsignedT: NativePType + BitPacking;
}

macro_rules! impl_bit_packed {
    ($T:ty, $Unsigned:ty) => {
        impl BitPacked for $T {
            type UnsignedT = $Unsigned;
        }
    };
}

impl_bit_packed!(i8, u8);
impl_bit_packed!(i16, u16);
impl_bit_packed!(i32, u32);
impl_bit_packed!(i64, u64);
impl_bit_packed!(u8, u8);
impl_bit_packed!(u16, u16);
impl_bit_packed!(u32, u32);
impl_bit_packed!(u64, u64);
