use std::mem;

use fastlanes::BitPacking;
use vortex_array::Array;
use vortex_array::builders::UninitRange;
use vortex_buffer::ByteBuffer;
use vortex_dtype::NativePType;

use crate::BitPackedArray;

pub struct BitUnpackedChunks<T: BitPacked> {
    bit_width: usize,
    offset: usize,
    num_chunks: usize,
    // 0 indicates full chunk of 1024
    last_chunk_length: usize,
    packed: ByteBuffer,
    buffer: [T; 1024],
}

impl<T: BitPacked> BitUnpackedChunks<T> {
    pub fn new(array: &BitPackedArray) -> Self {
        let offset = array.offset() as usize;
        let len = array.len();
        let bit_width = array.bit_width() as usize;
        let elems_per_chunk = 128 * bit_width / size_of::<T>();
        let num_chunks = (offset + len).div_ceil(1024);

        assert_eq!(
            array.packed().len() / size_of::<T>(),
            num_chunks * elems_per_chunk,
            "Invalid packed length: got {}, expected {}",
            array.packed().len() / size_of::<T>(),
            num_chunks * elems_per_chunk
        );

        let last_chunk_length = (offset + len) % 1024;
        Self {
            bit_width,
            offset,
            packed: array.packed().clone(),
            buffer: [T::zero(); 1024],
            num_chunks,
            last_chunk_length,
        }
    }

    #[inline(always)]
    const fn elems_per_chunk(&self) -> usize {
        128 * self.bit_width / size_of::<T>()
    }

    pub fn header(&mut self) -> Option<&[T]> {
        self.first_chunk_is_sliced().then(|| {
            let chunk: &[T::UnsignedT] = &buffer_as_slice(&self.packed)[..self.elems_per_chunk()];
            let dst: &mut [T] = &mut self.buffer;
            let dst: &mut [T::UnsignedT] = unsafe { mem::transmute(dst) };

            // SAFETY:
            // 1. chunk is elems_per_chunk.
            // 2. buffer is exactly 1024.
            unsafe { BitPacking::unchecked_unpack(self.bit_width, chunk, dst) };
            &self.buffer[self.offset..]
        })
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

    pub fn decode_full_chunks_into(&mut self, output: &mut UninitRange<T>) -> usize {
        let first_chunk_is_sliced = self.first_chunk_is_sliced();
        let last_chunk_is_sliced = self.last_chunk_is_sliced();
        let full_chunks_range =
            (first_chunk_is_sliced as usize)..(self.num_chunks - last_chunk_is_sliced as usize);

        let elems_per_chunk = self.elems_per_chunk();
        let packed_slice: &[T::UnsignedT] = buffer_as_slice(&self.packed);
        let mut out_idx = if first_chunk_is_sliced {
            1024 - self.offset
        } else {
            0
        };

        for i in full_chunks_range {
            let chunk = &packed_slice[i * elems_per_chunk..][..elems_per_chunk];

            unsafe {
                // SAFETY: &[T] and &[MaybeUninit<T>] have the same layout
                let dst: &mut [T::UnsignedT] = mem::transmute(&mut output[out_idx..][..1024]);
                BitPacking::unchecked_unpack(self.bit_width, chunk, dst);
            }
            out_idx += 1024;
        }
        out_idx
    }

    pub fn trailer(&mut self) -> Option<&[T]> {
        self.last_chunk_is_sliced().then(|| {
            let chunk: &[T::UnsignedT] = &buffer_as_slice(&self.packed)
                [(self.num_chunks - 1) * self.elems_per_chunk()..][..self.elems_per_chunk()];
            // SAFETY:
            // 1. chunk is elems_per_chunk.
            // 2. buffer is exactly 1024.
            let dst: &mut [T] = &mut self.buffer;
            let dst: &mut [T::UnsignedT] = unsafe { mem::transmute(dst) };
            unsafe { BitPacking::unchecked_unpack(self.bit_width, chunk, dst) };
            &self.buffer[..self.last_chunk_length]
        })
    }

    fn last_chunk_is_sliced(&self) -> bool {
        self.last_chunk_length != 0
    }

    fn first_chunk_is_sliced(&self) -> bool {
        self.offset != 0
    }
}

pub struct BitUnpackIterator<'a, T: BitPacked + 'a> {
    packed: &'a [T::UnsignedT],
    buffer: &'a mut [T; 1024],
    bit_width: usize,
    elems_per_chunk: usize,
    num_chunks: usize,
    idx: usize,
}

impl<'a, T: BitPacked> BitUnpackIterator<'a, T> {
    pub fn new(
        packed: &'a [T::UnsignedT],
        buffer: &'a mut [T; 1024],
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

impl<'a, T: BitPacked + 'a> Iterator for BitUnpackIterator<'a, T> {
    type Item = &'a [T; 1024];

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.num_chunks {
            return None;
        }

        let chunk = &self.packed[self.idx * self.elems_per_chunk..][..self.elems_per_chunk];

        unsafe {
            let dst: &mut [T] = self.buffer;
            let dst: &mut [T::UnsignedT] = mem::transmute(dst);

            BitPacking::unchecked_unpack(self.bit_width, chunk, dst);
        }
        self.idx += 1;
        // SAFETY: The buffer has the appropriate lifetime, the iterator signature doesn't account for it
        unsafe { mem::transmute(Some(&self.buffer)) }
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
