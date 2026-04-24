// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Provides types that can be used by I/O frameworks to work with byte buffer-shaped data.

use std::ops::Range;

use bytes::Bytes;
use vortex_buffer::Buffer;
use vortex_buffer::ConstByteBuffer;
use vortex_error::VortexExpect;

/// Trait for types that can provide a readonly byte buffer interface to I/O frameworks.
///
/// # Safety
/// The type must support contiguous raw memory access via pointer, such as `Vec` or `[u8]`.
pub unsafe trait IoBuf: Unpin + Send + 'static {
    /// Returns a raw pointer to the vector’s buffer.
    fn read_ptr(&self) -> *const u8;

    /// Number of initialized bytes.
    fn bytes_init(&self) -> usize;

    /// Access the buffer as a byte slice
    fn as_slice(&self) -> &[u8];

    /// Access the buffer as a byte slice with begin and end indices
    #[inline]
    fn slice_owned(self, range: Range<usize>) -> OwnedSlice<Self>
    where
        Self: Sized,
    {
        // Validate range bounds
        assert!(
            range.start <= range.end,
            "Invalid range: start ({}) must be <= end ({})",
            range.start,
            range.end
        );
        assert!(
            range.end <= self.bytes_init(),
            "Range end ({}) exceeds buffer length ({})",
            range.end,
            self.bytes_init()
        );

        OwnedSlice {
            buf: self,
            begin: range.start,
            end: range.end,
        }
    }
}

/// An owned view into a contiguous sequence of bytes.
pub struct OwnedSlice<T> {
    buf: T,
    begin: usize,
    end: usize,
}

impl<T> OwnedSlice<T> {
    /// Unwrap the slice into its underlying type.
    pub fn into_inner(self) -> T {
        self.buf
    }
}

unsafe impl IoBuf for &'static [u8] {
    #[inline]
    fn read_ptr(&self) -> *const u8 {
        self.as_ptr()
    }

    #[inline]
    fn bytes_init(&self) -> usize {
        self.len()
    }

    #[inline]
    fn as_slice(&self) -> &[u8] {
        self
    }
}

unsafe impl<const N: usize> IoBuf for [u8; N] {
    #[inline]
    fn read_ptr(&self) -> *const u8 {
        self.as_ptr()
    }

    #[inline]
    fn bytes_init(&self) -> usize {
        N
    }

    #[inline]
    fn as_slice(&self) -> &[u8] {
        self.as_ref()
    }
}

unsafe impl IoBuf for Vec<u8> {
    #[inline]
    fn read_ptr(&self) -> *const u8 {
        self.as_ptr()
    }

    #[inline]
    fn bytes_init(&self) -> usize {
        self.len()
    }

    #[inline]
    fn as_slice(&self) -> &[u8] {
        self.as_ref()
    }
}

unsafe impl<T: IoBuf> IoBuf for OwnedSlice<T> {
    #[inline]
    fn read_ptr(&self) -> *const u8 {
        debug_assert!(self.begin <= self.end, "Invalid slice bounds");
        debug_assert!(
            self.end <= self.buf.bytes_init(),
            "Slice end exceeds buffer bounds"
        );

        let base_ptr = self.buf.read_ptr();
        debug_assert!(!base_ptr.is_null(), "Base pointer is null");

        // Check for potential pointer overflow in debug builds
        #[cfg(debug_assertions)]
        {
            let max_offset = isize::MAX as usize;
            assert!(
                self.begin <= max_offset,
                "Offset too large for pointer arithmetic"
            );
        }

        unsafe { base_ptr.add(self.begin) }
    }

    #[inline]
    fn bytes_init(&self) -> usize {
        debug_assert!(self.begin <= self.end, "Invalid slice bounds");
        self.end - self.begin
    }

    #[inline]
    fn as_slice(&self) -> &[u8] {
        let ptr = self.read_ptr();
        let len = self.bytes_init();

        debug_assert!(
            !ptr.is_null() || len == 0,
            "Null pointer with non-zero length"
        );

        unsafe { std::slice::from_raw_parts(ptr, len) }
    }
}

unsafe impl IoBuf for Bytes {
    fn read_ptr(&self) -> *const u8 {
        self.as_ptr()
    }

    fn bytes_init(&self) -> usize {
        self.len()
    }

    fn as_slice(&self) -> &[u8] {
        self.as_ref()
    }
}

unsafe impl<const A: usize> IoBuf for ConstByteBuffer<A> {
    fn read_ptr(&self) -> *const u8 {
        self.as_ptr()
    }

    fn bytes_init(&self) -> usize {
        self.len()
    }

    fn as_slice(&self) -> &[u8] {
        self.as_ref()
    }
}

unsafe impl<T: Unpin + Send + 'static> IoBuf for Buffer<T> {
    fn read_ptr(&self) -> *const u8 {
        self.as_ptr().cast()
    }

    fn bytes_init(&self) -> usize {
        self.len()
            .checked_mul(size_of::<T>())
            .vortex_expect("Buffer size calculation overflow")
    }

    fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.read_ptr(), self.bytes_init()) }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[test]
    fn test_static_slice_io_buf() {
        let data: &'static [u8] = b"hello world";

        assert_eq!(data.read_ptr(), data.as_ptr());
        assert_eq!(data.bytes_init(), 11);
        assert_eq!(data.as_slice(), b"hello world");
    }

    #[test]
    fn test_static_empty_slice() {
        let data: &'static [u8] = b"";

        assert_eq!(data.bytes_init(), 0);
        assert_eq!(data.as_slice(), b"");
    }

    #[rstest]
    #[case([1u8, 2, 3, 4, 5], 5)]
    #[case([0u8; 256], 256)]
    #[case([255u8; 1], 1)]
    fn test_array_io_buf<const N: usize>(#[case] array: [u8; N], #[case] expected_len: usize) {
        assert_eq!(array.bytes_init(), expected_len);
        assert_eq!(array.as_slice().len(), expected_len);
        assert_eq!(array.read_ptr(), array.as_ptr());
    }

    #[test]
    fn test_vec_io_buf() {
        let vec = vec![1u8, 2, 3, 4, 5];

        assert_eq!(vec.read_ptr(), vec.as_ptr());
        assert_eq!(vec.bytes_init(), 5);
        assert_eq!(vec.as_slice(), &[1, 2, 3, 4, 5]);
    }

    #[rstest]
    #[case(vec![], 0)]
    #[case(vec![42u8], 1)]
    #[case(vec![1u8, 2, 3], 3)]
    #[case(vec![0u8; 1024], 1024)]
    fn test_vec_various_sizes(#[case] vec: Vec<u8>, #[case] expected_len: usize) {
        assert_eq!(vec.bytes_init(), expected_len);
        assert_eq!(vec.as_slice().len(), expected_len);
    }

    #[test]
    fn test_owned_slice_basic() {
        let data = vec![1u8, 2, 3, 4, 5];
        let slice = data.slice_owned(1..4);

        assert_eq!(slice.bytes_init(), 3);
        assert_eq!(slice.as_slice(), &[2, 3, 4]);
    }

    #[rstest]
    #[case(vec![1u8, 2, 3, 4, 5], 0..5, vec![1, 2, 3, 4, 5])]
    #[case(vec![1u8, 2, 3, 4, 5], 1..4, vec![2, 3, 4])]
    #[case(vec![1u8, 2, 3, 4, 5], 2..3, vec![3])]
    #[case(vec![1u8, 2, 3, 4, 5], 0..0, vec![])]
    #[case(vec![1u8, 2, 3, 4, 5], 5..5, vec![])]
    fn test_owned_slice_ranges(
        #[case] data: Vec<u8>,
        #[case] range: Range<usize>,
        #[case] expected: Vec<u8>,
    ) {
        let slice = data.slice_owned(range.clone());
        assert_eq!(slice.bytes_init(), range.end - range.start);
        assert_eq!(slice.as_slice(), &expected[..]);
    }

    #[test]
    fn test_owned_slice_into_inner() {
        let data = vec![1u8, 2, 3, 4, 5];
        let slice = data.clone().slice_owned(1..4);
        let recovered = slice.into_inner();

        assert_eq!(recovered, data);
    }

    #[test]
    fn test_nested_owned_slice() {
        let data = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        let slice1 = data.slice_owned(1..7); // [2, 3, 4, 5, 6, 7]
        let slice2 = slice1.slice_owned(1..4); // [3, 4, 5]

        assert_eq!(slice2.bytes_init(), 3);
        assert_eq!(slice2.as_slice(), &[3, 4, 5]);
    }

    #[test]
    fn test_bytes_io_buf() {
        let bytes = Bytes::from_static(b"test data");

        assert_eq!(bytes.read_ptr(), bytes.as_ptr());
        assert_eq!(bytes.bytes_init(), 9);
        assert_eq!(bytes.as_slice(), b"test data");
    }

    #[test]
    fn test_bytes_empty() {
        let bytes = Bytes::new();

        assert_eq!(bytes.bytes_init(), 0);
        assert_eq!(bytes.as_slice(), b"");
    }

    #[test]
    fn test_const_byte_buffer() {
        const ALIGNMENT: usize = 64;
        let data = b"aligned data".to_vec();
        let buffer = ConstByteBuffer::<ALIGNMENT>::copy_from(&data);

        assert_eq!(buffer.bytes_init(), 12);
        assert_eq!(buffer.as_slice(), b"aligned data");

        // Verify alignment
        let ptr_addr = buffer.read_ptr() as usize;
        assert_eq!(ptr_addr % ALIGNMENT, 0);
    }

    macro_rules! test_const_buffer_alignment {
        ($name:ident, $alignment:literal) => {
            #[test]
            fn $name() {
                let data = b"test".to_vec();
                let buffer = ConstByteBuffer::<$alignment>::copy_from(&data);
                let ptr_addr = buffer.read_ptr() as usize;
                assert_eq!(ptr_addr % $alignment, 0);
                assert_eq!(buffer.bytes_init(), 4);
            }
        };
    }

    test_const_buffer_alignment!(test_const_byte_buffer_alignment_8, 8);
    test_const_buffer_alignment!(test_const_byte_buffer_alignment_16, 16);
    test_const_buffer_alignment!(test_const_byte_buffer_alignment_32, 32);
    test_const_buffer_alignment!(test_const_byte_buffer_alignment_64, 64);
    test_const_buffer_alignment!(test_const_byte_buffer_alignment_128, 128);
    test_const_buffer_alignment!(test_const_byte_buffer_alignment_256, 256);

    #[test]
    fn test_buffer_u32() {
        let data = vec![1u32, 2, 3, 4];
        let mut buf_mut = vortex_buffer::BufferMut::<u32>::with_capacity(4);
        buf_mut.extend_from_slice(&data);
        let buffer: Buffer<u32> = buf_mut.freeze();

        // The buffer has 4 u32 elements, bytes_init should be 4 * 4 = 16 bytes
        assert_eq!(buffer.len(), 4); // 4 elements
        assert_eq!(buffer.bytes_init(), 16); // 4 * size_of::<u32>()
    }

    #[test]
    fn test_buffer_u64() {
        let data = vec![100u64, 200, 300];
        let mut buf_mut = vortex_buffer::BufferMut::<u64>::with_capacity(3);
        buf_mut.extend_from_slice(&data);
        let buffer: Buffer<u64> = buf_mut.freeze();

        // The buffer has 3 u64 elements, bytes_init should be 3 * 8 = 24 bytes
        assert_eq!(buffer.len(), 3); // 3 elements
        assert_eq!(buffer.bytes_init(), 24); // 3 * size_of::<u64>()
    }

    #[test]
    fn test_buffer_empty() {
        let buffer: Buffer<u8> = Buffer::from(vec![]);

        assert_eq!(buffer.bytes_init(), 0);
        assert_eq!(buffer.as_slice(), &[] as &[u8]);
    }

    #[test]
    fn test_buffer_various_types() {
        // u8 buffer
        let buffer = Buffer::from(vec![1u8, 2, 3]);
        assert_eq!(buffer.bytes_init(), 3);

        // u16 buffer
        let mut buf_mut = vortex_buffer::BufferMut::<u16>::with_capacity(3);
        buf_mut.extend_from_slice(&[1u16, 2, 3]);
        let buffer: Buffer<u16> = buf_mut.freeze();
        assert_eq!(buffer.bytes_init(), 6);

        // u32 buffer
        let mut buf_mut = vortex_buffer::BufferMut::<u32>::with_capacity(3);
        buf_mut.extend_from_slice(&[1u32, 2, 3]);
        let buffer: Buffer<u32> = buf_mut.freeze();
        assert_eq!(buffer.bytes_init(), 12);

        // u64 buffer
        let mut buf_mut = vortex_buffer::BufferMut::<u64>::with_capacity(3);
        buf_mut.extend_from_slice(&[1u64, 2, 3]);
        let buffer: Buffer<u64> = buf_mut.freeze();
        assert_eq!(buffer.bytes_init(), 24);
    }

    #[test]
    fn test_pointer_validity() {
        // Test that read_ptr returns valid pointers for different types
        let vec = vec![1u8, 2, 3];
        let slice: &'static [u8] = &[1, 2, 3];
        let array = [1u8, 2, 3];

        // These should not crash or cause UB
        let _ = vec.read_ptr();
        let _ = slice.read_ptr();
        let _ = array.read_ptr();

        // Verify pointer consistency
        assert_eq!(vec.read_ptr(), vec.as_ptr());
        assert_eq!(slice.read_ptr(), slice.as_ptr());
        assert_eq!(array.read_ptr(), array.as_ptr());
    }

    #[test]
    fn test_slice_owned_preserves_data() {
        let original = vec![10u8, 20, 30, 40, 50];
        let slice = original.clone().slice_owned(1..4);

        // Verify the slice sees the correct data
        assert_eq!(slice.as_slice(), &[20, 30, 40]);

        // Verify we can recover the original
        let recovered = slice.into_inner();
        assert_eq!(recovered, original);
    }

    // Panic tests for bounds checking
    #[test]
    #[should_panic(expected = "Invalid range")]
    fn test_owned_slice_invalid_range() {
        let data = vec![1, 2, 3];
        #[expect(
            clippy::reversed_empty_ranges,
            reason = "intentionally testing invalid range"
        )]
        data.slice_owned(5..3); // start > end
    }

    #[test]
    #[should_panic(expected = "exceeds buffer length")]
    fn test_owned_slice_out_of_bounds() {
        let data = vec![1, 2, 3];
        data.slice_owned(1..10); // end > len
    }

    #[test]
    fn test_owned_slice_zero_sized_at_boundary() {
        let data = vec![1, 2, 3];
        let slice = data.slice_owned(3..3); // Zero-sized at end
        assert_eq!(slice.bytes_init(), 0);
    }

    #[test]
    #[should_panic(expected = "exceeds buffer length")]
    fn test_owned_slice_start_out_of_bounds() {
        let data = vec![1, 2, 3];
        data.slice_owned(10..11); // start > len
    }

    // Buffer overflow protection tests
    #[test]
    fn test_buffer_size_calculation_u8() {
        let buffer: Buffer<u8> = Buffer::from(vec![1, 2, 3]);
        assert_eq!(buffer.bytes_init(), 3);
    }

    #[test]
    fn test_buffer_size_calculation_large_type() {
        use vortex_buffer::BufferMut;

        // Test with a struct containing a large array
        #[repr(C)]
        struct LargeType {
            _data: [u8; 1024],
        }

        let mut buf = BufferMut::<LargeType>::with_capacity(10);
        // Extend with 10 elements
        for _ in 0..10 {
            buf.push(LargeType { _data: [0u8; 1024] });
        }
        let buffer = buf.freeze();

        // This should use checked arithmetic and not overflow
        let size = buffer.bytes_init();
        assert_eq!(size, 10 * 1024);
    }

    #[test]
    fn test_buffer_size_near_max() {
        // Test with a moderately large buffer that won't cause OOM
        let large_size = 1_000_000;
        let buffer: Buffer<u8> = Buffer::from(vec![0u8; large_size]);
        assert_eq!(buffer.bytes_init(), large_size);
    }
}
