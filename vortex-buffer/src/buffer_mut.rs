// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use core::mem::MaybeUninit;
use std::any::type_name;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::io::Write;
use std::marker::PhantomData;
use std::ops::Deref;
use std::ops::DerefMut;
use std::ptr::NonNull;

use bytes::Buf;
use bytes::BufMut;
use bytes::Bytes;
use bytes::buf::UninitSlice;
use vortex_error::VortexExpect;
use vortex_error::vortex_panic;

use crate::Alignment;
use crate::Buffer;
use crate::ByteBufferMut;
use crate::debug::TruncatedDebug;
use crate::pool::POOL;
use crate::pool::PooledAllocation;
use crate::trusted_len::TrustedLen;

/// A mutable buffer that maintains a runtime-defined alignment through resizing operations.
///
/// Backed by a [`PooledAllocation`] from the global buffer pool, which recycles recently freed
/// memory blocks to reduce allocation churn.
pub struct BufferMut<T> {
    pub(crate) alloc: PooledAllocation,
    /// Byte offset into `alloc` where the visible data begins. Nonzero only after
    /// `Buf::advance` on a `ByteBufferMut`.
    pub(crate) byte_offset: usize,
    pub(crate) length: usize,
    pub(crate) alignment: Alignment,
    pub(crate) _marker: PhantomData<T>,
}

impl<T> BufferMut<T> {
    /// Create a new `BufferMut` with the requested alignment and capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_aligned(capacity, Alignment::of::<T>())
    }

    /// Create a new `BufferMut` with the requested alignment and capacity.
    pub fn with_capacity_aligned(capacity: usize, alignment: Alignment) -> Self {
        if !alignment.is_aligned_to(Alignment::of::<T>()) {
            vortex_panic!(
                "Alignment {} must align to the scalar type's alignment {}",
                alignment,
                align_of::<T>()
            );
        }

        let byte_capacity = capacity * size_of::<T>();
        let alloc = POOL.acquire(byte_capacity, *alignment);

        Self {
            alloc,
            byte_offset: 0,
            length: 0,
            alignment,
            _marker: PhantomData,
        }
    }

    /// Create a new zeroed `BufferMut`.
    pub fn zeroed(len: usize) -> Self {
        Self::zeroed_aligned(len, Alignment::of::<T>())
    }

    /// Create a new zeroed `BufferMut`.
    pub fn zeroed_aligned(len: usize, alignment: Alignment) -> Self {
        if !alignment.is_aligned_to(Alignment::of::<T>()) {
            vortex_panic!(
                "Alignment {} must align to the scalar type's alignment {}",
                alignment,
                align_of::<T>()
            );
        }

        let byte_len = len * size_of::<T>();
        let alloc = POOL.acquire(byte_len, *alignment);
        if byte_len > 0 {
            // SAFETY: The allocation is at least `byte_len` bytes. Writing zeros is always safe.
            unsafe { alloc.ptr.as_ptr().write_bytes(0, byte_len) };
        }

        Self {
            alloc,
            byte_offset: 0,
            length: len,
            alignment,
            _marker: PhantomData,
        }
    }

    /// Create a new empty `BufferMut` with the provided alignment.
    pub fn empty() -> Self {
        Self::empty_aligned(Alignment::of::<T>())
    }

    /// Create a new empty `BufferMut` with the provided alignment.
    pub fn empty_aligned(alignment: Alignment) -> Self {
        BufferMut::with_capacity_aligned(0, alignment)
    }

    /// Create a new full `BufferMut` with the given value.
    pub fn full(item: T, len: usize) -> Self
    where
        T: Copy,
    {
        let mut buffer = BufferMut::<T>::with_capacity(len);
        buffer.push_n(item, len);
        buffer
    }

    /// Create a mutable scalar buffer by copying the contents of the slice.
    pub fn copy_from(other: impl AsRef<[T]>) -> Self {
        Self::copy_from_aligned(other, Alignment::of::<T>())
    }

    /// Create a mutable scalar buffer with the alignment by copying the contents of the slice.
    ///
    /// ## Panics
    ///
    /// Panics when the requested alignment isn't itself aligned to type T.
    pub fn copy_from_aligned(other: impl AsRef<[T]>, alignment: Alignment) -> Self {
        if !alignment.is_aligned_to(Alignment::of::<T>()) {
            vortex_panic!("Given alignment is not aligned to type T")
        }
        let other = other.as_ref();
        let mut buffer = Self::with_capacity_aligned(other.len(), alignment);
        buffer.extend_from_slice(other);
        debug_assert_eq!(buffer.alignment(), alignment);
        buffer
    }

    /// Get the alignment of the buffer.
    #[inline(always)]
    pub fn alignment(&self) -> Alignment {
        self.alignment
    }

    /// Returns the length of the buffer.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.length
    }

    /// Returns whether the buffer is empty.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Returns the capacity of the buffer.
    #[inline]
    pub fn capacity(&self) -> usize {
        if size_of::<T>() == 0 {
            usize::MAX
        } else {
            (self.alloc.capacity - self.byte_offset) / size_of::<T>()
        }
    }

    /// Returns a pointer to the start of the data region.
    #[inline(always)]
    fn data_ptr(&self) -> *mut u8 {
        // SAFETY: byte_offset is always within [0, capacity].
        unsafe { self.alloc.ptr.as_ptr().add(self.byte_offset) }
    }

    /// Returns the byte offset past the end of the initialized data.
    #[inline(always)]
    fn data_end_offset(&self) -> usize {
        self.byte_offset + self.length * size_of::<T>()
    }

    /// Returns a slice over the buffer of elements of type T.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        if self.length == 0 {
            return &[];
        }
        // SAFETY: alignment of Buffer is checked on construction, and data_ptr points to
        // `length` initialized elements of type T.
        unsafe { std::slice::from_raw_parts(self.data_ptr().cast(), self.length) }
    }

    /// Returns a slice over the buffer of elements of type T.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        if self.length == 0 {
            return &mut [];
        }
        // SAFETY: alignment of Buffer is checked on construction, and data_ptr points to
        // `length` initialized elements of type T.
        unsafe { std::slice::from_raw_parts_mut(self.data_ptr().cast(), self.length) }
    }

    /// Clear the buffer, retaining any existing capacity.
    #[inline]
    pub fn clear(&mut self) {
        self.byte_offset = 0;
        self.length = 0;
    }

    /// Shortens the buffer, keeping the first `len` bytes and dropping the
    /// rest.
    ///
    /// If `len` is greater than the buffer's current length, this has no
    /// effect.
    ///
    /// Existing underlying capacity is preserved.
    #[inline]
    pub fn truncate(&mut self, len: usize) {
        if len <= self.len() {
            // SAFETY: Shrinking the buffer cannot expose uninitialized bytes.
            unsafe { self.set_len(len) };
        }
    }

    /// Reserves capacity for at least `additional` more elements to be inserted in the buffer.
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        let additional_bytes = additional * size_of::<T>();
        let spare_bytes = self.alloc.capacity - self.data_end_offset();
        if additional_bytes <= spare_bytes {
            return;
        }

        self.reserve_allocate(additional);
    }

    /// A separate function so we can inline the reserve call's fast path. According to `BytesMut`
    /// this has significant performance implications.
    fn reserve_allocate(&mut self, additional: usize) {
        let used_bytes = self.length * size_of::<T>();
        let needed = used_bytes + additional * size_of::<T>();
        // Make sure we at least double in size each time we re-allocate to amortize the cost.
        let new_capacity = needed.max(self.alloc.capacity.saturating_mul(2));

        let new_alloc = POOL.acquire(new_capacity, *self.alignment);
        if used_bytes > 0 {
            // SAFETY: Both pointers are valid and non-overlapping (different allocations).
            // The source has `used_bytes` initialized bytes.
            unsafe {
                std::ptr::copy_nonoverlapping(self.data_ptr(), new_alloc.ptr.as_ptr(), used_bytes);
            }
        }
        // Dropping old alloc returns it to the pool.
        self.alloc = new_alloc;
        self.byte_offset = 0;
    }

    /// Returns the spare capacity of the buffer as a slice of `MaybeUninit<T>`.
    /// Has identical semantics to [`Vec::spare_capacity_mut`].
    ///
    /// The returned slice can be used to fill the buffer with data (e.g. by
    /// reading from a file) before marking the data as initialized using the
    /// [`set_len`] method.
    ///
    /// [`set_len`]: BufferMut::set_len
    /// [`Vec::spare_capacity_mut`]: Vec::spare_capacity_mut
    ///
    /// # Examples
    ///
    /// ```
    /// use vortex_buffer::BufferMut;
    ///
    /// // Allocate vector big enough for 10 elements.
    /// let mut b = BufferMut::<u64>::with_capacity(10);
    ///
    /// // Fill in the first 3 elements.
    /// let uninit = b.spare_capacity_mut();
    /// uninit[0].write(0);
    /// uninit[1].write(1);
    /// uninit[2].write(2);
    ///
    /// // Mark the first 3 elements of the vector as being initialized.
    /// unsafe {
    ///     b.set_len(3);
    /// }
    ///
    /// assert_eq!(b.as_slice(), &[0u64, 1, 2]);
    /// ```
    #[inline]
    pub fn spare_capacity_mut(&mut self) -> &mut [MaybeUninit<T>] {
        let spare_count = self.capacity() - self.length;
        if spare_count == 0 {
            return &mut [];
        }
        let data_end = self.data_end_offset();
        // SAFETY: The pointer is within the allocation and points to spare_count uninitialized
        // elements of type T.
        unsafe {
            std::slice::from_raw_parts_mut(
                self.alloc.ptr.as_ptr().add(data_end) as *mut MaybeUninit<T>,
                spare_count,
            )
        }
    }

    /// Sets the length of the buffer.
    ///
    /// # Safety
    ///
    /// - `new_len` must be less than or equal to [`capacity()`].
    /// - The elements at `old_len..new_len` must be initialized.
    ///
    /// [`capacity()`]: Self::capacity
    #[inline]
    pub unsafe fn set_len(&mut self, len: usize) {
        debug_assert!(len <= self.capacity());
        self.length = len;
    }

    /// Appends a scalar to the buffer.
    #[inline]
    pub fn push(&mut self, value: T) {
        self.reserve(1);
        unsafe { self.push_unchecked(value) }
    }

    /// Appends a scalar to the buffer without checking for sufficient capacity.
    ///
    /// ## Safety
    ///
    /// The caller must ensure there is sufficient capacity in the array.
    #[inline]
    pub unsafe fn push_unchecked(&mut self, item: T) {
        // SAFETY: the caller ensures we have sufficient capacity.
        unsafe {
            let data_end = self.data_end_offset();
            let dst: *mut T = self.alloc.ptr.as_ptr().add(data_end).cast();
            dst.write(item);
        }
        self.length += 1;
    }

    /// Appends n scalars to the buffer.
    ///
    /// This function is slightly more optimized than `extend(iter::repeat_n(item, b))`.
    #[inline]
    pub fn push_n(&mut self, item: T, n: usize)
    where
        T: Copy,
    {
        self.reserve(n);
        unsafe { self.push_n_unchecked(item, n) }
    }

    /// Appends n scalars to the buffer.
    ///
    /// ## Safety
    ///
    /// The caller must ensure there is sufficient capacity in the array.
    #[inline]
    pub unsafe fn push_n_unchecked(&mut self, item: T, n: usize)
    where
        T: Copy,
    {
        // SAFETY: we checked the capacity in the reserve call
        unsafe {
            let data_end = self.data_end_offset();
            let mut dst: *mut T = self.alloc.ptr.as_ptr().add(data_end).cast();
            let end = dst.add(n);
            while dst < end {
                dst.write(item);
                dst = dst.add(1);
            }
        }
        self.length += n;
    }

    /// Appends a slice of type `T`, growing the internal buffer as needed.
    ///
    /// # Example:
    ///
    /// ```
    /// # use vortex_buffer::BufferMut;
    ///
    /// let mut builder = BufferMut::<u16>::with_capacity(10);
    /// builder.extend_from_slice(&[42, 44, 46]);
    ///
    /// assert_eq!(builder.len(), 3);
    /// ```
    #[inline]
    pub fn extend_from_slice(&mut self, slice: &[T]) {
        self.reserve(slice.len());
        let byte_len = size_of_val(slice);
        if byte_len > 0 {
            let data_end = self.data_end_offset();
            // SAFETY: We have reserved enough capacity, and the source/dest don't overlap.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    slice.as_ptr().cast::<u8>(),
                    self.alloc.ptr.as_ptr().add(data_end),
                    byte_len,
                );
            }
        }
        self.length += slice.len();
    }

    /// Splits the buffer into two at the given index.
    ///
    /// Afterward, self contains elements `[0, at)`, and the returned buffer contains elements
    /// `[at, len)`.
    ///
    /// This is an O(n) operation that copies the right half into a new pooled allocation.
    ///
    /// Panics if either half would have a length that is not a multiple of the alignment.
    pub fn split_off(&mut self, at: usize) -> Self {
        if at > self.capacity() {
            vortex_panic!("Cannot split buffer of capacity {} at {}", self.len(), at);
        }

        let bytes_at = at * size_of::<T>();
        if !bytes_at.is_multiple_of(*self.alignment) {
            vortex_panic!(
                "Cannot split buffer at {}, resulting alignment is not {}",
                at,
                self.alignment
            );
        }

        let new_length = self.length.saturating_sub(at);
        let new_byte_len = new_length * size_of::<T>();

        let new_alloc = POOL.acquire(new_byte_len, *self.alignment);
        if new_byte_len > 0 {
            // SAFETY: Both allocations are valid and non-overlapping. The source region
            // [data_ptr + bytes_at .. data_ptr + bytes_at + new_byte_len] is initialized.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.data_ptr().add(bytes_at),
                    new_alloc.ptr.as_ptr(),
                    new_byte_len,
                );
            }
        }

        self.length = self.length.min(at);

        BufferMut {
            alloc: new_alloc,
            byte_offset: 0,
            length: new_length,
            alignment: self.alignment,
            _marker: PhantomData,
        }
    }

    /// Absorbs a mutable buffer that was previously split off.
    ///
    /// This copies the other buffer's data via `extend_from_slice`.
    pub fn unsplit(&mut self, other: Self) {
        if self.alignment != other.alignment {
            vortex_panic!(
                "Cannot unsplit buffers with different alignments: {} and {}",
                self.alignment,
                other.alignment
            );
        }
        self.extend_from_slice(other.as_slice());
        // `other` is dropped here, returning its allocation to the pool.
    }

    /// Return the [`ByteBufferMut`] for this [`BufferMut`].
    pub fn into_byte_buffer(self) -> ByteBufferMut {
        let byte_length = self.length * size_of::<T>();
        // Destructure self to move fields without triggering any Drop.
        let Self {
            alloc,
            byte_offset,
            alignment,
            ..
        } = self;
        ByteBufferMut {
            alloc,
            byte_offset,
            length: byte_length,
            alignment,
            _marker: PhantomData,
        }
    }

    /// Freeze the `BufferMut` into a `Buffer`.
    pub fn freeze(self) -> Buffer<T> {
        let Self {
            mut alloc,
            byte_offset,
            length,
            alignment,
            ..
        } = self;

        let byte_len = length * size_of::<T>();

        // Set the visible range for AsRef<[u8]>, which Bytes::from_owner will use.
        alloc.offset = byte_offset;
        alloc.len = byte_len;

        // Always use from_owner so the resulting Bytes pointer preserves the pool
        // allocation's alignment. This is important because Buffer::as_slice() casts
        // the pointer to *const T, which requires T-alignment even for zero-length slices.
        let bytes = Bytes::from_owner(alloc);

        Buffer {
            bytes,
            length,
            alignment,
            _marker: PhantomData,
        }
    }

    /// Map each element of the buffer with a closure.
    pub fn map_each_in_place<R, F>(self, mut f: F) -> BufferMut<R>
    where
        T: Copy,
        F: FnMut(T) -> R,
    {
        assert_eq!(
            size_of::<T>(),
            size_of::<R>(),
            "Size of T and R do not match"
        );
        // SAFETY: we have checked that `size_of::<T>` == `size_of::<R>`.
        let mut buf: BufferMut<R> = unsafe { std::mem::transmute(self) };
        buf.iter_mut()
            .for_each(|item| *item = f(unsafe { std::mem::transmute_copy(item) }));
        buf
    }

    /// Return a `BufferMut<T>` with the same data as this one with the given alignment.
    ///
    /// If the data is already properly aligned, this is a metadata-only operation.
    ///
    /// If the data is not aligned, we copy it into a new allocation.
    pub fn aligned(self, alignment: Alignment) -> Self {
        if self.as_ptr().align_offset(*alignment) == 0 {
            let Self {
                alloc,
                byte_offset,
                length,
                ..
            } = self;
            Self {
                alloc,
                byte_offset,
                length,
                alignment,
                _marker: PhantomData,
            }
        } else {
            Self::copy_from_aligned(self, alignment)
        }
    }

    /// Transmute a `Buffer<T>` into a `Buffer<U>`.
    ///
    /// # Safety
    ///
    /// The caller must ensure that all possible bit representations of type `T` are valid when
    /// interpreted as type `U`.
    /// See [`std::mem::transmute`] for more details.
    ///
    /// # Panics
    ///
    /// Panics if the type `U` does not have the same size and alignment as `T`.
    pub unsafe fn transmute<U>(self) -> BufferMut<U> {
        assert_eq!(size_of::<T>(), size_of::<U>(), "Buffer type size mismatch");
        assert_eq!(
            align_of::<T>(),
            align_of::<U>(),
            "Buffer type alignment mismatch"
        );

        let Self {
            alloc,
            byte_offset,
            length,
            alignment,
            ..
        } = self;

        BufferMut {
            alloc,
            byte_offset,
            length,
            alignment,
            _marker: PhantomData,
        }
    }
}

impl<T> PartialEq for BufferMut<T> {
    fn eq(&self, other: &Self) -> bool {
        if self.length != other.length || self.alignment != other.alignment {
            return false;
        }
        let byte_len = self.length * size_of::<T>();
        if byte_len == 0 {
            return true;
        }
        // SAFETY: Both slices are within valid allocations and are initialized.
        unsafe {
            let self_slice = std::slice::from_raw_parts(self.data_ptr(), byte_len);
            let other_slice = std::slice::from_raw_parts(other.data_ptr(), byte_len);
            self_slice == other_slice
        }
    }
}

impl<T> Eq for BufferMut<T> {}

impl<T> Clone for BufferMut<T> {
    fn clone(&self) -> Self {
        let mut buffer = BufferMut::<T>::with_capacity_aligned(self.capacity(), self.alignment);
        buffer.extend_from_slice(self.as_slice());
        buffer
    }
}

impl<T: Debug> Debug for BufferMut<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct(&format!("BufferMut<{}>", type_name::<T>()))
            .field("length", &self.length)
            .field("alignment", &self.alignment)
            .field("as_slice", &TruncatedDebug(self.as_slice()))
            .finish()
    }
}

impl<T> Default for BufferMut<T> {
    fn default() -> Self {
        Self::empty()
    }
}

impl<T> Deref for BufferMut<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T> DerefMut for BufferMut<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<T> AsRef<[T]> for BufferMut<T> {
    #[inline]
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> AsMut<[T]> for BufferMut<T> {
    #[inline]
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T> BufferMut<T> {
    /// A helper method for the two [`Extend`] implementations.
    ///
    /// We use the lower bound hint on the iterator to manually write data, and then we continue to
    /// push items normally past the lower bound.
    fn extend_iter(&mut self, mut iter: impl Iterator<Item = T>) {
        // Since we do not know the length of the iterator, we can only guess how much memory we
        // need to reserve. Note that these hints may be inaccurate.
        let (lower_bound, _) = iter.size_hint();

        // We choose not to use the optional upper bound size hint to match the standard library.

        self.reserve(lower_bound);

        let unwritten = self.capacity() - self.len();

        // We store `begin` in the case that the lower bound hint is incorrect.
        let data_end = self.data_end_offset();
        let begin: *const T = unsafe { self.alloc.ptr.as_ptr().add(data_end) }.cast();
        let mut dst: *mut T = begin.cast_mut();

        // As a first step, we manually iterate the iterator up to the known capacity.
        for _ in 0..unwritten {
            let Some(item) = iter.next() else {
                // The lower bound hint may be incorrect.
                break;
            };

            // SAFETY: We have reserved enough capacity to hold this item, and `dst` is a pointer
            // derived from a valid reference to byte data.
            unsafe { dst.write(item) };

            // Note: We used to have `dst.add(iteration).write(item)`, here. However this was much
            // slower than just incrementing `dst`.
            // SAFETY: The offsets fits in `isize`, and because we were able to reserve the memory
            // we know that `add` will not overflow.
            unsafe { dst = dst.add(1) };
        }

        // SAFETY: `dst` was derived from `begin`, which were both valid references to byte data,
        // and since the only operation that `dst` has is `add`, we know that `dst >= begin`.
        let items_written = unsafe { dst.offset_from_unsigned(begin) };
        let length = self.len() + items_written;

        // SAFETY: We have written valid items between the old length and the new length.
        unsafe { self.set_len(length) };

        // Finally, since the iterator will have arbitrarily more items to yield, we push the
        // remaining items normally.
        iter.for_each(|item| self.push(item));
    }

    /// Extends the `BufferMut` with an iterator with `TrustedLen`.
    ///
    /// The caller guarantees that the iterator will have a trusted upper bound, which allows the
    /// implementation to reserve all of the memory needed up front.
    pub fn extend_trusted<I: TrustedLen<Item = T>>(&mut self, iter: I) {
        // Since we know the exact upper bound (from `TrustedLen`), we can reserve all of the memory
        // for this operation up front.
        let (_, upper_bound) = iter.size_hint();
        self.reserve(
            upper_bound
                .vortex_expect("`TrustedLen` iterator somehow didn't have valid upper bound"),
        );

        // We store `begin` in the case that the upper bound hint is incorrect.
        let data_end = self.data_end_offset();
        let begin: *const T = unsafe { self.alloc.ptr.as_ptr().add(data_end) }.cast();
        let mut dst: *mut T = begin.cast_mut();

        iter.for_each(|item| {
            // SAFETY: We have reserved enough capacity to hold this item, and `dst` is a pointer
            // derived from a valid reference to byte data.
            unsafe { dst.write(item) };

            // Note: We used to have `dst.add(iteration).write(item)`, here. However this was much
            // slower than just incrementing `dst`.
            // SAFETY: The offsets fits in `isize`, and because we were able to reserve the memory
            // we know that `add` will not overflow.
            unsafe { dst = dst.add(1) };
        });

        // SAFETY: `dst` was derived from `begin`, which were both valid references to byte data,
        // and since the only operation that `dst` has is `add`, we know that `dst >= begin`.
        let items_written = unsafe { dst.offset_from_unsigned(begin) };
        let length = self.len() + items_written;

        // SAFETY: We have written valid items between the old length and the new length.
        unsafe { self.set_len(length) };
    }

    /// Creates a `BufferMut` from an iterator with a trusted length.
    ///
    /// Internally, this calls [`extend_trusted()`](Self::extend_trusted).
    pub fn from_trusted_len_iter<I>(iter: I) -> Self
    where
        I: TrustedLen<Item = T>,
    {
        let (_, upper_bound) = iter.size_hint();
        let mut buffer = Self::with_capacity(
            upper_bound
                .vortex_expect("`TrustedLen` iterator somehow didn't have valid upper bound"),
        );

        buffer.extend_trusted(iter);
        buffer
    }
}

impl<T> Extend<T> for BufferMut<T> {
    #[inline]
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        self.extend_iter(iter.into_iter())
    }
}

impl<'a, T> Extend<&'a T> for BufferMut<T>
where
    T: Copy + 'a,
{
    #[inline]
    fn extend<I: IntoIterator<Item = &'a T>>(&mut self, iter: I) {
        self.extend_iter(iter.into_iter().copied())
    }
}

impl<T> FromIterator<T> for BufferMut<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        // We don't infer the capacity here and just let the first call to `extend` do it for us.
        let mut buffer = Self::with_capacity(0);
        buffer.extend(iter);
        buffer
    }
}

impl Buf for ByteBufferMut {
    fn remaining(&self) -> usize {
        self.len()
    }

    fn chunk(&self) -> &[u8] {
        self.as_slice()
    }

    fn advance(&mut self, cnt: usize) {
        if !cnt.is_multiple_of(*self.alignment) {
            vortex_panic!(
                "Cannot advance buffer by {} items, resulting alignment is not {}",
                cnt,
                self.alignment
            );
        }
        self.byte_offset += cnt;
        self.length -= cnt;
    }
}

/// As per the BufMut implementation, we must support internal resizing when
/// asked to extend the buffer.
/// See: <https://github.com/tokio-rs/bytes/issues/131>
unsafe impl BufMut for ByteBufferMut {
    #[inline]
    fn remaining_mut(&self) -> usize {
        usize::MAX - self.len()
    }

    #[inline]
    unsafe fn advance_mut(&mut self, cnt: usize) {
        self.length += cnt;
    }

    #[inline]
    fn chunk_mut(&mut self) -> &mut UninitSlice {
        let data_end = self.data_end_offset();
        let spare = self.alloc.capacity.saturating_sub(data_end);
        if spare == 0 {
            // SAFETY: An empty UninitSlice from a dangling pointer is valid.
            return unsafe { UninitSlice::from_raw_parts_mut(NonNull::dangling().as_ptr(), 0) };
        }
        // SAFETY: The pointer is within the allocation and covers `spare` uninitialized bytes.
        unsafe { UninitSlice::from_raw_parts_mut(self.alloc.ptr.as_ptr().add(data_end), spare) }
    }

    fn put<T: Buf>(&mut self, mut src: T)
    where
        Self: Sized,
    {
        while src.has_remaining() {
            let chunk = src.chunk();
            self.extend_from_slice(chunk);
            src.advance(chunk.len());
        }
    }

    #[inline]
    fn put_slice(&mut self, src: &[u8]) {
        self.extend_from_slice(src);
    }

    #[inline]
    fn put_bytes(&mut self, val: u8, cnt: usize) {
        self.push_n(val, cnt)
    }
}

impl Write for ByteBufferMut {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use bytes::Buf;
    use bytes::BufMut;

    use crate::Alignment;
    use crate::BufferMut;
    use crate::ByteBufferMut;
    use crate::buffer_mut;

    #[test]
    fn capacity() {
        let mut n = 57;
        let mut buf = BufferMut::<i32>::with_capacity_aligned(n, Alignment::new(1024));
        assert!(buf.capacity() >= 57);

        while n > 0 {
            buf.push(0);
            assert!(buf.capacity() >= n);
            n -= 1
        }

        assert_eq!(buf.alignment(), Alignment::new(1024));
    }

    #[test]
    fn from_iter() {
        let buf = BufferMut::from_iter([0, 10, 20, 30]);
        assert_eq!(buf.as_slice(), &[0, 10, 20, 30]);
    }

    #[test]
    fn extend() {
        let mut buf = BufferMut::empty();
        buf.extend([0i32, 10, 20, 30]);
        buf.extend([40, 50, 60]);
        assert_eq!(buf.as_slice(), &[0, 10, 20, 30, 40, 50, 60]);
    }

    #[test]
    fn push() {
        let mut buf = BufferMut::empty();
        buf.push(1);
        buf.push(2);
        buf.push(3);
        assert_eq!(buf.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn push_n() {
        let mut buf = BufferMut::empty();
        buf.push_n(0, 100);
        assert_eq!(buf.as_slice(), &[0; 100]);
    }

    #[test]
    fn as_mut() {
        let mut buf = buffer_mut![0, 1, 2];
        // Uses DerefMut
        buf[1] = 0;
        // Uses as_mut
        buf.as_mut()[2] = 0;
        assert_eq!(buf.as_slice(), &[0, 0, 0]);
    }

    #[test]
    fn map_each() {
        let buf = buffer_mut![0i32, 1, 2];
        // Add one, and cast to an unsigned u32 in the same closure
        let buf = buf.map_each_in_place(|i| (i + 1) as u32);
        assert_eq!(buf.as_slice(), &[1u32, 2, 3]);
    }

    #[test]
    fn bytes_buf() {
        let mut buf = ByteBufferMut::copy_from("helloworld".as_bytes());
        assert_eq!(buf.remaining(), 10);
        assert_eq!(buf.chunk(), b"helloworld");

        Buf::advance(&mut buf, 5);
        assert_eq!(buf.remaining(), 5);
        assert_eq!(buf.as_slice(), b"world");
        assert_eq!(buf.chunk(), b"world");
    }

    #[test]
    fn bytes_buf_mut() {
        let mut buf = ByteBufferMut::copy_from("hello".as_bytes());
        assert_eq!(BufMut::remaining_mut(&buf), usize::MAX - 5);

        BufMut::put_slice(&mut buf, b"world");
        assert_eq!(buf.as_slice(), b"helloworld");
    }
}
