// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use core::mem::MaybeUninit;
use std::alloc::Layout;
use std::any::type_name;
use std::cmp::max;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::ops::Deref;
use std::ops::DerefMut;

use itertools::Itertools;
use vortex_error::VortexExpect;
use vortex_error::vortex_panic;

use crate::Alignment;
use crate::Allocation;
use crate::Buffer;
use crate::BufferAllocatorRef;
use crate::ByteBufferMut;
use crate::debug::TruncatedDebug;
use crate::trusted_len::TrustedLen;

/// A mutable buffer that maintains a runtime-defined alignment through resizing operations.
pub struct BufferMut<T> {
    pub(crate) allocation: Allocation,
    pub(crate) ptr: std::ptr::NonNull<T>,
    pub(crate) length: usize,
    pub(crate) capacity: usize,
    pub(crate) alignment: Alignment,
    pub(crate) _marker: std::marker::PhantomData<T>,
}

// SAFETY: BufferMut uniquely owns its allocation and only exposes T across threads.
unsafe impl<T: Send> Send for BufferMut<T> {}
// SAFETY: shared access to BufferMut only exposes shared access to T.
unsafe impl<T: Sync> Sync for BufferMut<T> {}

impl<T> BufferMut<T> {
    /// Create a new `BufferMut` with the requested alignment and capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_in(capacity, BufferAllocatorRef::statically_allocated())
    }

    /// Create a new `BufferMut` with the requested capacity and allocator.
    pub fn with_capacity_in(capacity: usize, allocator: BufferAllocatorRef) -> Self {
        Self::with_capacity_aligned_in(capacity, Alignment::of::<T>(), allocator)
    }

    /// Create a new `BufferMut` with the requested alignment and capacity.
    ///
    /// The allocation is over-aligned to [`Alignment::DEFAULT_ALIGNMENT`] when that is larger than
    /// `alignment`. Use [`with_capacity_preferred_aligned`] to control the over-alignment.
    ///
    /// [`with_capacity_preferred_aligned`]: Self::with_capacity_preferred_aligned
    pub fn with_capacity_aligned(capacity: usize, alignment: Alignment) -> Self {
        Self::with_capacity_aligned_in(
            capacity,
            alignment,
            BufferAllocatorRef::statically_allocated(),
        )
    }

    /// Create a new `BufferMut` with the requested alignment, capacity, and allocator.
    pub fn with_capacity_aligned_in(
        capacity: usize,
        alignment: Alignment,
        allocator: BufferAllocatorRef,
    ) -> Self {
        Self::with_capacity_preferred_aligned_in(
            capacity,
            alignment,
            Some(Alignment::DEFAULT_ALIGNMENT),
            allocator,
        )
    }

    /// Create a new `BufferMut` with the requested alignment and capacity.
    ///
    /// The buffer reports `alignment`, but the underlying allocation is over-aligned to the larger
    /// of `alignment` and `preferred_alignment`.
    pub fn with_capacity_preferred_aligned(
        capacity: usize,
        alignment: Alignment,
        preferred_alignment: Option<Alignment>,
    ) -> Self {
        Self::with_capacity_preferred_aligned_in(
            capacity,
            alignment,
            preferred_alignment,
            BufferAllocatorRef::statically_allocated(),
        )
    }

    /// Create a new allocator-backed `BufferMut` with a requested and preferred alignment.
    pub fn with_capacity_preferred_aligned_in(
        capacity: usize,
        alignment: Alignment,
        preferred_alignment: Option<Alignment>,
        allocator: BufferAllocatorRef,
    ) -> Self {
        let actual = max(
            alignment,
            preferred_alignment.unwrap_or(Alignment::of::<u8>()),
        );

        if !alignment.is_aligned_to(Alignment::of::<T>()) {
            vortex_panic!(
                "Alignment {} must align to the scalar type's alignment {}",
                alignment,
                align_of::<T>()
            );
        }

        let size = capacity
            .checked_mul(size_of::<T>())
            .vortex_expect("buffer capacity overflow");
        let layout = if size == 0 {
            Layout::from_size_align(0, actual.as_usize())
                .unwrap_or_else(|_| vortex_panic!("invalid empty buffer alignment"))
        } else {
            let allocation_size = size
                .checked_add(actual.as_usize())
                .vortex_expect("buffer capacity overflow");
            Layout::from_size_align(allocation_size, 1).unwrap_or_else(|_| {
                vortex_panic!("buffer capacity exceeds maximum allocation size")
            })
        };
        let allocation = Allocation::allocate(layout, allocator);
        let offset = allocation.ptr().as_ptr().align_offset(actual.as_usize());
        // SAFETY: the allocation includes enough padding to reach this aligned pointer.
        let ptr = unsafe { allocation.ptr().add(offset).cast() };
        let capacity = if size_of::<T>() == 0 {
            capacity
        } else {
            (allocation.size() - offset) / size_of::<T>()
        };
        Self {
            allocation,
            ptr,
            length: 0,
            capacity,
            alignment,
            _marker: Default::default(),
        }
    }

    /// Create a new zeroed `BufferMut`.
    pub fn zeroed(len: usize) -> Self {
        Self::zeroed_in(len, BufferAllocatorRef::statically_allocated())
    }

    /// Create a new zeroed `BufferMut` with the requested allocator.
    pub fn zeroed_in(len: usize, allocator: BufferAllocatorRef) -> Self {
        Self::zeroed_aligned_in(len, Alignment::of::<T>(), allocator)
    }

    /// Create a new zeroed `BufferMut` with the requested alignment.
    ///
    /// The allocation is over-aligned to [`Alignment::DEFAULT_ALIGNMENT`] when that is larger than
    /// `alignment`. Use [`zeroed_preferred_aligned`] to control the over-alignment.
    ///
    /// [`zeroed_preferred_aligned`]: Self::zeroed_preferred_aligned
    pub fn zeroed_aligned(len: usize, alignment: Alignment) -> Self {
        Self::zeroed_aligned_in(len, alignment, BufferAllocatorRef::statically_allocated())
    }

    /// Create a zeroed `BufferMut` with an alignment and allocator.
    pub fn zeroed_aligned_in(
        len: usize,
        alignment: Alignment,
        allocator: BufferAllocatorRef,
    ) -> Self {
        Self::zeroed_preferred_aligned_in(
            len,
            alignment,
            Some(Alignment::DEFAULT_ALIGNMENT),
            allocator,
        )
    }

    /// Create a new zeroed `BufferMut` with the requested alignment.
    ///
    /// The buffer reports `alignment`, but the underlying allocation is over-aligned to the larger
    /// of `alignment` and `preferred_alignment`.
    pub fn zeroed_preferred_aligned(
        len: usize,
        alignment: Alignment,
        preferred_alignment: Option<Alignment>,
    ) -> Self {
        Self::zeroed_preferred_aligned_in(
            len,
            alignment,
            preferred_alignment,
            BufferAllocatorRef::statically_allocated(),
        )
    }

    /// Create a zeroed allocator-backed buffer with a requested and preferred alignment.
    pub fn zeroed_preferred_aligned_in(
        len: usize,
        alignment: Alignment,
        preferred_alignment: Option<Alignment>,
        allocator: BufferAllocatorRef,
    ) -> Self {
        let preferred_alignment = preferred_alignment.unwrap_or(Alignment::of::<u8>());
        let actual_alignment = max(preferred_alignment, alignment);
        let size = len
            .checked_mul(size_of::<T>())
            .vortex_expect("buffer length overflow");
        let layout = if size == 0 {
            Layout::from_size_align(0, actual_alignment.as_usize())
                .unwrap_or_else(|_| vortex_panic!("invalid empty buffer alignment"))
        } else {
            let allocation_size = size
                .checked_add(actual_alignment.as_usize())
                .vortex_expect("buffer length overflow");
            Layout::from_size_align(allocation_size, 1)
                .unwrap_or_else(|_| vortex_panic!("buffer length exceeds maximum allocation size"))
        };
        let allocation = Allocation::allocate_zeroed(layout, allocator);
        let offset = allocation
            .ptr()
            .as_ptr()
            .align_offset(actual_alignment.as_usize());
        // SAFETY: the allocation includes enough padding to reach this aligned pointer.
        let ptr = unsafe { allocation.ptr().add(offset).cast() };
        let capacity = if size_of::<T>() == 0 {
            len
        } else {
            (allocation.size() - offset) / size_of::<T>()
        };
        Self {
            allocation,
            ptr,
            length: len,
            capacity,
            alignment,
            _marker: Default::default(),
        }
    }

    /// Create a new empty `BufferMut` with the provided alignment.
    pub fn empty() -> Self {
        Self::empty_aligned(Alignment::of::<T>())
    }

    /// Create a new empty `BufferMut` with the provided alignment.
    ///
    /// The allocation is over-aligned to [`Alignment::DEFAULT_ALIGNMENT`] when that is larger than
    /// `alignment`. Use [`empty_preferred_aligned`] to control the over-alignment.
    ///
    /// [`empty_preferred_aligned`]: Self::empty_preferred_aligned
    pub fn empty_aligned(alignment: Alignment) -> Self {
        Self::empty_aligned_in(alignment, BufferAllocatorRef::statically_allocated())
    }

    /// Create an empty `BufferMut` with an alignment and allocator.
    pub fn empty_aligned_in(alignment: Alignment, allocator: BufferAllocatorRef) -> Self {
        Self::with_capacity_aligned_in(0, alignment, allocator)
    }

    /// Create a new empty `BufferMut` with the provided alignment.
    ///
    /// The buffer reports `alignment`, but the underlying allocation is over-aligned to the larger
    /// of `alignment` and `preferred_alignment`.
    pub fn empty_preferred_aligned(
        alignment: Alignment,
        preferred_alignment: Option<Alignment>,
    ) -> Self {
        BufferMut::with_capacity_preferred_aligned_in(
            0,
            alignment,
            preferred_alignment,
            BufferAllocatorRef::statically_allocated(),
        )
    }

    /// Create a new full `BufferMut` with the given value.
    pub fn full(item: T, len: usize) -> Self
    where
        T: Copy,
    {
        Self::full_in(item, len, BufferAllocatorRef::statically_allocated())
    }

    /// Create a full `BufferMut` with the given value and allocator.
    pub fn full_in(item: T, len: usize, allocator: BufferAllocatorRef) -> Self
    where
        T: Copy,
    {
        let mut buffer = BufferMut::<T>::with_capacity_in(len, allocator);
        buffer.push_n(item, len);
        buffer
    }

    /// Create a mutable scalar buffer by copying the contents of the slice.
    pub fn copy_from(other: impl AsRef<[T]>) -> Self {
        Self::copy_from_in(other, BufferAllocatorRef::statically_allocated())
    }

    /// Create a mutable scalar buffer by copying with the given allocator.
    pub fn copy_from_in(other: impl AsRef<[T]>, allocator: BufferAllocatorRef) -> Self {
        Self::copy_from_aligned_in(other, Alignment::of::<T>(), allocator)
    }

    /// Create a mutable scalar buffer with the alignment by copying the contents of the slice.
    ///
    /// The allocation is over-aligned to [`Alignment::DEFAULT_ALIGNMENT`] when that is larger than
    /// `alignment`. Use [`copy_from_preferred_aligned`] to control the over-alignment.
    ///
    /// [`copy_from_preferred_aligned`]: Self::copy_from_preferred_aligned
    ///
    /// ## Panics
    ///
    /// Panics when the requested alignment isn't itself aligned to type T.
    pub fn copy_from_aligned(other: impl AsRef<[T]>, alignment: Alignment) -> Self {
        Self::copy_from_aligned_in(other, alignment, BufferAllocatorRef::statically_allocated())
    }

    /// Copy values into a mutable buffer with the given alignment and allocator.
    pub fn copy_from_aligned_in(
        other: impl AsRef<[T]>,
        alignment: Alignment,
        allocator: BufferAllocatorRef,
    ) -> Self {
        Self::copy_from_preferred_aligned_in(
            other,
            alignment,
            Some(Alignment::DEFAULT_ALIGNMENT),
            allocator,
        )
    }

    /// Create a mutable scalar buffer with the alignment by copying the contents of the slice.
    ///
    /// The buffer reports `alignment`, but the underlying allocation is over-aligned to the larger
    /// of `alignment` and `preferred_alignment`.
    ///
    /// ## Panics
    ///
    /// Panics when the requested alignment isn't itself aligned to type T.
    pub fn copy_from_preferred_aligned(
        other: impl AsRef<[T]>,
        alignment: Alignment,
        preferred_alignment: Option<Alignment>,
    ) -> Self {
        Self::copy_from_preferred_aligned_in(
            other,
            alignment,
            preferred_alignment,
            BufferAllocatorRef::statically_allocated(),
        )
    }

    /// Copy values with the given allocator, requested alignment, and preferred alignment.
    pub fn copy_from_preferred_aligned_in(
        other: impl AsRef<[T]>,
        alignment: Alignment,
        preferred_alignment: Option<Alignment>,
        allocator: BufferAllocatorRef,
    ) -> Self {
        if !alignment.is_aligned_to(Alignment::of::<T>()) {
            vortex_panic!("Given alignment is not aligned to type T")
        }
        let other = other.as_ref();
        let mut buffer = Self::with_capacity_preferred_aligned_in(
            other.len(),
            alignment,
            preferred_alignment,
            allocator,
        );
        buffer.extend_from_slice(other);
        debug_assert_eq!(buffer.alignment(), alignment);
        buffer
    }

    /// Get the alignment of the buffer.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub fn alignment(&self) -> Alignment {
        self.alignment
    }

    /// Returns the allocator that owns this buffer.
    pub fn allocator(&self) -> &BufferAllocatorRef {
        self.allocation.allocator()
    }

    /// Returns the length of the buffer.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.length
    }

    /// Returns whether the buffer is empty.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Returns the capacity of the buffer.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns a raw pointer to the buffer's data.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    /// Returns a mutable raw pointer to the buffer's data.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    /// Returns a slice over the buffer of elements of type T.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: ptr is in the live allocation and construction checks its alignment.
        unsafe { std::slice::from_raw_parts(self.as_ptr(), self.length) }
    }

    /// Returns a slice over the buffer of elements of type T.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: BufferMut uniquely owns the allocation and the initialized range is in bounds.
        unsafe { std::slice::from_raw_parts_mut(self.as_mut_ptr(), self.length) }
    }

    /// Clear the buffer, retaining any existing capacity.
    #[inline]
    pub fn clear(&mut self) {
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
        if additional <= self.capacity() - self.length {
            // We can fit the additional bytes in the remaining capacity. Nothing to do.
            return;
        }

        // Otherwise, reserve additional + alignment bytes in case we need to realign the buffer.
        self.reserve_allocate(additional);
    }

    /// A separate function so we can inline the reserve call's fast path.
    fn reserve_allocate(&mut self, additional: usize) {
        let required = self
            .length
            .checked_add(additional)
            .vortex_expect("buffer capacity overflow");
        let required_size = required
            .checked_mul(size_of::<T>())
            .vortex_expect("buffer capacity overflow");
        let alignment = self.alignment;
        let current_size = self
            .capacity
            .checked_mul(size_of::<T>())
            .vortex_expect("buffer capacity overflow");
        let logical_size = required_size
            .max(current_size.saturating_mul(2))
            .max(Alignment::DEFAULT_ALIGNMENT.as_usize());
        let allocation_size = logical_size
            .checked_add(alignment.as_usize())
            .vortex_expect("buffer capacity overflow");
        let allocation_alignment = if self.allocation.size() == 0 {
            1
        } else {
            self.allocation.alignment()
        };
        let layout = Layout::from_size_align(allocation_size, allocation_alignment)
            .unwrap_or_else(|_| vortex_panic!("buffer capacity exceeds maximum allocation size"));

        let old_offset = self.ptr.cast::<u8>().addr().get() - self.allocation.ptr().addr().get();
        let new_offset = if self.allocation.allocator().is_statically_allocated() {
            let allocation =
                Allocation::allocate(layout, BufferAllocatorRef::statically_allocated());
            let new_offset = allocation.ptr().as_ptr().align_offset(alignment.as_usize());
            // SAFETY: both allocations have room for the initialized elements and do not overlap.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.ptr.cast::<u8>().as_ptr(),
                    allocation.ptr().as_ptr().add(new_offset),
                    self.length * size_of::<T>(),
                );
            }
            self.allocation = allocation;
            new_offset
        } else {
            self.allocation.grow(layout);
            let new_offset = self
                .allocation
                .ptr()
                .as_ptr()
                .align_offset(alignment.as_usize());
            if new_offset != old_offset {
                // SAFETY: grow preserved the initialized elements at old_offset. The new allocation
                // has room for the requested elements plus alignment padding, and copy permits
                // overlap.
                unsafe {
                    std::ptr::copy(
                        self.allocation.ptr().as_ptr().add(old_offset),
                        self.allocation.ptr().as_ptr().add(new_offset),
                        self.length * size_of::<T>(),
                    );
                }
            }
            new_offset
        };
        // SAFETY: new_offset was computed within the allocation for alignment.
        self.ptr = unsafe { self.allocation.ptr().add(new_offset).cast() };
        self.capacity = logical_size / size_of::<T>();
    }

    /// Returns the spare capacity of the buffer as a slice of `MaybeUninit<T>`.
    /// Has identical semantics to [`Vec::spare_capacity_mut`].
    ///
    /// The returned slice can be used to fill the buffer with data (e.g. by
    /// reading from a file) before marking the data as initialized using the
    /// [`set_len`] method.
    ///
    /// Note that the returned slice may be larger than the capacity requested at
    /// construction, since the underlying allocation can be rounded up (e.g. to
    /// satisfy alignment requirements).
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
        // SAFETY: offset + length is within the allocation and points at spare capacity.
        let dst = unsafe { self.as_mut_ptr().add(self.length) }.cast::<MaybeUninit<T>>();
        unsafe { std::slice::from_raw_parts_mut(dst, self.capacity() - self.length) }
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
        // SAFETY: the caller ensures we have sufficient capacity
        unsafe {
            let dst = self.as_mut_ptr().add(self.length);
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
        // SAFETY: the caller guarantees enough spare capacity.
        let mut dst = unsafe { self.as_mut_ptr().add(self.length) };
        // SAFETY: we checked the capacity in the reserve call
        unsafe {
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
        // SAFETY: reserve made the destination valid and non-overlapping for slice.len() values.
        unsafe {
            std::ptr::copy_nonoverlapping(
                slice.as_ptr(),
                self.as_mut_ptr().add(self.length),
                slice.len(),
            );
        }
        self.length += slice.len();
    }

    /// Return the [`ByteBufferMut`] for this [`BufferMut`].
    pub fn into_byte_buffer(self) -> ByteBufferMut {
        let capacity = self
            .capacity
            .checked_mul(size_of::<T>())
            .vortex_expect("buffer capacity overflow");
        ByteBufferMut {
            allocation: self.allocation,
            ptr: self.ptr.cast(),
            length: self.length * size_of::<T>(),
            capacity,
            alignment: self.alignment,
            _marker: Default::default(),
        }
    }

    /// Freeze the `BufferMut` into a `Buffer`.
    pub fn freeze(self) -> Buffer<T> {
        let offset = self.ptr.cast::<u8>().addr().get() - self.allocation.ptr().addr().get();
        Buffer::from_allocation(self.allocation, offset, self.length, self.alignment)
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
        if self.as_ptr().align_offset(alignment.as_usize()) == 0 {
            Self { alignment, ..self }
        } else {
            let capacity = self.capacity();
            let allocator = self.allocation.allocator().clone();
            let mut aligned = Self::with_capacity_aligned_in(capacity, alignment, allocator);
            aligned.extend_from_slice(&self);
            aligned.capacity = capacity;
            aligned
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

        BufferMut {
            allocation: self.allocation,
            ptr: self.ptr.cast(),
            length: self.length,
            capacity: self.capacity,
            alignment: self.alignment,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<T> Clone for BufferMut<T> {
    fn clone(&self) -> Self {
        let mut buffer = BufferMut::<T>::with_capacity_aligned_in(
            self.capacity(),
            self.alignment,
            self.allocation.allocator().clone(),
        );
        buffer.extend_from_slice(self.as_slice());
        buffer
    }
}

impl<T: PartialEq> PartialEq for BufferMut<T> {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: Eq> Eq for BufferMut<T> {}

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
        let begin: *const T = self.spare_capacity_mut().as_mut_ptr().cast();
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
        let (_, upper_bound) = iter.size_hint();
        self.reserve(
            upper_bound
                .vortex_expect("`TrustedLen` iterator somehow didn't have valid upper bound"),
        );

        let begin: *const T = self.spare_capacity_mut().as_mut_ptr().cast();
        let mut dst: *mut T = begin.cast_mut();

        iter.for_each(|item| {
            // SAFETY: We have reserved enough capacity to hold this item, and `dst` is a pointer
            // derived from a valid reference to byte data.
            unsafe { dst.write(item) };

            // Note: We used to have `dst.add(iteration).write(item)`, here. However this was much
            // slower than just incrementing `dst`.
            // SAFETY: The offset fits in `isize`, and because we were able to reserve the memory
            // we know that `add` will not overflow.
            unsafe { dst = dst.add(1) };
        });

        // SAFETY: `dst` starts at `begin` and advances by one for each item, so both pointers refer
        // to the same allocation and `dst` is at or after `begin`.
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

    /// Like [`extend_trusted()`](Self::extend_trusted), but the iterator yields `Result<T, E>`
    /// and the extension short-circuits on the first `Err`.
    ///
    /// On error, items written before the failure remain in the buffer.
    pub fn try_extend_trusted<E, I>(&mut self, iter: I) -> Result<(), E>
    where
        I: TrustedLen<Item = Result<T, E>>,
    {
        iter.process_results(|values| self.extend_trusted(values))
    }

    /// Like [`from_trusted_len_iter()`](Self::from_trusted_len_iter), but the iterator yields
    /// `Result<T, E>` and construction short-circuits on the first `Err`.
    pub fn try_from_trusted_len_iter<E, I>(iter: I) -> Result<Self, E>
    where
        I: TrustedLen<Item = Result<T, E>>,
    {
        iter.process_results(|values| Self::from_trusted_len_iter(values))
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
        let iter = iter.into_iter();
        let mut buffer = Self::with_capacity(iter.size_hint().0);
        buffer.extend(iter);
        buffer
    }
}

#[cfg(test)]
mod test {
    use crate::Alignment;
    use crate::BufferMut;
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
    fn growth_preserves_alignment_and_values() {
        let alignment = Alignment::new(4096);
        let mut buffer = BufferMut::<u64>::with_capacity_aligned(1, alignment);

        for value in 0..10_000 {
            buffer.push(value);
            assert!(alignment.is_offset_aligned(buffer.as_ptr().addr()));
        }

        assert_eq!(buffer.as_slice(), (0..10_000).collect::<Vec<_>>());
    }

    #[test]
    fn growth_seeds_and_doubles_logical_capacity() {
        let alignment = Alignment::new(64);
        let mut buffer = BufferMut::<u8>::empty_aligned(alignment);

        buffer.push(0);
        let capacity = buffer.capacity();
        assert_eq!(capacity, Alignment::DEFAULT_ALIGNMENT.as_usize());

        buffer.reserve(capacity);
        assert_eq!(buffer.capacity(), capacity * 2);
    }

    #[test]
    fn static_growth_copies_live_data() {
        let mut buffer = BufferMut::<u32>::with_capacity(1);
        let capacity = buffer.capacity();
        buffer.extend(std::iter::repeat_n(7, capacity));
        let old_ptr = buffer.as_ptr();

        buffer.push(u32::MAX);

        assert_ne!(buffer.as_ptr(), old_ptr);
        assert_eq!(&buffer[..capacity], vec![7; capacity]);
        assert_eq!(buffer[capacity], u32::MAX);
    }

    #[test]
    fn raising_logical_alignment_preserves_capacity() {
        let buffer =
            BufferMut::<u8>::with_capacity_preferred_aligned(1, Alignment::of::<u8>(), None);
        let capacity = buffer.capacity();

        let mut buffer = buffer.aligned(Alignment::new(2));

        assert_eq!(buffer.capacity(), capacity);
        buffer.extend(0..100);
        assert!(Alignment::new(2).is_ptr_aligned(buffer.as_ptr()));
        assert_eq!(buffer.as_slice(), (0..100).collect::<Vec<_>>());
    }

    #[test]
    fn from_iter() {
        let buf = BufferMut::from_iter([0, 10, 20, 30]);
        assert_eq!(buf.as_slice(), &[0, 10, 20, 30]);
    }

    #[test]
    fn try_from_trusted_len_iter_ok() {
        let buf = BufferMut::<i32>::try_from_trusted_len_iter(
            [0, 10, 20, 30].iter().map(|&v| Ok::<_, ()>(v)),
        )
        .unwrap();
        assert_eq!(buf.as_slice(), &[0, 10, 20, 30]);
    }

    #[test]
    fn try_from_trusted_len_iter_err() {
        let result: Result<BufferMut<i32>, &'static str> = BufferMut::try_from_trusted_len_iter(
            [0, 10, 20, 30]
                .iter()
                .map(|&v| if v == 20 { Err("bad") } else { Ok(v) }),
        );
        assert_eq!(result.err(), Some("bad"));
    }

    #[test]
    fn try_extend_trusted_retains_values_before_error() {
        let mut buf = BufferMut::from_iter([0, 10]);
        let result = buf.try_extend_trusted([Ok(20), Err("bad"), Ok(30)].into_iter());

        assert_eq!(result, Err("bad"));
        assert_eq!(buf.as_slice(), &[0, 10, 20]);
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
    fn buffer_mut_zeroed() {
        const LEN: usize = 17;

        let mut buf = BufferMut::<u32>::zeroed(LEN);

        assert_eq!(
            buf.as_ptr().align_offset(Alignment::of::<u32>().as_usize()),
            0
        );
        assert_eq!(buf.as_slice(), &[0; LEN]);

        buf[3] = 7;
        assert_eq!(buf.as_slice()[3], 7);
    }

    #[test]
    fn buffer_mut_zeroed_aligned() {
        const LEN: usize = 17;
        let alignment = Alignment::new(64);

        let mut buf = BufferMut::<u32>::zeroed_aligned(LEN, alignment);

        assert_eq!(buf.as_ptr().align_offset(alignment.as_usize()), 0);
        assert_eq!(buf.as_slice(), &[0; LEN]);

        buf[3] = 7;
        assert_eq!(buf.as_slice()[3], 7);
    }
}
