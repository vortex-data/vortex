// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::type_name;
use std::cmp::Ordering;
use std::collections::Bound;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::ops::Deref;
use std::ops::RangeBounds;
use std::ptr::NonNull;
use std::sync::Arc;

use bytes::Buf;
use bytes::Bytes;
use vortex_error::VortexExpect;
use vortex_error::vortex_panic;

use crate::Alignment;
use crate::Allocation;
use crate::BufferAllocatorRef;
use crate::BufferBacking;
use crate::BufferMut;
use crate::ByteBuffer;
use crate::debug::TruncatedDebug;
use crate::trusted_len::TrustedLen;

/// An immutable buffer of items of `T`.
#[derive(Clone)]
pub struct Buffer<T> {
    pub(crate) ptr: NonNull<T>,
    pub(crate) length: usize,
    pub(crate) alignment: Alignment,
    pub(crate) backing: Option<Arc<BufferBacking>>,
}

// SAFETY: Buffer is an immutable view over backing memory. Its pointer remains valid while the
// backing is live, and sharing elements follows the same bounds as sharing a slice.
unsafe impl<T: Send> Send for Buffer<T> {}
// SAFETY: see the Send implementation above.
unsafe impl<T: Sync> Sync for Buffer<T> {}

impl<T> Default for Buffer<T> {
    fn default() -> Self {
        Self {
            ptr: empty_ptr(),
            length: 0,
            alignment: Alignment::of::<T>(),
            backing: None,
        }
    }
}

impl<T: PartialEq> PartialEq for Buffer<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: Eq> Eq for Buffer<T> {}

impl<T: Ord> Ord for Buffer<T> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_slice().cmp(other.as_slice())
    }
}

impl<T: PartialOrd> PartialOrd for Buffer<T> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.as_slice().partial_cmp(other.as_slice())
    }
}

impl<T: Hash> Hash for Buffer<T> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_slice().hash(state)
    }
}

impl<T> Buffer<T> {
    pub(crate) fn from_allocation(
        allocation: Allocation,
        offset: usize,
        length: usize,
        alignment: Alignment,
    ) -> Self {
        // SAFETY: BufferMut keeps offset within allocation, including for empty buffers.
        let ptr = unsafe { allocation.ptr().add(offset).cast() };
        Self {
            ptr,
            length,
            alignment,
            backing: Some(Arc::new(BufferBacking::Owned(allocation))),
        }
    }

    fn from_owner(owner: impl crate::BufferOwner, alignment: Alignment) -> Self {
        let owner: Box<dyn crate::BufferOwner> = Box::new(owner);
        let length = owner.len() / size_of::<T>();
        let ptr = if length == 0 {
            empty_ptr()
        } else {
            NonNull::new(owner.as_ptr().cast_mut().cast()).vortex_expect("owner pointer is null")
        };
        Self {
            ptr,
            length,
            alignment,
            backing: Some(Arc::new(BufferBacking::External { _owner: owner })),
        }
    }

    fn from_bytes(bytes: Bytes, alignment: Alignment) -> Self {
        let length = bytes.len() / size_of::<T>();
        if length == 0 {
            return Self::empty_aligned(alignment);
        }
        let ptr =
            NonNull::new(bytes.as_ptr().cast_mut().cast()).vortex_expect("Bytes pointer is null");
        Self {
            ptr,
            length,
            alignment,
            backing: Some(Arc::new(BufferBacking::Bytes(bytes))),
        }
    }

    #[cfg(feature = "arrow")]
    pub(crate) fn from_arrow_owner(
        arrow: arrow_buffer::Buffer,
        length: usize,
        alignment: Alignment,
    ) -> Self {
        if length == 0 {
            return Self::empty_aligned(alignment);
        }
        let ptr = NonNull::new(arrow.as_ptr().cast_mut().cast())
            .vortex_expect("Arrow buffer pointer is null");
        Self {
            ptr,
            length,
            alignment,
            backing: Some(Arc::new(BufferBacking::Arrow(arrow))),
        }
    }

    /// Returns a new `Buffer<T>` copied from the provided `Vec<T>`, `&[T]`, etc.
    ///
    /// Due to our underlying usage of `bytes::Bytes`, we are unable to take zero-copy ownership
    /// of the provided `Vec<T>` while maintaining the ability to convert it back into a mutable
    /// buffer. We could fix this by forking `Bytes`, or in many other complex ways, but for now
    /// callers should prefer to construct `Buffer<T>` from a `BufferMut<T>`.
    pub fn copy_from(values: impl AsRef<[T]>) -> Self {
        BufferMut::copy_from(values).freeze()
    }

    /// Returns a new `Buffer<T>` copied with the provided allocator.
    pub fn copy_from_in(values: impl AsRef<[T]>, allocator: BufferAllocatorRef) -> Self {
        BufferMut::copy_from_in(values, allocator).freeze()
    }

    /// Returns a new `Buffer<T>` copied from the provided slice and with the requested alignment.
    ///
    /// The allocation is over-aligned to [`Alignment::DEFAULT_ALIGNMENT`] when that is larger than
    /// `alignment`. Use [`copy_from_preferred_aligned`] to control the over-alignment.
    ///
    /// [`copy_from_preferred_aligned`]: Self::copy_from_preferred_aligned
    pub fn copy_from_aligned(values: impl AsRef<[T]>, alignment: Alignment) -> Self {
        Self::copy_from_preferred_aligned(values, alignment, Some(Alignment::DEFAULT_ALIGNMENT))
    }

    /// Returns a new `Buffer<T>` copied from the provided slice and with the requested alignment.
    ///
    /// The buffer reports `alignment`, but the underlying allocation is over-aligned to the larger
    /// of `alignment` and `preferred_alignment`.
    pub fn copy_from_preferred_aligned(
        values: impl AsRef<[T]>,
        alignment: Alignment,
        preferred_alignment: Option<Alignment>,
    ) -> Self {
        BufferMut::copy_from_preferred_aligned(values, alignment, preferred_alignment).freeze()
    }

    /// Create a new zeroed `Buffer` with the given value.
    pub fn zeroed(len: usize) -> Self {
        Self::zeroed_aligned(len, Alignment::of::<T>())
    }

    /// Create a new zeroed `Buffer` with the provided allocator.
    pub fn zeroed_in(len: usize, allocator: BufferAllocatorRef) -> Self {
        BufferMut::zeroed_in(len, allocator).freeze()
    }

    /// Create a new zeroed `Buffer` with the requested alignment.
    ///
    /// The allocation is over-aligned to [`Alignment::DEFAULT_ALIGNMENT`] when that is larger than
    /// `alignment`. Use [`zeroed_preferred_aligned`] to control the over-alignment.
    ///
    /// [`zeroed_preferred_aligned`]: Self::zeroed_preferred_aligned
    pub fn zeroed_aligned(len: usize, alignment: Alignment) -> Self {
        Self::zeroed_preferred_aligned(len, alignment, Some(Alignment::DEFAULT_ALIGNMENT))
    }

    /// Create a new zeroed `Buffer` with the requested alignment.
    ///
    /// The buffer reports `alignment`, but the underlying allocation is over-aligned to the larger
    /// of `alignment` and `preferred_alignment`.
    pub fn zeroed_preferred_aligned(
        len: usize,
        alignment: Alignment,
        preferred_alignment: Option<Alignment>,
    ) -> Self {
        BufferMut::zeroed_preferred_aligned(len, alignment, preferred_alignment).freeze()
    }

    /// Create a new empty `ByteBuffer` with the provided alignment.
    pub fn empty() -> Self {
        Self::empty_aligned(Alignment::of::<T>())
    }

    /// Create a new empty `ByteBuffer` with the provided alignment.
    ///
    /// This does not allocate. Empty buffers use an aligned dangling pointer.
    pub fn empty_aligned(alignment: Alignment) -> Self {
        if !alignment.is_aligned_to(Alignment::of::<T>()) {
            vortex_panic!(
                "Alignment {} must align to the scalar type's alignment {}",
                alignment,
                Alignment::of::<T>(),
            );
        }
        Self {
            ptr: empty_ptr(),
            length: 0,
            alignment,
            backing: None,
        }
    }

    /// Create a new full `ByteBuffer` with the given value.
    pub fn full(item: T, len: usize) -> Self
    where
        T: Copy,
    {
        BufferMut::full(item, len).freeze()
    }

    /// Create a full `Buffer` with the given value and allocator.
    pub fn full_in(item: T, len: usize, allocator: BufferAllocatorRef) -> Self
    where
        T: Copy,
    {
        BufferMut::full_in(item, len, allocator).freeze()
    }

    /// Create a `Buffer<T>` zero-copy from a `ByteBuffer`.
    ///
    /// ## Panics
    ///
    /// Panics if the buffer is not aligned to the size of `T`, or the length is not a multiple of
    /// the size of `T`.
    pub fn from_byte_buffer(buffer: ByteBuffer) -> Self {
        // TODO(ngates): should this preserve the current alignment of the buffer?
        Self::from_byte_buffer_aligned(buffer, Alignment::of::<T>())
    }

    /// Create a `Buffer<T>` zero-copy from a `ByteBuffer`.
    ///
    /// ## Panics
    ///
    /// Panics if the buffer is not aligned to the given alignment, if the length is not a multiple
    /// of the size of `T`, or if the given alignment is not aligned to that of `T`.
    pub fn from_byte_buffer_aligned(buffer: ByteBuffer, alignment: Alignment) -> Self {
        if !alignment.is_aligned_to(Alignment::of::<T>()) {
            vortex_panic!(
                "Alignment {} must be compatible with the scalar type's alignment {}",
                alignment,
                Alignment::of::<T>(),
            );
        }
        if !alignment.is_ptr_aligned(buffer.as_ptr()) {
            vortex_panic!("Buffer must align to the requested alignment {}", alignment);
        }
        if !buffer.len().is_multiple_of(size_of::<T>()) {
            vortex_panic!(
                "Buffer length {} must be a multiple of the scalar type's size {}",
                buffer.len(),
                size_of::<T>()
            );
        }
        Self {
            ptr: buffer.ptr.cast(),
            length: buffer.length / size_of::<T>(),
            alignment,
            backing: buffer.backing,
        }
    }

    /// Create a `Buffer<T>` zero-copy from a `Bytes`.
    ///
    /// ## Panics
    ///
    /// Panics if the buffer is not aligned to the size of `T`, or the length is not a multiple of
    /// the size of `T`.
    pub fn from_bytes_aligned(bytes: Bytes, alignment: Alignment) -> Self {
        if !alignment.is_aligned_to(Alignment::of::<T>()) {
            vortex_panic!(
                "Alignment {} must be compatible with the scalar type's alignment {}",
                alignment,
                Alignment::of::<T>(),
            );
        }
        if !alignment.is_ptr_aligned(bytes.as_ptr()) {
            vortex_panic!(
                "Bytes alignment must align to the requested alignment {}",
                alignment,
            );
        }
        if !bytes.len().is_multiple_of(size_of::<T>()) {
            vortex_panic!(
                "Bytes length {} must be a multiple of the scalar type's size {}",
                bytes.len(),
                size_of::<T>()
            );
        }
        Self::from_bytes(bytes, alignment)
    }

    /// Create a buffer with values from the TrustedLen iterator.
    /// Should be preferred over `from_iter` when the iterator is known to be `TrustedLen`.
    pub fn from_trusted_len_iter<I: TrustedLen<Item = T>>(iter: I) -> Self {
        BufferMut::from_trusted_len_iter(iter).freeze()
    }

    /// Map each element of the buffer with a closure.
    pub fn map_each_in_place<R, F>(self, mut f: F) -> BufferMut<R>
    where
        T: Copy,
        F: FnMut(T) -> R,
    {
        match self.try_into_mut() {
            Ok(mut_buf) => mut_buf.map_each_in_place(f),
            Err(buf) => {
                let len = buf.len();
                let allocator = buf.allocator().clone();
                let mut out_buf = BufferMut::with_capacity_in(len, allocator);
                out_buf
                    .spare_capacity_mut()
                    .iter_mut()
                    .zip(buf)
                    .for_each(|(out, in_)| {
                        out.write(f(in_));
                    });
                // Safety: just assigned to each value
                unsafe { out_buf.set_len(len) }
                out_buf
            }
        }
    }

    /// Clear the buffer, preserving existing capacity.
    pub fn clear(&mut self) {
        self.length = 0;
    }

    /// Returns the length of the buffer in elements of type T.
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

    /// Returns the alignment of the buffer.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub fn alignment(&self) -> Alignment {
        self.alignment
    }

    /// Returns the allocator to use for derived buffers.
    ///
    /// External buffers use the static allocator.
    pub fn allocator(&self) -> &BufferAllocatorRef {
        match self.backing.as_deref() {
            Some(backing) => backing.allocator(),
            None => BufferAllocatorRef::static_ref(),
        }
    }

    /// Returns a raw pointer to the buffer's data.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    /// Returns a slice over the buffer of elements of type T.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: ptr points into the live backing and construction checks its alignment.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.length) }
    }

    /// Return a view over the buffer as an opaque byte slice.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub fn as_bytes(&self) -> &[u8] {
        // SAFETY: the element range is initialized and remains live through backing.
        unsafe {
            std::slice::from_raw_parts(self.ptr.as_ptr().cast(), size_of_val(self.as_slice()))
        }
    }

    /// Returns an iterator over the buffer of elements of type T.
    pub fn iter(&self) -> Iter<'_, T> {
        Iter {
            inner: self.as_slice().iter(),
        }
    }

    /// Returns a slice of self for the provided range.
    ///
    /// # Panics
    ///
    /// Requires that `begin <= end` and `end <= self.len()`.
    /// Also requires that both `begin` and `end` are aligned to the buffer's required alignment.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub fn slice(&self, range: impl RangeBounds<usize>) -> Self {
        self.slice_with_alignment(range, self.alignment)
    }

    /// Returns a slice of self for the provided range, with no guarantees about the resulting
    /// alignment.
    ///
    /// # Panics
    ///
    /// Requires that `begin <= end` and `end <= self.len()`.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub fn slice_unaligned(&self, range: impl RangeBounds<usize>) -> Self {
        self.slice_with_alignment(range, Alignment::of::<u8>())
    }

    /// Returns a slice of self for the provided range, ensuring the resulting slice has the
    /// given alignment.
    ///
    /// # Panics
    ///
    /// Requires that `begin <= end` and `end <= self.len()`.
    /// Also requires that both `begin` and `end` are aligned to the given alignment.
    pub fn slice_with_alignment(
        &self,
        range: impl RangeBounds<usize>,
        alignment: Alignment,
    ) -> Self {
        let len = self.len();
        let begin = match range.start_bound() {
            Bound::Included(&n) => n,
            Bound::Excluded(&n) => n.checked_add(1).vortex_expect("out of range"),
            Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            Bound::Included(&n) => n.checked_add(1).vortex_expect("out of range"),
            Bound::Excluded(&n) => n,
            Bound::Unbounded => len,
        };

        if begin > end {
            vortex_panic!(
                "range start must not be greater than end: {:?} <= {:?}",
                begin,
                end
            );
        }
        if end > len {
            vortex_panic!("range end out of bounds: {:?} > {:?}", end, len);
        }

        if end == begin {
            // We prefer to return a new empty buffer instead of sharing this one and creating a
            // strong reference just to hold an empty slice.
            return Self::empty_aligned(alignment);
        }

        let begin_byte = begin * size_of::<T>();
        if !alignment.is_offset_aligned(begin_byte) {
            vortex_panic!(
                "range start must be aligned to {alignment:?}, byte {}",
                begin_byte
            );
        }
        if !alignment.is_aligned_to(Alignment::of::<T>()) {
            vortex_panic!("Slice alignment must at least align to type T")
        }

        Self {
            // SAFETY: begin is in bounds and the alignment check applies to the new pointer.
            ptr: unsafe { self.ptr.add(begin) },
            length: end - begin,
            alignment,
            backing: self.backing.clone(),
        }
    }

    /// Returns a slice of self that is equivalent to the given subset.
    ///
    /// When processing the buffer you will often end up with `&[T]` that is a subset
    /// of the underlying buffer. This function turns the slice into a slice of the buffer
    /// it has been taken from.
    ///
    /// # Panics:
    /// Requires that the given sub slice is in fact contained within the Bytes buffer; otherwise this function will panic.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub fn slice_ref(&self, subset: &[T]) -> Self {
        self.slice_ref_with_alignment(subset, Alignment::of::<T>())
    }

    /// Returns a slice of self that is equivalent to the given subset.
    ///
    /// When processing the buffer you will often end up with `&[T]` that is a subset
    /// of the underlying buffer. This function turns the slice into a slice of the buffer
    /// it has been taken from.
    ///
    /// # Panics:
    /// Requires that the given sub slice is in fact contained within the Bytes buffer; otherwise this function will panic.
    /// Also requires that the given alignment aligns to the type of slice and is smaller or equal to the buffers alignment
    pub fn slice_ref_with_alignment(&self, subset: &[T], alignment: Alignment) -> Self {
        if !alignment.is_aligned_to(Alignment::of::<T>()) {
            vortex_panic!("slice_ref alignment must at least align to type T")
        }

        if !self.alignment.is_aligned_to(alignment) {
            vortex_panic!("slice_ref subset alignment must at least align to the buffer alignment")
        }

        if !alignment.is_ptr_aligned(subset.as_ptr()) {
            vortex_panic!("slice_ref subset must be aligned to {:?}", alignment);
        }

        let start = self.as_ptr().addr();
        let end = start + size_of_val(self.as_slice());
        let subset_start = subset.as_ptr().addr();
        let subset_end = subset_start
            .checked_add(size_of_val(subset))
            .vortex_expect("slice_ref address overflow");
        if subset_start < start || subset_end > end {
            vortex_panic!("slice_ref subset must be contained in the buffer");
        }

        Self {
            ptr: NonNull::new(subset.as_ptr().cast_mut()).vortex_expect("slice pointer is null"),
            length: subset.len(),
            alignment,
            backing: self.backing.clone(),
        }
    }

    /// Returns the underlying bytes without copying.
    pub fn into_bytes(self) -> Bytes {
        if let Some(backing) = self.backing.as_ref()
            && let BufferBacking::Bytes(bytes) = backing.as_ref()
        {
            let offset = self.ptr.cast::<u8>().addr().get() - bytes.as_ptr().addr();
            let length = self.length * size_of::<T>();
            if offset == 0 && length == bytes.len() && Arc::strong_count(backing) == 1 {
                return match self.backing {
                    Some(backing) => match Arc::try_unwrap(backing) {
                        Ok(BufferBacking::Bytes(bytes)) => bytes,
                        _ => unreachable!(),
                    },
                    None => unreachable!(),
                };
            }
            return bytes.slice(offset..offset + length);
        }
        match self.backing {
            Some(backing) => Bytes::from_owner(BufferBytesOwner {
                ptr: self.ptr.cast(),
                length: self.length * size_of::<T>(),
                backing,
            }),
            None => Bytes::new(),
        }
    }

    /// Return the ByteBuffer for this `Buffer<T>`.
    pub fn into_byte_buffer(self) -> ByteBuffer {
        ByteBuffer {
            ptr: self.ptr.cast(),
            length: self.length * size_of::<T>(),
            alignment: self.alignment,
            backing: self.backing,
        }
    }

    /// Try to convert self into `BufferMut<T>` if there is only a single strong reference.
    pub fn try_into_mut(self) -> Result<BufferMut<T>, Self> {
        let Self {
            ptr,
            length,
            alignment,
            backing,
        } = self;
        let Some(backing) = backing else {
            return Ok(BufferMut::empty_aligned(alignment));
        };
        if !matches!(backing.as_ref(), BufferBacking::Owned(_)) {
            return Err(Self {
                ptr,
                length,
                alignment,
                backing: Some(backing),
            });
        }
        match Arc::try_unwrap(backing) {
            Ok(BufferBacking::Owned(allocation)) => {
                let offset = ptr.addr().get() - allocation.ptr().addr().get();
                let capacity = if allocation.size() == 0 {
                    0
                } else {
                    (allocation.size() - offset) / size_of::<T>()
                };
                Ok(BufferMut {
                    allocation,
                    ptr,
                    length,
                    capacity,
                    alignment,
                    _marker: Default::default(),
                })
            }
            Ok(_) => unreachable!(),
            Err(backing) => Err(Self {
                ptr,
                length,
                alignment,
                backing: Some(backing),
            }),
        }
    }

    /// Convert self into `BufferMut<T>`, cloning the data if there are multiple strong references.
    pub fn into_mut(self) -> BufferMut<T> {
        self.try_into_mut().unwrap_or_else(|buffer| {
            let allocator = buffer.allocator().clone();
            BufferMut::<T>::copy_from_aligned_in(&buffer, buffer.alignment, allocator)
        })
    }

    /// Returns whether a `Buffer<T>` is aligned to the given alignment.
    pub fn is_aligned(&self, alignment: Alignment) -> bool {
        alignment.is_ptr_aligned(self.as_ptr())
    }

    /// Return a `Buffer<T>` with the given alignment. Where possible, this will be zero-copy.
    pub fn aligned(mut self, alignment: Alignment) -> Self {
        if alignment.is_ptr_aligned(self.as_ptr()) {
            self.alignment = alignment;
            self
        } else {
            #[cfg(feature = "warn-copy")]
            {
                let bt = std::backtrace::Backtrace::capture();
                tracing::warn!(
                    "Buffer is not aligned to requested alignment {alignment}, copying: {bt}"
                )
            }
            let allocator = self.allocator().clone();
            BufferMut::copy_from_aligned_in(self, alignment, allocator).freeze()
        }
    }

    /// Return a `Buffer<T>` with the given alignment. Panics if the buffer is not aligned.
    pub fn ensure_aligned(mut self, alignment: Alignment) -> Self {
        if alignment.is_ptr_aligned(self.as_ptr()) {
            self.alignment = alignment;
            self
        } else {
            vortex_panic!("Buffer is not aligned to requested alignment {}", alignment)
        }
    }
}

impl<T> Buffer<T> {
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
    pub unsafe fn transmute<U>(self) -> Buffer<U> {
        assert_eq!(size_of::<T>(), size_of::<U>(), "Buffer type size mismatch");
        assert_eq!(
            align_of::<T>(),
            align_of::<U>(),
            "Buffer type alignment mismatch"
        );

        Buffer {
            ptr: self.ptr.cast(),
            length: self.length,
            alignment: self.alignment,
            backing: self.backing,
        }
    }
}

/// An iterator over Buffer elements.
///
/// This is an analog to the `std::slice::Iter` type.
pub struct Iter<'a, T> {
    inner: std::slice::Iter<'a, T>,
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }

    #[inline]
    fn count(self) -> usize {
        self.inner.count()
    }

    #[inline]
    fn last(self) -> Option<Self::Item> {
        self.inner.last()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.inner.nth(n)
    }
}

impl<T> ExactSizeIterator for Iter<'_, T> {
    #[inline]
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<T: Debug> Debug for Buffer<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct(&format!("Buffer<{}>", type_name::<T>()))
            .field("length", &self.length)
            .field("alignment", &self.alignment)
            .field("as_slice", &TruncatedDebug(self.as_slice()))
            .finish()
    }
}

impl<T> Deref for Buffer<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T> AsRef<[T]> for Buffer<T> {
    #[inline]
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> FromIterator<T> for Buffer<T> {
    #[inline]
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        BufferMut::from_iter(iter).freeze()
    }
}

// Helper struct that preserves drop glue for non-native Vec elements.
#[repr(transparent)]
struct Wrapper<T>(Vec<T>);

impl<T: Send + Sync + 'static> crate::BufferOwner for Wrapper<T> {
    fn as_ptr(&self) -> *const u8 {
        self.0.as_ptr().cast()
    }

    fn len(&self) -> usize {
        self.0.len() * size_of::<T>()
    }
}

impl<T> From<Vec<T>> for Buffer<T>
where
    T: Send + Sync + 'static,
{
    fn from(value: Vec<T>) -> Self {
        let length = value.len();
        let alignment = Alignment::of::<T>();
        if std::mem::needs_drop::<T>() {
            Self::from_owner(Wrapper(value), alignment)
        } else {
            Self::from_allocation(Allocation::from_vec(value), 0, length, alignment)
        }
    }
}

impl From<Bytes> for ByteBuffer {
    fn from(bytes: Bytes) -> Self {
        Self::from_bytes(bytes, Alignment::of::<u8>())
    }
}

impl Buf for ByteBuffer {
    #[inline]
    fn remaining(&self) -> usize {
        self.len()
    }

    #[inline]
    fn chunk(&self) -> &[u8] {
        self.as_slice()
    }

    #[inline]
    fn advance(&mut self, cnt: usize) {
        if !self.alignment.is_offset_aligned(cnt) {
            vortex_panic!(
                "Cannot advance buffer by {} items, resulting alignment is not {}",
                cnt,
                self.alignment
            );
        }
        assert!(cnt <= self.length, "cannot advance past the buffer length");
        // SAFETY: cnt is within the initialized byte range.
        self.ptr = unsafe { self.ptr.add(cnt) };
        self.length -= cnt;
    }
}

struct BufferBytesOwner {
    ptr: NonNull<u8>,
    length: usize,
    backing: Arc<BufferBacking>,
}

// SAFETY: the owner exposes immutable initialized bytes and keeps their backing live.
unsafe impl Send for BufferBytesOwner {}
unsafe impl Sync for BufferBytesOwner {}

impl AsRef<[u8]> for BufferBytesOwner {
    fn as_ref(&self) -> &[u8] {
        let _ = &self.backing;
        // SAFETY: ptr and length came from a live Buffer.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.length) }
    }
}

fn empty_ptr<T>() -> NonNull<T> {
    let addr = 1usize << (usize::BITS - 1);
    NonNull::new(std::ptr::without_provenance_mut(addr)).vortex_expect("empty pointer is non-null")
}

/// Owned iterator over a [`Buffer`].
pub struct BufferIterator<T: Copy> {
    // Keep the buffer alive for the duration of the iteration.
    _buffer: Buffer<T>,
    ptr: *const T,
    end: *const T,
}

// SAFETY: `BufferIterator` is a `Buffer<T>` plus two cursors into it, so it can safely be
// `Send`/`Sync` exactly when `Buffer<T>` is. Same bounds as `std::vec::IntoIter`.
unsafe impl<T: Copy + Send> Send for BufferIterator<T> {}
unsafe impl<T: Copy + Sync> Sync for BufferIterator<T> {}

impl<T: Copy> Iterator for BufferIterator<T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.ptr == self.end {
            None
        } else {
            // SAFETY: ptr is within the buffer and has not reached end.
            let value = unsafe { self.ptr.read() };
            self.ptr = unsafe { self.ptr.add(1) };
            Some(value)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = unsafe { self.end.offset_from(self.ptr) } as usize;
        (remaining, Some(remaining))
    }
}

impl<T: Copy> ExactSizeIterator for BufferIterator<T> {}

impl<T: Copy> IntoIterator for Buffer<T> {
    type Item = T;
    type IntoIter = BufferIterator<T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        let ptr = self.as_slice().as_ptr();
        let end = unsafe { ptr.add(self.len()) };
        BufferIterator {
            _buffer: self,
            ptr,
            end,
        }
    }
}

impl<T> From<BufferMut<T>> for Buffer<T> {
    #[inline]
    fn from(value: BufferMut<T>) -> Self {
        value.freeze()
    }
}

#[cfg(test)]
mod test {
    use std::mem::align_of;
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use bytes::Buf;
    use bytes::Bytes;

    use crate::Alignment;
    use crate::Buffer;
    use crate::BufferBacking;
    use crate::ByteBuffer;
    use crate::buffer;

    #[test]
    fn align() {
        let buf = buffer![0u8, 1, 2];
        let aligned = buf.aligned(Alignment::new(32));
        assert_eq!(aligned.alignment(), Alignment::new(32));
        assert_eq!(aligned.as_slice(), &[0, 1, 2]);
    }

    #[test]
    fn buffer_iterator_send_sync() {
        fn assert_send_sync<T: Send + Sync>(_: &T) {}

        let mut iter = buffer![0i32, 1, 2, 3].into_iter();
        assert_send_sync(&iter);
        iter.next();
        let remaining: Vec<i32> = std::thread::spawn(move || iter.collect()).join().unwrap();
        assert_eq!(remaining, vec![1, 2, 3]);
    }

    #[test]
    fn slice() {
        let buf = buffer![0, 1, 2, 3, 4];
        assert_eq!(buf.slice(1..3).as_slice(), &[1, 2]);
        assert_eq!(buf.slice(1..=3).as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn slice_unaligned() {
        let buf = buffer![0i32, 1, 2, 3, 4].into_byte_buffer();
        // With a regular slice, this would panic. See [`slice_bad_alignment`].
        let sliced = buf.slice_unaligned(1..2);
        // Verify the slice has the expected length (1 byte from index 1 to 2).
        assert_eq!(sliced.len(), 1);
        // The original buffer has i32 values [0, 1, 2, 3, 4].
        // In little-endian bytes, 0i32 = [0, 0, 0, 0], so byte at index 1 is 0.
        assert_eq!(sliced.as_slice(), &[0]);
    }

    #[test]
    #[should_panic]
    fn slice_bad_alignment() {
        let buf = buffer![0i32, 1, 2, 3, 4].into_byte_buffer();
        // We should only be able to slice this buffer on 4-byte (i32) boundaries.
        buf.slice(1..2);
    }

    #[test]
    fn bytes_buf() {
        let mut buf = ByteBuffer::copy_from("helloworld".as_bytes());
        assert_eq!(buf.remaining(), 10);
        assert_eq!(buf.chunk(), b"helloworld");

        buf.advance(5);
        assert_eq!(buf.remaining(), 5);
        assert_eq!(buf.as_slice(), b"world");
        assert_eq!(buf.chunk(), b"world");
    }

    #[test]
    fn buffer_zeroed() {
        const LEN: usize = 17;

        let buf = Buffer::<u32>::zeroed(LEN);

        assert!(buf.is_aligned(Alignment::of::<u32>()));
        assert_eq!(buf.as_slice(), &[0; LEN]);
    }

    #[test]
    fn buffer_zeroed_aligned() {
        const LEN: usize = 17;
        let alignment = Alignment::new(64);

        let buf = Buffer::<u32>::zeroed_aligned(LEN, alignment);

        assert!(buf.is_aligned(alignment));
        assert_eq!(buf.as_slice(), &[0; LEN]);
    }

    #[test]
    fn copy_from_over_aligns_to_default() {
        let values = [1u32, 2, 3];
        let buf = Buffer::<u32>::copy_from(values);

        // The buffer reports the scalar type's alignment, ...
        assert_eq!(buf.alignment(), Alignment::of::<u32>());
        // ... but the underlying allocation is over-aligned to DEFAULT_ALIGNMENT.
        assert!(buf.is_aligned(Alignment::DEFAULT_ALIGNMENT));
        assert_eq!(buf.as_slice(), &values);
    }

    #[test]
    fn zeroed_over_aligns_to_default() {
        const LEN: usize = 17;

        let buf = Buffer::<u32>::zeroed(LEN);

        assert_eq!(buf.alignment(), Alignment::of::<u32>());
        assert!(buf.is_aligned(Alignment::DEFAULT_ALIGNMENT));
        assert_eq!(buf.as_slice(), &[0; LEN]);
    }

    #[test]
    fn from_vec() {
        let vec = vec![1, 2, 3, 4, 5];
        let buff = Buffer::from(vec.clone());
        assert!(buff.is_aligned(Alignment::of::<i32>()));
        assert_eq!(vec, buff.as_ref());
    }

    #[test]
    fn from_vec_adopts_allocation() {
        let mut vec = Vec::with_capacity(16);
        vec.extend([1u32, 2, 3, 4, 5]);
        let ptr = vec.as_ptr();
        let capacity = vec.capacity();

        let buffer = Buffer::from(vec);
        assert_eq!(buffer.as_ptr(), ptr);

        let Ok(mut buffer) = buffer.try_into_mut() else {
            panic!("Vec-backed buffer should be uniquely owned")
        };
        assert_eq!(buffer.capacity(), capacity);
        assert_eq!(buffer.allocation.alignment(), align_of::<u32>());

        buffer.extend(6..=32);
        assert_eq!(buffer.as_slice(), (1..=32).collect::<Vec<_>>());
        assert_eq!(buffer.allocation.alignment(), align_of::<u32>());
    }

    #[test]
    fn bytes_round_trip_reuses_owner() {
        let bytes = Bytes::from_static(&[1, 2, 3, 4]);
        let ptr = bytes.as_ptr();

        let buffer = ByteBuffer::from(bytes);
        assert!(matches!(
            buffer.backing.as_deref(),
            Some(BufferBacking::Bytes(_))
        ));
        let bytes = buffer.into_bytes();

        assert_eq!(bytes.as_ptr(), ptr);
        assert_eq!(bytes.as_ref(), &[1, 2, 3, 4]);
    }

    #[test]
    fn external_try_into_mut_preserves_backing() {
        let buffer = ByteBuffer::from(Bytes::from_static(&[1, 2, 3, 4]));
        let Some(original_backing) = buffer.backing.as_ref() else {
            panic!("external buffer has no backing")
        };
        let backing = Arc::as_ptr(original_backing);

        let Err(buffer) = buffer.try_into_mut() else {
            panic!("external buffer became mutable")
        };

        let Some(new_backing) = buffer.backing.as_ref() else {
            panic!("external buffer has no backing")
        };
        assert_eq!(Arc::as_ptr(new_backing), backing);
    }

    #[test]
    fn from_u8_vec_preserves_capacity() {
        let mut vec = Vec::with_capacity(16);
        vec.extend([1u8, 2, 3]);

        let buffer = Buffer::from(vec);
        let Ok(buffer) = buffer.try_into_mut() else {
            panic!("Vec-backed buffer should be uniquely owned")
        };
        assert_eq!(buffer.capacity(), 16);
    }

    #[test]
    fn sliced_buffer_into_mut_has_safe_capacity() {
        let mut original = crate::BufferMut::with_capacity(128);
        original.extend(0u32..100);
        let original = original.freeze();
        let sliced = original.slice(64..96);
        drop(original);

        let Ok(mut sliced) = sliced.try_into_mut() else {
            panic!("uniquely owned slice should become mutable")
        };
        let capacity = sliced.capacity();
        sliced.push_n(0, capacity - sliced.len());
        assert_eq!(sliced.len(), capacity);
    }

    #[test]
    fn from_vec_preserves_drop_glue() {
        struct DropValue(Arc<AtomicUsize>);

        impl Drop for DropValue {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }

        let drops = Arc::new(AtomicUsize::new(0));
        let values = (0..3)
            .map(|_| DropValue(Arc::clone(&drops)))
            .collect::<Vec<_>>();
        let buffer = Buffer::from(values);

        assert_eq!(drops.load(Ordering::Relaxed), 0);
        drop(buffer);
        assert_eq!(drops.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn empty_aligned_max_alignment() {
        // Empty buffers are backed by a static and must satisfy any valid alignment.
        let buf = Buffer::<u8>::empty_aligned(Alignment::MAX);
        assert!(buf.is_empty());
        assert!(buf.is_aligned(Alignment::MAX));
    }

    #[test]
    fn empty_has_no_backing() {
        assert!(Buffer::<u8>::empty().backing.is_none());
    }

    #[test]
    fn empty_slice_preserves_alignment() {
        let buf = Buffer::<u64>::zeroed_aligned(8, Alignment::new(64));
        let sliced = buf.slice(0..0);
        assert!(sliced.is_empty());
        assert_eq!(sliced.alignment(), Alignment::new(64));
        assert!(sliced.is_aligned(Alignment::new(64)));
    }

    #[test]
    fn empty_into_mut_preserves_alignment() {
        let buf = Buffer::<u8>::empty_aligned(Alignment::new(64));
        let buf_mut = buf.into_mut();
        assert_eq!(buf_mut.alignment(), Alignment::new(64));
        assert!(buf_mut.is_empty());
    }

    #[test]
    fn test_slice_unaligned_end_pos() {
        let data = vec![0u8; 2];
        // Overalign the u8 vector.
        let aligned_buffer = Buffer::copy_from_aligned(&data, Alignment::new(8));
        // Previously, `Buffer::slice` incorrectly asserted that the end position
        // must be aligned. That assertion has been removed such that the end
        // position can be arbitrary and only the beginning of the slice needs
        // to be aligned.
        aligned_buffer.slice(0..1);
    }

    #[test]
    fn test_empty_equality() {
        let a = Buffer::<u16>::empty();
        let b = Buffer::<u16>::empty();

        assert_eq!(a, b);
    }
}
