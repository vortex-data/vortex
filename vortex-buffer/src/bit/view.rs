// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Bound;
use std::ops::RangeBounds;

use crate::BitBuffer;
use crate::BitBufferMeta;
use crate::BitBufferMut;
use crate::ByteBuffer;
use crate::bit::BitChunks;
use crate::bit::BitIndexIterator;
use crate::bit::BitIterator;
use crate::bit::BitSliceIterator;
use crate::bit::UnalignedBitChunk;
use crate::bit::buf_mut::fill_bits;
use crate::bit::count_ones::count_ones;
use crate::bit::get_bit_unchecked;
use crate::bit::select::bit_select;
use crate::bit::set_bit_unchecked;
use crate::bit::unset_bit_unchecked;

/// Resolve `start..end` bounds against a logical length, panicking on invalid ranges.
#[inline]
fn resolve_range(range: impl RangeBounds<usize>, len: usize) -> (usize, usize) {
    let start = match range.start_bound() {
        Bound::Included(&s) => s,
        Bound::Excluded(&s) => s + 1,
        Bound::Unbounded => 0,
    };
    let end = match range.end_bound() {
        Bound::Included(&e) => e + 1,
        Bound::Excluded(&e) => e,
        Bound::Unbounded => len,
    };

    assert!(start <= end);
    assert!(start <= len);
    assert!(end <= len);
    (start, end)
}

/// Normalize a byte slice and bit offset so that the returned offset is `< 8`.
#[inline]
fn normalize(buffer: &[u8], offset: usize) -> (&[u8], usize) {
    let byte_offset = offset / 8;
    (&buffer[byte_offset..], offset % 8)
}

/// An immutable, borrowed view over a packed bitset.
///
/// This is the borrowing analogue of [`BitBuffer`]: it stores a byte slice together with a bit
/// `offset` (always `< 8`) and a logical bit `len`, without owning or reference-counting the
/// backing allocation. Use it to read a bitset without cloning the underlying [`ByteBuffer`].
#[derive(Debug, Clone, Copy)]
pub struct BitBufferView<'a> {
    buffer: &'a [u8],
    offset: usize,
    len: usize,
}

impl<'a> BitBufferView<'a> {
    /// Create a new view over `buffer` with `len` bits, starting at bit zero.
    ///
    /// Panics if the buffer is not large enough to hold `len` bits.
    pub fn new(buffer: &'a [u8], len: usize) -> Self {
        Self::new_with_offset(buffer, len, 0)
    }

    /// Create a new view over `buffer` with `len` bits, starting at the given bit `offset`.
    ///
    /// Panics if the buffer is not large enough to hold `len` bits after the offset.
    pub fn new_with_offset(buffer: &'a [u8], len: usize, offset: usize) -> Self {
        assert!(
            len.saturating_add(offset) <= buffer.len().saturating_mul(8),
            "provided slice (len={}) not large enough to back BitBufferView with offset {offset} len {len}",
            buffer.len()
        );

        let (buffer, offset) = normalize(buffer, offset);
        Self {
            buffer,
            offset,
            len,
        }
    }

    /// Create a new view over `buffer` described by `meta`.
    pub fn from_meta(buffer: &'a [u8], meta: BitBufferMeta) -> Self {
        Self::new_with_offset(buffer, meta.len(), meta.offset())
    }

    /// Returns the [`BitBufferMeta`] (offset and length) describing this view.
    pub fn meta(&self) -> BitBufferMeta {
        BitBufferMeta::new(self.offset, self.len)
    }

    /// Get the logical length of this view in bits.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the view is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Offset of the start of the view in bits. Always `< 8`.
    #[inline(always)]
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Get a reference to the underlying byte slice.
    #[inline(always)]
    pub fn inner(&self) -> &'a [u8] {
        self.buffer
    }

    /// Retrieve the value at the given index.
    ///
    /// Panics if the index is out of bounds.
    #[inline]
    pub fn value(&self, index: usize) -> bool {
        assert!(index < self.len);
        // SAFETY: checked by assertion
        unsafe { self.value_unchecked(index) }
    }

    /// Retrieve the value at the given index without bounds checking.
    ///
    /// # Safety
    ///
    /// Caller must ensure that `index` is within the range of the view.
    #[inline]
    pub unsafe fn value_unchecked(&self, index: usize) -> bool {
        unsafe { get_bit_unchecked(self.buffer.as_ptr(), index + self.offset) }
    }

    /// Create a new view over the range `[start, end)` of this view.
    ///
    /// Panics if the slice would extend beyond the end of the view.
    pub fn slice(&self, range: impl RangeBounds<usize>) -> BitBufferView<'a> {
        let (start, end) = resolve_range(range, self.len);
        BitBufferView::new_with_offset(self.buffer, end - start, self.offset + start)
    }

    /// Access chunks of the buffer aligned to an 8 byte boundary as
    /// `[prefix, <full chunks>, suffix]`.
    pub fn unaligned_chunks(&self) -> UnalignedBitChunk<'a> {
        UnalignedBitChunk::new(self.buffer, self.offset, self.len)
    }

    /// Access chunks of the underlying buffer as 8 byte chunks with a final trailer.
    pub fn chunks(&self) -> BitChunks<'a> {
        BitChunks::new(self.buffer, self.offset, self.len)
    }

    /// Get the number of set bits in the view.
    pub fn true_count(&self) -> usize {
        count_ones(self.buffer, self.offset, self.len)
    }

    /// Get the number of unset bits in the view.
    pub fn false_count(&self) -> usize {
        self.len - self.true_count()
    }

    /// Returns the position of the `nth` set bit (0-indexed), or `None` if out of range.
    pub fn select(&self, nth: usize) -> Option<usize> {
        bit_select(self.buffer, self.offset, self.len, nth)
    }

    /// Iterator over bits in the view.
    pub fn iter(&self) -> BitIterator<'a> {
        BitIterator::new(self.buffer, self.offset, self.len)
    }

    /// Iterator over set indices of the underlying buffer.
    pub fn set_indices(&self) -> BitIndexIterator<'a> {
        BitIndexIterator::new(self.buffer, self.offset, self.len)
    }

    /// Iterator over set slices of the underlying buffer.
    pub fn set_slices(&self) -> BitSliceIterator<'a> {
        BitSliceIterator::new(self.buffer, self.offset, self.len)
    }

    /// Copy this view into an owned [`BitBuffer`].
    pub fn to_bit_buffer(&self) -> BitBuffer {
        let bytes = (self.offset + self.len).div_ceil(8);
        BitBuffer::new_with_offset(
            ByteBuffer::copy_from(&self.buffer[..bytes]),
            self.len,
            self.offset,
        )
    }
}

impl<'a> IntoIterator for BitBufferView<'a> {
    type Item = bool;
    type IntoIter = BitIterator<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl PartialEq for BitBufferView<'_> {
    fn eq(&self, other: &Self) -> bool {
        if self.len != other.len {
            return false;
        }

        self.chunks()
            .iter_padded()
            .zip(other.chunks().iter_padded())
            .all(|(a, b)| a == b)
    }
}

impl Eq for BitBufferView<'_> {}

/// A mutable, borrowed view over a packed bitset.
///
/// This is the borrowing analogue of [`BitBufferMut`]: it stores a mutable byte slice together
/// with a bit `offset` (always `< 8`) and a logical bit `len`. Unlike [`BitBufferMut`] it cannot
/// grow or reallocate, so it only supports in-place reads and writes (such as
/// [`set`](Self::set), [`unset`](Self::unset), and [`fill_range`](Self::fill_range)).
#[derive(Debug)]
pub struct BitBufferMutView<'a> {
    buffer: &'a mut [u8],
    offset: usize,
    len: usize,
}

impl<'a> BitBufferMutView<'a> {
    /// Create a new mutable view over `buffer` with `len` bits, starting at bit zero.
    ///
    /// Panics if the buffer is not large enough to hold `len` bits.
    pub fn new(buffer: &'a mut [u8], len: usize) -> Self {
        Self::new_with_offset(buffer, len, 0)
    }

    /// Create a new mutable view over `buffer` with `len` bits, starting at bit `offset`.
    ///
    /// Panics if the buffer is not large enough to hold `len` bits after the offset.
    pub fn new_with_offset(buffer: &'a mut [u8], len: usize, offset: usize) -> Self {
        assert!(
            len.saturating_add(offset) <= buffer.len().saturating_mul(8),
            "provided slice (len={}) not large enough to back BitBufferMutView with offset {offset} len {len}",
            buffer.len()
        );

        let byte_offset = offset / 8;
        let offset = offset % 8;
        Self {
            buffer: &mut buffer[byte_offset..],
            offset,
            len,
        }
    }

    /// Borrow this mutable view as an immutable [`BitBufferView`].
    #[inline]
    pub fn as_view(&self) -> BitBufferView<'_> {
        BitBufferView {
            buffer: self.buffer,
            offset: self.offset,
            len: self.len,
        }
    }

    /// Get the logical length of this view in bits.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the view is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Offset of the start of the view in bits. Always `< 8`.
    #[inline(always)]
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Get the underlying bytes as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        self.buffer
    }

    /// Get the underlying bytes as a mutable slice.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        self.buffer
    }

    /// Retrieve the value at the given index.
    ///
    /// Panics if the index is out of bounds.
    #[inline]
    pub fn value(&self, index: usize) -> bool {
        assert!(index < self.len);
        // SAFETY: checked by assertion
        unsafe { self.value_unchecked(index) }
    }

    /// Retrieve the value at the given index without bounds checking.
    ///
    /// # Safety
    ///
    /// Caller must ensure that `index` is within the range of the view.
    #[inline]
    pub unsafe fn value_unchecked(&self, index: usize) -> bool {
        unsafe { get_bit_unchecked(self.buffer.as_ptr(), index + self.offset) }
    }

    /// Get the number of set bits in the view.
    pub fn true_count(&self) -> usize {
        self.as_view().true_count()
    }

    /// Get the number of unset bits in the view.
    pub fn false_count(&self) -> usize {
        self.as_view().false_count()
    }

    /// Iterator over bits in the view.
    pub fn iter(&self) -> BitIterator<'_> {
        self.as_view().iter()
    }

    /// Set the bit at `index` to the given boolean value.
    ///
    /// Panics if `index` exceeds the view length.
    pub fn set_to(&mut self, index: usize, value: bool) {
        if value {
            self.set(index);
        } else {
            self.unset(index);
        }
    }

    /// Set the bit at `index` to the given boolean value without bounds checking.
    ///
    /// # Safety
    ///
    /// Caller must ensure that `index` is within the range of the view.
    pub unsafe fn set_to_unchecked(&mut self, index: usize, value: bool) {
        if value {
            // SAFETY: checked by caller
            unsafe { self.set_unchecked(index) }
        } else {
            // SAFETY: checked by caller
            unsafe { self.unset_unchecked(index) }
        }
    }

    /// Set the bit at `index` to `true`.
    ///
    /// Panics if `index` exceeds the view length.
    pub fn set(&mut self, index: usize) {
        assert!(index < self.len, "index {index} exceeds len {}", self.len);
        // SAFETY: checked by assertion
        unsafe { self.set_unchecked(index) };
    }

    /// Set the bit at `index` to `false`.
    ///
    /// Panics if `index` exceeds the view length.
    pub fn unset(&mut self, index: usize) {
        assert!(index < self.len, "index {index} exceeds len {}", self.len);
        // SAFETY: checked by assertion
        unsafe { self.unset_unchecked(index) };
    }

    /// Set the bit at `index` to `true` without bounds checking.
    ///
    /// # Safety
    ///
    /// Caller must ensure that `index` is within the range of the view.
    #[inline]
    pub unsafe fn set_unchecked(&mut self, index: usize) {
        // SAFETY: checked by caller
        unsafe { set_bit_unchecked(self.buffer.as_mut_ptr(), self.offset + index) }
    }

    /// Set the bit at `index` to `false` without bounds checking.
    ///
    /// # Safety
    ///
    /// Caller must ensure that `index` is within the range of the view.
    #[inline]
    pub unsafe fn unset_unchecked(&mut self, index: usize) {
        // SAFETY: checked by caller
        unsafe { unset_bit_unchecked(self.buffer.as_mut_ptr(), self.offset + index) }
    }

    /// Sets all bits in the range `[start, end)` to `value`.
    ///
    /// Panics if `end > self.len()` or `start > end`.
    #[inline(always)]
    pub fn fill_range(&mut self, start: usize, end: usize, value: bool) {
        assert!(end <= self.len, "end {end} exceeds len {}", self.len);
        assert!(start <= end, "start {start} exceeds end {end}");
        // SAFETY: assertions guarantee start <= end <= self.len.
        unsafe { self.fill_range_unchecked(start, end, value) }
    }

    /// Sets all bits in the range `[start, end)` to `value` without bounds checking.
    ///
    /// # Safety
    ///
    /// Caller must ensure that `start <= end <= self.len()`.
    #[inline(always)]
    pub unsafe fn fill_range_unchecked(&mut self, start: usize, end: usize, value: bool) {
        fill_bits(self.buffer, self.offset + start, self.offset + end, value);
    }

    /// Copy this view into an owned [`BitBuffer`].
    pub fn to_bit_buffer(&self) -> BitBuffer {
        self.as_view().to_bit_buffer()
    }
}

impl BitBuffer {
    /// Borrow this buffer as a [`BitBufferView`] without cloning the backing allocation.
    #[inline]
    pub fn as_view(&self) -> BitBufferView<'_> {
        BitBufferView {
            buffer: self.inner().as_slice(),
            offset: self.offset(),
            len: self.len(),
        }
    }
}

impl BitBufferMut {
    /// Borrow this buffer as an immutable [`BitBufferView`].
    #[inline]
    pub fn as_view(&self) -> BitBufferView<'_> {
        BitBufferView {
            buffer: self.as_slice(),
            offset: self.offset(),
            len: self.len(),
        }
    }

    /// Borrow this buffer as a [`BitBufferMutView`].
    #[inline]
    pub fn as_mut_view(&mut self) -> BitBufferMutView<'_> {
        let offset = self.offset();
        let len = self.len();
        BitBufferMutView {
            buffer: self.as_mut_slice(),
            offset,
            len,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::BitBuffer;
    use crate::BitBufferMut;
    use crate::bitbuffer;

    #[test]
    fn view_reads_match_buffer() {
        let buffer = bitbuffer![true, false, true, true, false, true, false, false];
        let view = buffer.as_view();

        assert_eq!(view.len(), buffer.len());
        assert_eq!(view.true_count(), buffer.true_count());
        assert_eq!(view.false_count(), buffer.false_count());
        for i in 0..buffer.len() {
            assert_eq!(view.value(i), buffer.value(i));
        }
        assert_eq!(
            view.iter().collect::<Vec<_>>(),
            buffer.iter().collect::<Vec<_>>()
        );
    }

    #[test]
    fn view_slice_preserves_offset() {
        let buffer = BitBuffer::new_set(20);
        let sliced = buffer.slice(5..17);
        let view = buffer.as_view().slice(5..17);

        assert_eq!(view.len(), sliced.len());
        assert_eq!(view.true_count(), sliced.true_count());
        assert_eq!(view.to_bit_buffer(), sliced);
    }

    #[test]
    fn view_offset_buffer() {
        let buffer = BitBuffer::new_set(64).slice(3..40);
        let view = buffer.as_view();
        assert_eq!(view.offset(), buffer.offset());
        assert_eq!(view.len(), buffer.len());
        assert_eq!(view.to_bit_buffer(), buffer);
    }

    #[test]
    fn mut_view_set_unset() {
        let mut buffer = BitBufferMut::new_unset(16);
        {
            let mut view = buffer.as_mut_view();
            view.set(0);
            view.set(15);
            view.set_to(7, true);
            view.fill_range(2, 5, true);
            assert!(view.value(0));
            assert_eq!(view.true_count(), 6);
            view.unset(0);
        }
        let frozen = buffer.freeze();
        assert!(!frozen.value(0));
        assert!(frozen.value(2));
        assert!(frozen.value(4));
        assert!(frozen.value(7));
        assert!(frozen.value(15));
        assert_eq!(frozen.true_count(), 5);
    }

    #[test]
    fn mut_view_with_offset() {
        let mut buffer = BitBufferMut::from_buffer(crate::buffer_mut![0u8; 4], 3, 20);
        {
            let mut view = buffer.as_mut_view();
            assert_eq!(view.offset(), 3);
            view.fill_range(0, 20, true);
        }
        let frozen = buffer.freeze();
        assert_eq!(frozen.true_count(), 20);
    }
}
