// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A [`VarBinViewArray`] with its data buffers resolved once for per-row access.

use crate::arrays::VarBinViewArray;
use crate::arrays::varbinview::BinaryView;

/// A canonical [`VarBinViewArray`] with its data buffers resolved: the view structs alongside a
/// borrowed slice of every data buffer.
///
/// A view holds values up to twelve bytes inline and spills longer ones to a data buffer, so
/// reading a value means indexing the array's buffer list. Resolving that list once turns each
/// read into a slice index, rather than the buffer-handle lookup a per-element accessor such as
/// [`VarBinViewData::bytes_at`](crate::arrays::varbinview::VarBinViewData::bytes_at) repeats per
/// row.
///
/// Kernels that compare or match values should still prefer the
/// [comparison words](BinaryView#comparison-words) a view carries, and resolve the bytes only
/// once those leave the answer open: for a value in a data buffer, that read is a scattered one.
pub struct ResolvedViews<'a> {
    views: &'a [BinaryView],
    buffers: Vec<&'a [u8]>,
}

impl<'a> ResolvedViews<'a> {
    /// Resolve the data buffers of `array`.
    pub fn new(array: &'a VarBinViewArray) -> Self {
        Self {
            views: array.views(),
            buffers: (0..array.data_buffers().len())
                .map(|idx| array.buffer(idx).as_slice())
                .collect(),
        }
    }

    /// The array's views, one per row.
    #[inline]
    pub fn views(&self) -> &'a [BinaryView] {
        self.views
    }

    /// The resolved data buffers, in the order a view's `buffer_index` addresses them.
    ///
    /// Pass these to [`BinaryView::bytes`] to read a value from a view held elsewhere, such as a
    /// sub-slice of [`views`](Self::views).
    #[inline]
    pub fn buffers(&self) -> &[&'a [u8]] {
        &self.buffers
    }

    /// The number of rows.
    #[inline]
    pub fn len(&self) -> usize {
        self.views.len()
    }

    /// Whether the array has no rows.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.views.is_empty()
    }

    /// The view at `index` without a bounds check.
    ///
    /// # Safety
    ///
    /// `index` must be strictly less than [`len`](Self::len).
    #[inline]
    pub unsafe fn view_unchecked(&self, index: usize) -> &'a BinaryView {
        // SAFETY: caller guarantees index < self.views.len().
        unsafe { self.views.get_unchecked(index) }
    }

    /// The bytes of the value at `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    #[inline]
    pub fn bytes(&self, index: usize) -> &'a [u8] {
        self.view_bytes(&self.views[index])
    }

    /// The bytes of `view`, which must belong to this array.
    #[inline]
    pub fn view_bytes(&self, view: &'a BinaryView) -> &'a [u8] {
        view.bytes(&self.buffers)
    }

    /// Whether every value is pure ASCII, including values under a null.
    pub fn is_ascii(&self) -> bool {
        self.views
            .iter()
            .all(|view| self.view_bytes(view).is_ascii())
    }

    /// The last `suffix_len` bytes of `view`, which must belong to this array.
    ///
    /// # Safety
    ///
    /// `suffix_len` must be at most `view.len()`. Buffer bounds are guaranteed by
    /// [`VarBinViewArray::validate`], which checks every view against its data buffer at
    /// construction.
    #[inline]
    pub unsafe fn suffix_bytes_unchecked(
        &self,
        view: &'a BinaryView,
        suffix_len: usize,
    ) -> &'a [u8] {
        let len = view.len() as usize;
        if view.is_inlined() {
            // SAFETY: inlined values hold `len <= 12` value bytes, and the caller
            // guarantees `suffix_len <= len`.
            unsafe { view.as_inlined().value().get_unchecked(len - suffix_len..) }
        } else {
            let view = view.as_view();
            let end = view.offset as usize + len;
            // SAFETY: validated views reference `buffer_index < buffers.len()` and bytes
            // `offset..offset + len` within that buffer.
            unsafe {
                self.buffers
                    .get_unchecked(view.buffer_index as usize)
                    .get_unchecked(end - suffix_len..end)
            }
        }
    }
}

#[cfg(test)]
mod tests;
