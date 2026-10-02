// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A [`VarBinViewArray`] with its data buffers resolved for per-row access.

use crate::arrays::VarBinViewArray;
use crate::arrays::varbinview::BinaryView;

/// A canonical [`VarBinViewArray`] with its data buffers resolved to slices, so reading a row is
/// a slice index rather than a buffer-handle lookup.
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

    /// The data buffers, indexed by a view's `buffer_index`.
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
    /// `index` must be less than [`len`](Self::len).
    #[inline]
    pub unsafe fn view_unchecked(&self, index: usize) -> &'a BinaryView {
        // SAFETY: caller guarantees index < len.
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
}

#[cfg(test)]
mod tests;
