// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[allow(unused)] // Used for documentation.
use crate::arrays::{ListArray, ListViewArray};

/// The "shape" of the a [`ListViewArray`]'s data.
///
/// ### Why do we need this?
///
/// In comparison to [`ListArray`], [`ListViewArray`] has much more relaxed invariants:
///
/// - Offsets are allowed to stored be out-of-order (since size is not implicit via
///   `offsets[i+1] - offsets[i]` and is instead stored separately)
/// - Views (which we define the a tuple `(offset, size)` denoting a range of elements) are allowed
///   to overlap with each other / share elements in the `elements` child array
/// - Views do not have to perfectly cover the `elements` child array, and there can be gaps of
///   "unused" elements that are not referenced by any views
///
/// This allows for more flexibility in how we store data, but at the same time it restricts what
/// operations we are able to do.
///
/// For example, we cannot do a constant-time slice of a [`ListViewArray`] unless we know that:
/// 1. The offsets are in sorted order
/// 2. There are no overlaps caused sizes larger than the gaps between offsets.
///
/// This type keeps track of information that allows us to perform operations like slicing and
/// rebuilding much more efficiently.
///
/// ### Zero-copy to [`ListArray`]
///
/// If all of the flags in this struct are set to `true`, then we know that the corresponding
/// [`ListViewArray`] is "zero-copyable" to a [`ListArray`]. Note that technically it can never be
/// truly zero-copyable since we must add a single `offset` to get the correct `n+1` offsets that
/// [`ListArray`] needs, but the data is zero-copyable in spirit.
///
/// This is not only helpful when we want to get a [`ListArray`] from a [`ListViewArray`], but we
/// also can know that operations like slicing can be done efficiently.
///
/// _Note that in the actual `slice` implementation of [`ListViewArray`], we do not actually perform
/// a slice of the underlying `elements` array, and we defer that operation to when the array gets
/// rebuilt via [`ListViewArray::rebuild`].
///
/// ### Nulls
///
/// We do not consider null views as part of the "shape" of a `ListView`. In other words, even if a
/// view is null, the corresponding offset must still be in order with the rest of the `offsets`
/// array for us to consider `has_sorted_offsets` to be `true`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ListViewShape {
    /// A flag indicating if the `offsets` array is sorted (not strictly sorted).
    ///
    /// Even if a view is defined as null in the validity array, the offsets must still be in sorted
    /// order.
    has_sorted_offsets: bool,

    /// A flag indicating that no views overlap / share elements with each other.
    has_no_overlaps: bool,

    /// A flag indicating that there are no unused elements between views (ignoring any leading or
    /// trailing unused elements).
    has_no_gaps: bool,
}

impl ListViewShape {
    /// Creates a new `ListViewShape`.
    pub fn new(has_sorted_offsets: bool, has_no_overlaps: bool, has_no_gaps: bool) -> Self {
        Self {
            has_sorted_offsets,
            has_no_overlaps,
            has_no_gaps,
        }
    }

    /// Checks if the shape of the [`ListViewArray`] allows for zero-copying to a [`ListArray`].
    pub fn is_zero_copy_to_list(&self) -> bool {
        self.has_sorted_offsets && self.has_no_overlaps && self.has_no_gaps
    }

    /// Creates a `ListViewShape` that indicates that the corresponding [`ListViewArray`] is
    /// zero-copyable to [`ListArray`], as well as that it can be constant-time sliced.
    pub fn as_zero_copy_to_list() -> Self {
        Self::default()
            .with_sorted_offsets(true)
            .with_no_overlaps(true)
            .with_no_gaps(true)
    }

    /// Returns a new `ListViewShape` with the sorted offsets flag updated.
    pub fn with_sorted_offsets(self, has_sorted_offsets: bool) -> Self {
        Self {
            has_sorted_offsets,
            ..self
        }
    }

    /// Returns a new `ListViewShape` with the no overlaps flag updated.
    pub fn with_no_overlaps(self, has_no_overlaps: bool) -> Self {
        Self {
            has_no_overlaps,
            ..self
        }
    }

    /// Returns a new `ListViewShape` with the no gaps flag updated.
    pub fn with_no_gaps(self, has_no_gaps: bool) -> Self {
        Self {
            has_no_gaps,
            ..self
        }
    }

    /// Returns whether the offsets are sorted.
    pub fn has_sorted_offsets(&self) -> bool {
        self.has_sorted_offsets
    }

    /// Returns whether views have no overlapping elements.
    pub fn has_no_overlaps(&self) -> bool {
        self.has_no_overlaps
    }

    /// Returns whether there are no unused elements between views.
    pub fn has_no_gaps(&self) -> bool {
        self.has_no_gaps
    }
}
