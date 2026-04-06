// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::array::Array;
use crate::arrays::Filter;
use crate::dtype::DType;
use crate::stats::ArrayStats;

/// The source array being filtered.
pub(super) const CHILD_SLOT: usize = 0;
pub(super) const NUM_SLOTS: usize = 1;
pub(super) const SLOT_NAMES: [&str; NUM_SLOTS] = ["child"];

/// Decomposed parts of the filter array.
pub struct FilterArrayParts {
    /// Child array that is filtered by the mask
    pub child: ArrayRef,
    /// Mask to apply at filter time. Child elements with set indices are kept, the rest discarded.
    pub mask: Mask,
}

// TODO(connor): Write docs on why we have this, and what we had in the old world so that the future
// does not repeat the mistakes of the past.
/// A lazy array that represents filtering a child array by a boolean [`Mask`].
///
/// The resulting array contains only the elements where the mask is true.
#[derive(Clone, Debug)]
pub struct FilterData {
    pub(super) slots: Vec<Option<ArrayRef>>,

    /// The boolean mask selecting which elements to keep.
    pub(super) mask: Mask,

    /// The stats for this array.
    pub(super) stats: ArrayStats,
}

impl FilterData {
    pub fn new(array: ArrayRef, mask: Mask) -> Self {
        Self::try_new(array, mask).vortex_expect("FilterArray construction failed")
    }

    pub fn try_new(array: ArrayRef, mask: Mask) -> VortexResult<Self> {
        vortex_ensure_eq!(
            array.len(),
            mask.len(),
            "FilterArray length mismatch: array has length {} but mask has length {}",
            array.len(),
            mask.len()
        );

        Ok(Self {
            slots: vec![Some(array)],
            mask,
            stats: ArrayStats::default(),
        })
    }

    /// Returns the length of this array (number of elements after filtering).
    pub fn len(&self) -> usize {
        self.mask.true_count()
    }

    /// Returns the [`DType`] of this array.
    pub fn dtype(&self) -> &DType {
        self.child().dtype()
    }

    /// Returns `true` if this array is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The child array being filtered.
    pub fn child(&self) -> &ArrayRef {
        self.slots[CHILD_SLOT]
            .as_ref()
            .vortex_expect("FilterArray child slot")
    }

    /// The mask used to filter the child array.
    pub fn filter_mask(&self) -> &Mask {
        &self.mask
    }
}

impl Array<Filter> {
    /// Creates a new `FilterArray`.
    pub fn new(array: ArrayRef, mask: Mask) -> Self {
        Array::try_from_data(FilterData::new(array, mask))
            .vortex_expect("FilterData is always valid")
    }

    /// Constructs a new `FilterArray`.
    pub fn try_new(array: ArrayRef, mask: Mask) -> VortexResult<Self> {
        Array::try_from_data(FilterData::try_new(array, mask)?)
    }
}

impl FilterData {
    /// Consume the array and return its individual components.
    pub fn into_parts(mut self) -> FilterArrayParts {
        FilterArrayParts {
            child: self.slots[CHILD_SLOT]
                .take()
                .vortex_expect("FilterArray child slot"),
            mask: self.mask,
        }
    }
}
