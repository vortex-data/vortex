// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::array::Array;
use crate::array::ArrayParts;
use crate::array::TypedArrayRef;
use crate::arrays::Filter;

/// The source array being filtered.
pub(super) const CHILD_SLOT: usize = 0;
pub(super) const NUM_SLOTS: usize = 1;
pub(super) const SLOT_NAMES: [&str; NUM_SLOTS] = ["child"];

// TODO(connor): Write docs on why we have this, and what we had in the old world so that the future
// does not repeat the mistakes of the past.
/// A lazy array that represents filtering a child array by a boolean [`Mask`].
///
/// The resulting array contains only the elements where the mask is true.
#[derive(Clone, Debug)]
pub struct FilterData {
    /// The boolean mask selecting which elements to keep.
    pub(super) mask: Mask,
}

pub struct FilterDataParts {
    pub mask: Mask,
}

pub trait FilterArrayExt: TypedArrayRef<Filter> {
    fn child(&self) -> &ArrayRef {
        self.as_ref().slots()[CHILD_SLOT]
            .as_ref()
            .vortex_expect("validated filter child slot")
    }
}
impl<T: TypedArrayRef<Filter>> FilterArrayExt for T {}

impl FilterData {
    pub fn new(array: ArrayRef, mask: Mask) -> Self {
        Self::try_new(array, mask).vortex_expect("FilterArray construction failed")
    }

    fn try_new(array: ArrayRef, mask: Mask) -> VortexResult<Self> {
        vortex_ensure_eq!(
            array.len(),
            mask.len(),
            "FilterArray length mismatch: array has length {} but mask has length {}",
            array.len(),
            mask.len()
        );

        Ok(Self { mask })
    }

    /// Returns the length of this array (number of elements after filtering).
    pub fn len(&self) -> usize {
        self.mask.true_count()
    }

    /// Returns `true` if this array is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The mask used to filter the child array.
    pub fn filter_mask(&self) -> &Mask {
        &self.mask
    }

    pub fn into_parts(self) -> FilterDataParts {
        FilterDataParts { mask: self.mask }
    }
}

impl Array<Filter> {
    /// Creates a new `FilterArray`.
    pub fn new(array: ArrayRef, mask: Mask) -> Self {
        let dtype = array.dtype().clone();
        let len = mask.true_count();
        let data = FilterData::new(array.clone(), mask);
        unsafe {
            Array::from_parts_unchecked(
                ArrayParts::new(Filter, dtype, len, data).with_slots(vec![Some(array)]),
            )
        }
    }

    /// Constructs a new `FilterArray`.
    pub fn try_new(array: ArrayRef, mask: Mask) -> VortexResult<Self> {
        let dtype = array.dtype().clone();
        let len = mask.true_count();
        let data = FilterData::try_new(array.clone(), mask)?;
        Ok(unsafe {
            Array::from_parts_unchecked(
                ArrayParts::new(Filter, dtype, len, data).with_slots(vec![Some(array)]),
            )
        })
    }
}
