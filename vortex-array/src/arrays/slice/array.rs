// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;

use crate::ArrayRef;
use crate::array::Array;
use crate::array::ArrayParts;
use crate::arrays::Slice;
use crate::dtype::DType;

/// The underlying child array being sliced.
pub(super) const CHILD_SLOT: usize = 0;
pub(super) const NUM_SLOTS: usize = 1;
pub(super) const SLOT_NAMES: [&str; NUM_SLOTS] = ["child"];

#[derive(Clone, Debug)]
pub struct SliceData {
    pub(super) slots: Vec<Option<ArrayRef>>,
    pub(super) range: Range<usize>,
}

pub struct SliceDataParts {
    pub child: ArrayRef,
    pub range: Range<usize>,
}

impl SliceData {
    fn try_new(child: ArrayRef, range: Range<usize>) -> VortexResult<Self> {
        if range.end > child.len() {
            vortex_panic!(
                "SliceArray range out of bounds: range {:?} exceeds child array length {}",
                range,
                child.len()
            );
        }
        Ok(Self {
            slots: vec![Some(child)],
            range,
        })
    }

    pub fn new(child: ArrayRef, range: Range<usize>) -> Self {
        Self::try_new(child, range).vortex_expect("failed")
    }

    /// Returns the length of this array.
    pub fn len(&self) -> usize {
        self.range.len()
    }

    /// Returns the [`DType`] of this array.
    pub fn dtype(&self) -> &DType {
        self.child().dtype()
    }

    /// Returns `true` if this array is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The range used to slice the child array.
    pub fn slice_range(&self) -> &Range<usize> {
        &self.range
    }

    /// The child array being sliced.
    pub fn child(&self) -> &ArrayRef {
        self.slots[CHILD_SLOT]
            .as_ref()
            .vortex_expect("SliceArray child slot")
    }

    pub fn into_parts(mut self) -> SliceDataParts {
        SliceDataParts {
            child: self.slots[CHILD_SLOT]
                .take()
                .vortex_expect("SliceArray child slot"),
            range: self.range,
        }
    }
}

impl Array<Slice> {
    /// Constructs a new `SliceArray`.
    pub fn try_new(child: ArrayRef, range: Range<usize>) -> VortexResult<Self> {
        let len = range.len();
        let dtype = child.dtype().clone();
        let data = SliceData::try_new(child, range)?;
        Ok(unsafe { Array::from_parts_unchecked(ArrayParts::new(Slice, dtype, len, data)) })
    }

    /// Constructs a new `SliceArray`.
    pub fn new(child: ArrayRef, range: Range<usize>) -> Self {
        let len = range.len();
        let dtype = child.dtype().clone();
        let data = SliceData::new(child, range);
        unsafe { Array::from_parts_unchecked(ArrayParts::new(Slice, dtype, len, data)) }
    }
}
