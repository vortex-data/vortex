// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;

use crate::ArrayRef;
use crate::array::Array;
use crate::arrays::Slice;
use crate::dtype::DType;
use crate::stats::ArrayStats;

/// The underlying child array being sliced.
pub(super) const CHILD_SLOT: usize = 0;
pub(super) const NUM_SLOTS: usize = 1;
pub(super) const SLOT_NAMES: [&str; NUM_SLOTS] = ["child"];

#[derive(Clone, Debug)]
pub struct SliceData {
    pub(super) slots: Vec<Option<ArrayRef>>,
    pub(super) range: Range<usize>,
    pub(super) stats: ArrayStats,
}

pub struct SliceArrayParts {
    pub child: ArrayRef,
    pub range: Range<usize>,
}

impl SliceData {
    pub fn try_new(child: ArrayRef, range: Range<usize>) -> VortexResult<Self> {
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
            stats: ArrayStats::default(),
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
}

impl Array<Slice> {
    /// Constructs a new `SliceArray`.
    pub fn try_new(child: ArrayRef, range: Range<usize>) -> VortexResult<Self> {
        Array::try_from_data(SliceData::try_new(child, range)?)
    }

    /// Constructs a new `SliceArray`.
    pub fn new(child: ArrayRef, range: Range<usize>) -> Self {
        Array::try_from_data(SliceData::new(child, range))
            .vortex_expect("SliceData is always valid")
    }
}

impl SliceData {
    /// Consume the slice array and return its components.
    pub fn into_parts(mut self) -> SliceArrayParts {
        SliceArrayParts {
            child: self.slots[CHILD_SLOT]
                .take()
                .vortex_expect("SliceArray child slot"),
            range: self.range,
        }
    }
}
