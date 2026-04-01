// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;

use crate::ArrayRef;
use crate::stats::ArrayStats;

pub(super) const CHILD_SLOT: usize = 0;
pub(super) const NUM_SLOTS: usize = 1;
pub(super) const SLOT_NAMES: [&str; NUM_SLOTS] = ["child"];

#[derive(Clone, Debug)]
pub struct SliceArray {
    pub(super) slots: Vec<Option<ArrayRef>>,
    pub(super) range: Range<usize>,
    pub(super) stats: ArrayStats,
}

pub struct SliceArrayParts {
    pub child: ArrayRef,
    pub range: Range<usize>,
}

impl SliceArray {
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
