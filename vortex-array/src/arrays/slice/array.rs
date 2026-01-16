// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_error::vortex_panic;

use crate::ArrayRef;
use crate::stats::ArrayStats;

#[derive(Clone, Debug)]
pub struct SliceArray {
    pub(super) child: ArrayRef,
    pub(super) range: Range<usize>,
    pub(super) stats: ArrayStats,
}

impl SliceArray {
    pub fn new(child: ArrayRef, range: Range<usize>) -> Self {
        if range.end > child.len() {
            vortex_panic!(
                "SliceArray range out of bounds: range {:?} exceeds child array length {}",
                range,
                child.len()
            );
        }
        Self {
            child,
            range,
            stats: ArrayStats::default(),
        }
    }

    /// The range used to slice the child array.
    pub fn slice_range(&self) -> &Range<usize> {
        &self.range
    }

    /// The child array being sliced.
    pub fn child(&self) -> &ArrayRef {
        &self.child
    }
}
