// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::OnceLock;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;

use crate::ArrayRef;
use crate::stats::ArrayStats;
use crate::validity::Validity;

#[derive(Clone, Debug)]
pub struct SliceArray {
    pub(super) child: ArrayRef,
    pub(super) range: Range<usize>,
    pub(super) stats: ArrayStats,
    pub(super) cached_validity: OnceLock<Validity>,
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
            child,
            range,
            stats: ArrayStats::default(),
            cached_validity: OnceLock::new(),
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
        &self.child
    }

    /// Consume the slice array and return its components.
    pub fn into_parts(self) -> SliceArrayParts {
        SliceArrayParts {
            child: self.child,
            range: self.range,
        }
    }
}
