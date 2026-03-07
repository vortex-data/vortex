// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;

use crate::ArrayCommon;
use crate::ArrayRef;

#[derive(Clone, Debug)]
pub struct SliceArray {
    pub(super) child: ArrayRef,
    pub(super) range: Range<usize>,
    pub(super) common: ArrayCommon,
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
        let len = range.len();
        let dtype = child.dtype().clone();
        Ok(Self {
            child,
            range,
            common: ArrayCommon::new(len, dtype),
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
