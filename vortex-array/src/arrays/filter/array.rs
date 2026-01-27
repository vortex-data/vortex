// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::stats::ArrayStats;

#[derive(Clone, Debug)]
pub struct FilterArray {
    pub(super) child: ArrayRef,
    pub(super) mask: Mask,
    pub(super) stats: ArrayStats,
}

pub struct FilterArrayParts {
    pub child: ArrayRef,
    pub mask: Mask,
}

impl FilterArray {
    pub fn new(array: ArrayRef, mask: Mask) -> Self {
        Self::try_new(array, mask).vortex_expect("new FilterArray")
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
            child: array,
            mask,
            stats: ArrayStats::default(),
        })
    }

    /// The child array being filtered.
    pub fn child(&self) -> &ArrayRef {
        &self.child
    }

    /// The mask used to filter the child array.
    pub fn filter_mask(&self) -> &Mask {
        &self.mask
    }

    /// Consume the `FilterArray` into its components.
    pub fn into_parts(self) -> FilterArrayParts {
        FilterArrayParts {
            child: self.child,
            mask: self.mask,
        }
    }
}
