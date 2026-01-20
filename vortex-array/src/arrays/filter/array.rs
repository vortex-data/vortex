// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::vortex_panic;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::stats::ArrayStats;

#[derive(Clone, Debug)]
pub struct FilterArray {
    pub(super) child: ArrayRef,
    pub(super) mask: Mask,
    pub(super) stats: ArrayStats,
}

impl FilterArray {
    pub fn new(child: ArrayRef, mask: Mask) -> Self {
        if child.len() != mask.len() {
            vortex_panic!(
                "FilterArray length mismatch: child array has length {} but mask has length {}",
                child.len(),
                mask.len()
            );
        }
        Self {
            child,
            mask,
            stats: ArrayStats::default(),
        }
    }

    /// The child array being filtered.
    pub fn child(&self) -> &ArrayRef {
        &self.child
    }

    /// The mask used to filter the child array.
    pub fn filter_mask(&self) -> &Mask {
        &self.mask
    }
}
