// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::vortex_panic;
use vortex_mask::Mask;

use crate::Array;
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

    pub fn mask(&self) -> &Mask {
        &self.mask
    }
}
