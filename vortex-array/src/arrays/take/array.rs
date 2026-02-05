// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::ArrayRef;
use crate::stats::ArrayStats;

/// Decomposed parts of the take array.
pub struct TakeArrayParts {
    /// Child array to take elements from
    pub child: ArrayRef,
    /// Indices specifying which elements to take from the child
    pub indices: ArrayRef,
}

/// A lazy array that represents taking elements from a child array at specified indices.
///
/// The resulting array contains the elements at the positions specified by the indices.
#[derive(Clone, Debug)]
pub struct TakeArray {
    /// The source array to take elements from.
    pub(super) child: ArrayRef,

    /// The indices specifying which elements to take.
    pub(super) indices: ArrayRef,

    /// The stats for this array.
    pub(super) stats: ArrayStats,
}

impl TakeArray {
    pub fn new(array: ArrayRef, indices: ArrayRef) -> Self {
        Self::try_new(array, indices).vortex_expect("TakeArray construction failed")
    }

    pub fn try_new(array: ArrayRef, indices: ArrayRef) -> VortexResult<Self> {
        vortex_ensure!(
            indices.dtype().is_int(),
            "TakeArray indices must have integer dtype, got {}",
            indices.dtype()
        );

        Ok(Self {
            child: array,
            indices,
            stats: ArrayStats::default(),
        })
    }

    /// The child array to take elements from.
    pub fn child(&self) -> &ArrayRef {
        &self.child
    }

    /// The indices specifying which elements to take.
    pub fn indices(&self) -> &ArrayRef {
        &self.indices
    }

    /// Consume the array and return its individual components.
    pub fn into_parts(self) -> TakeArrayParts {
        TakeArrayParts {
            child: self.child,
            indices: self.indices,
        }
    }
}
