// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

/// Describes the lifetime of a plan node.
pub enum Lifetime {
    /// The duration of the scan. Never evict.
    Scan,
    /// Alive for a specific row range.
    RowRange(Range<u64>),
    /// Alive until the dynamic "generation" ticks over. e.g. for dynamic expressions.
    Dynamic(Arc<AtomicUsize>),
    /// Unknown lifetime
    Unknown,
}

impl Lifetime {
    pub fn covers(&self, _row_range: &Range<u64>) -> bool {
        unimplemented!()
    }
}
