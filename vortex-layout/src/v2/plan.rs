// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use vortex_array::ArrayRef;
use vortex_error::VortexResult;

use crate::segments::SegmentId;
use crate::segments::SegmentSource;

/// Builder passed into a layout for producing per-split plans.
pub struct PlanBuilder {
    nodes: Vec<PlanNode>,
}

enum PlanNode {
    SegmentRead {
        id: NodeId,
        lifetime: Lifetime,
        source: Arc<dyn SegmentSource>,
        segment_id: SegmentId,
    },
    Compute {
        id: NodeId,
        lifetime: Lifetime,
        function: ComputeFn,
    },
}

pub type ComputeFn = Box<dyn FnOnce(&[ArrayRef]) -> VortexResult<ArrayRef> + Send + 'static>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeId(usize);

impl PlanBuilder {
    /// Create a new builder for a single split graph.
    pub fn new_split(&self, row_range: Range<u64>) -> SplitBuilder {
        SplitBuilder { row_range }
    }
}

/// Builder used to construct the I/O graph for a single split.
pub struct SplitBuilder {
    row_range: Range<u64>,
}

impl SplitBuilder {
    pub fn finish(self) -> SplitPlan {}
}

pub struct SplitPlan {}

/// Describes the lifetime of a plan node.
pub enum Lifetime {
    /// The duration of the scan. Never evict.
    Scan,
    /// Alive for a specific row range.
    RowRange(Range<u64>),
    /// Alive until the dynamic "generation" ticks over. e.g. for dynamic expressions.
    Dynamic(Arc<AtomicUsize>),
}

impl Lifetime {
    pub fn covers(&self, _row_range: &Range<u64>) -> bool {
        unimplemented!()
    }
}
