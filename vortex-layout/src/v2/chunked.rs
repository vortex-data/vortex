// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`ChunkedPlan`] — partitioning node over an ordered sequence of
//! child plans. One chunk per child.
//!
//! See `LAYOUT_PLAN.md` § Per-layout `plan` walkthrough / `ChunkedPlan`.

use std::ops::Range;
use std::sync::Arc;

use futures::TryStreamExt;
use futures::stream;
use vortex_array::dtype::DType;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::stream::SendableArrayStream;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::v2::plan::LayoutPlan;
use crate::v2::plan::LayoutPlanRef;
use crate::v2::plan::PartitionStats;
use crate::v2::scan_ctx::ScanCtx;

/// Routes one partition per child chunk. `partition_count == children.len()`
/// in the default (ordered) mode; relaxed mode is a follow-up PR.
pub struct ChunkedPlan {
    children: Vec<LayoutPlanRef>,
    /// Cumulative row offsets, length `children.len() + 1`. Chunk
    /// `i` covers rows `chunk_offsets[i]..chunk_offsets[i + 1]`.
    chunk_offsets: Vec<u64>,
    output_dtype: DType,
    preserve_order: bool,
}

impl ChunkedPlan {
    pub fn new(children: Vec<LayoutPlanRef>, chunk_offsets: Vec<u64>, output_dtype: DType) -> Self {
        debug_assert_eq!(
            chunk_offsets.len(),
            children.len() + 1,
            "ChunkedPlan: chunk_offsets must have len children.len() + 1",
        );
        Self {
            children,
            chunk_offsets,
            output_dtype,
            preserve_order: true,
        }
    }

    /// In-place flip of the order-preservation flag. See `LAYOUT_PLAN.md`
    /// § Ordered vs. relaxed `ChunkedPlan`.
    pub fn with_preserve_order(self: Arc<Self>, preserve: bool) -> Arc<dyn LayoutPlan> {
        Arc::new(Self {
            children: self.children.clone(),
            chunk_offsets: self.chunk_offsets.clone(),
            output_dtype: self.output_dtype.clone(),
            preserve_order: preserve,
        })
    }

    /// Total row count covered by this Chunked.
    fn total_rows(&self) -> u64 {
        *self.chunk_offsets.last().unwrap_or(&0)
    }

    fn chunk_range(&self, idx: usize) -> Range<u64> {
        self.chunk_offsets[idx]..self.chunk_offsets[idx + 1]
    }
}

impl LayoutPlan for ChunkedPlan {
    fn schema(&self) -> &DType {
        &self.output_dtype
    }

    fn partition_count(&self) -> usize {
        self.children.len()
    }

    fn partition_stats(&self, partition: usize) -> VortexResult<PartitionStats> {
        if partition >= self.children.len() {
            vortex_bail!("ChunkedPlan partition out of range: {partition}");
        }
        Ok(PartitionStats::for_range(self.chunk_range(partition)))
    }

    fn output_ordered(&self) -> bool {
        self.preserve_order
    }

    fn required_input_ordered(&self) -> Vec<bool> {
        vec![true; self.children.len()]
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        // When preserving order, we route partition k → children[k]
        // with no reordering, so each child's order is preserved.
        // When relaxed, we may emit children in arrival order.
        vec![self.preserve_order; self.children.len()]
    }

    fn children(&self) -> &[LayoutPlanRef] {
        &self.children
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<LayoutPlanRef>,
    ) -> VortexResult<LayoutPlanRef> {
        if children.len() != self.children.len() {
            vortex_bail!(
                "ChunkedPlan::with_new_children expected {} children, got {}",
                self.children.len(),
                children.len()
            );
        }
        Ok(Arc::new(Self {
            children,
            chunk_offsets: self.chunk_offsets.clone(),
            output_dtype: self.output_dtype.clone(),
            preserve_order: self.preserve_order,
        }))
    }

    fn execute(&self, row_range: Range<u64>, ctx: &ScanCtx) -> VortexResult<SendableArrayStream> {
        if row_range.start > self.total_rows() || row_range.end > self.total_rows() {
            vortex_bail!(
                "ChunkedPlan::execute row range {row_range:?} exceeds total {}",
                self.total_rows()
            );
        }
        if row_range.start > row_range.end {
            vortex_bail!("ChunkedPlan::execute received reversed row range {row_range:?}");
        }

        // Find chunks intersecting the requested range. Walk each
        // intersecting chunk and dispatch a sub-range relative to
        // the chunk's own row coordinate space.
        //
        // Chunks are ordered and disjoint by construction, so a
        // simple linear scan is fine for typical chunk counts. If a
        // single Chunked grows to thousands of chunks we can swap to
        // binary search over `chunk_offsets`.
        let mut child_streams: Vec<SendableArrayStream> = Vec::new();
        for idx in 0..self.children.len() {
            let chunk_start = self.chunk_offsets[idx];
            let chunk_end = self.chunk_offsets[idx + 1];
            if chunk_end <= row_range.start || chunk_start >= row_range.end {
                continue;
            }
            let intersect_start = chunk_start.max(row_range.start);
            let intersect_end = chunk_end.min(row_range.end);
            // Translate to the child's own row coordinates.
            let child_range = (intersect_start - chunk_start)..(intersect_end - chunk_start);
            child_streams.push(self.children[idx].execute(child_range, ctx)?);
        }

        let dtype = self.output_dtype.clone();
        let flat = stream::iter(child_streams.into_iter().map(VortexResult::Ok)).try_flatten();
        Ok(Box::pin(ArrayStreamAdapter::new(dtype, flat)))
    }
}
