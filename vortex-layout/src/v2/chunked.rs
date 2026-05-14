// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`ChunkedPlan`] — partitioning node over an ordered sequence of
//! child plans. One chunk per child.
//!
//! See `LAYOUT_PLAN.md` § Per-layout `plan` walkthrough / `ChunkedPlan`.

use std::sync::Arc;

use vortex_array::dtype::DType;
use vortex_array::stream::SendableArrayStream;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;

use crate::v2::plan::LayoutPlan;
use crate::v2::plan::LayoutPlanRef;
use crate::v2::plan::PartitionStats;

/// Routes one partition per child chunk. `partition_count == children.len()`
/// in the default (ordered) mode; relaxed mode is a follow-up PR.
pub struct ChunkedPlan {
    children: Vec<LayoutPlanRef>,
    output_dtype: DType,
    preserve_order: bool,
}

impl ChunkedPlan {
    pub fn new(children: Vec<LayoutPlanRef>, output_dtype: DType) -> Self {
        Self {
            children,
            output_dtype,
            preserve_order: true,
        }
    }

    /// In-place flip of the order-preservation flag. See `LAYOUT_PLAN.md`
    /// § Ordered vs. relaxed `ChunkedPlan`.
    pub fn with_preserve_order(self: Arc<Self>, preserve: bool) -> Arc<dyn LayoutPlan> {
        Arc::new(Self {
            children: self.children.clone(),
            output_dtype: self.output_dtype.clone(),
            preserve_order: preserve,
        })
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
        let child = self.children.get(partition).ok_or_else(|| {
            vortex_error::vortex_err!("ChunkedPlan partition out of range: {partition}")
        })?;
        child.partition_stats(0)
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
            output_dtype: self.output_dtype.clone(),
            preserve_order: self.preserve_order,
        }))
    }

    fn execute(
        &self,
        partition: usize,
        session: &VortexSession,
    ) -> VortexResult<SendableArrayStream> {
        let child = self.children.get(partition).ok_or_else(|| {
            vortex_error::vortex_err!("ChunkedPlan partition out of range: {partition}")
        })?;
        // One partition == one chunk. Each chunk plan exposes a single
        // partition of its own, so we always execute partition 0.
        child.execute(0, session)
    }
}
