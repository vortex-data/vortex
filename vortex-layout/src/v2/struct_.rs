// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`StructPlan`] — field-routing node over a struct layout. Zips
//! per-field child plans positionally.
//!
//! See `LAYOUT_PLAN.md` § Per-layout `plan` walkthrough / `StructLayout::plan`.

use std::sync::Arc;

use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::stream::SendableArrayStream;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;

use crate::v2::plan::{LayoutPlan, LayoutPlanRef, PartitionStats};

/// Composes child plans positionally; each child produces values for
/// one field, and `StructPlan::execute` zips them into struct arrays.
pub struct StructPlan {
    children: Vec<LayoutPlanRef>,
    field_names: Vec<FieldName>,
    output_dtype: DType,
}

impl StructPlan {
    pub fn new(
        children: Vec<LayoutPlanRef>,
        field_names: Vec<FieldName>,
        output_dtype: DType,
    ) -> Self {
        debug_assert_eq!(
            children.len(),
            field_names.len(),
            "StructPlan: children and field_names must agree"
        );
        Self {
            children,
            field_names,
            output_dtype,
        }
    }

    pub fn field_names(&self) -> &[FieldName] {
        &self.field_names
    }
}

impl LayoutPlan for StructPlan {
    fn schema(&self) -> &DType {
        &self.output_dtype
    }

    fn partition_count(&self) -> usize {
        // Children are positionally aligned. Partition counts must
        // match across children; the plan's count is any of them.
        self.children
            .first()
            .map(|c| c.partition_count())
            .unwrap_or(1)
    }

    fn partition_stats(&self, partition: usize) -> VortexResult<PartitionStats> {
        // First child's row count is authoritative since children
        // are positionally aligned.
        self.children
            .first()
            .map(|c| c.partition_stats(partition))
            .unwrap_or_else(|| Ok(PartitionStats::unknown()))
    }

    fn output_ordered(&self) -> bool {
        self.children.iter().all(|c| c.output_ordered())
    }

    fn required_input_ordered(&self) -> Vec<bool> {
        // Positional zip — all children must agree on partition order.
        vec![true; self.children.len()]
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true; self.children.len()]
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
                "StructPlan::with_new_children expected {} children, got {}",
                self.children.len(),
                children.len()
            );
        }
        Ok(Arc::new(Self {
            children,
            field_names: self.field_names.clone(),
            output_dtype: self.output_dtype.clone(),
        }))
    }

    fn execute(
        &self,
        _partition: usize,
        _session: &VortexSession,
    ) -> VortexResult<SendableArrayStream> {
        todo!("StructPlan::execute — implemented in PR 3 alongside the differential test")
    }
}
