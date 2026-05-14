// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`StructPlan`] — field-routing node over a struct layout. Zips
//! per-field child plans positionally.
//!
//! See `LAYOUT_PLAN.md` § Per-layout `plan` walkthrough / `StructLayout::plan`.

use std::ops::Range;
use std::sync::Arc;

use futures::StreamExt;
use vortex_array::IntoArray;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::FieldNames;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::stream::SendableArrayStream;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::v2::aligned::AlignedArrayStream;
use crate::v2::plan::LayoutPlan;
use crate::v2::plan::LayoutPlanRef;
use crate::v2::plan::PartitionStats;
use crate::v2::scan_ctx::ScanCtx;

/// Composes child plans positionally; each child produces values for
/// one field, and `StructPlan::execute` zips them into struct arrays.
pub struct StructPlan {
    children: Vec<LayoutPlanRef>,
    field_names: Vec<FieldName>,
    output_dtype: DType,
    /// Total row count of the struct — all children share this row
    /// space even when their internal chunking differs (the writer's
    /// byte-based coalescing leaves fields with non-aligned chunks).
    row_count: u64,
}

impl StructPlan {
    pub fn new(
        children: Vec<LayoutPlanRef>,
        field_names: Vec<FieldName>,
        output_dtype: DType,
        row_count: u64,
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
            row_count,
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
        // A struct's natural splits are the union of its children's
        // split boundaries — that's the finest aligned granularity
        // every field can produce without slicing. For PR4 we don't
        // attempt that union and just expose a single partition over
        // the full row range; engines usually drive partitioning from
        // the dominant `Chunked` layer above the struct (e.g., zoned
        // splits in TPC-H).
        1
    }

    fn partition_stats(&self, partition: usize) -> VortexResult<PartitionStats> {
        if partition >= 1 {
            vortex_bail!("StructPlan partition out of range: {partition}");
        }
        Ok(PartitionStats::for_range(0..self.row_count))
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
            row_count: self.row_count,
        }))
    }

    fn execute(&self, row_range: Range<u64>, ctx: &ScanCtx) -> VortexResult<SendableArrayStream> {
        if self.output_dtype.is_nullable() {
            // Nullable structs need a validity child; that wiring lives in
            // StructLayout::plan and isn't plumbed through StructPlan yet.
            vortex_bail!("StructPlan does not yet support nullable structs");
        }

        let mut child_streams = Vec::with_capacity(self.children.len());
        for child in &self.children {
            child_streams.push(child.execute(row_range.clone(), ctx)?);
        }

        // Different fields can return arrays at different chunk
        // granularities for the same row range (the writer's
        // byte-based coalescing produces one big chunk for narrow
        // numeric columns and several smaller chunks for wide string
        // columns). `AlignedArrayStream` k-way-zips the children,
        // slicing each step to the smallest currently-available
        // length so we emit row-aligned struct batches without
        // collecting either side fully into memory.
        let names: FieldNames = FieldNames::from(self.field_names.as_slice());
        let dtype = self.output_dtype.clone();

        let aligned = AlignedArrayStream::new(child_streams);
        let zipped = aligned.map(move |result| {
            let arrays = result?;
            let len = arrays.first().map_or(0, |a| a.len());
            Ok(
                StructArray::try_new(names.clone(), arrays, len, Validity::NonNullable)?
                    .into_array(),
            )
        });

        Ok(Box::pin(ArrayStreamAdapter::new(dtype, zipped)))
    }
}
