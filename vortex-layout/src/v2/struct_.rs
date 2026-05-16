// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`StructPlan`] — field-routing node over a struct layout. Zips
//! per-field child plans positionally.
//!
//! See `LAYOUT_PLAN.md` § Per-layout `plan` walkthrough / `StructLayout::plan`.

use std::hash::Hash;
use std::hash::Hasher;
use std::ops::Range;
use std::sync::Arc;

use async_stream::try_stream;
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
use vortex_io::session::RuntimeSessionExt;

use crate::v2::aligned::AlignedArrayStream;
use crate::v2::dataflow::OutputFrontier;
use crate::v2::demand::RowDemand;
use crate::v2::experiment::bool_var;
use crate::v2::experiment::trace_flow;
use crate::v2::placeholder::default_array;
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

impl PartialEq for StructPlan {
    fn eq(&self, other: &Self) -> bool {
        crate::v2::plan::plan_slices_eq(&self.children, &other.children)
            && self.field_names == other.field_names
            && self.output_dtype == other.output_dtype
            && self.row_count == other.row_count
    }
}

impl Eq for StructPlan {}

impl Hash for StructPlan {
    fn hash<H: Hasher>(&self, state: &mut H) {
        crate::v2::plan::hash_plan_slice(&self.children, state);
        self.field_names.hash(state);
        self.output_dtype.hash(state);
        self.row_count.hash(state);
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

    fn try_pushdown_mask(self: Arc<Self>, mask_plan: LayoutPlanRef) -> Option<LayoutPlanRef> {
        if bool_var("VORTEX_V2_DISABLE_STRUCT_MASK_PUSHDOWN") {
            return None;
        }
        if self.children.len() > 1 && !bool_var("VORTEX_V2_ENABLE_MULTI_FIELD_STRUCT_MASK_PUSHDOWN")
        {
            return None;
        }
        // All fields share the struct's row space, so each can
        // accept the same mask. If any field can't absorb, bail —
        // a partial pushdown would mean some fields filter and
        // others don't, breaking AlignedArrayStream's row alignment.
        //
        // The same `mask_plan` Arc is cloned once per field. The
        // CSE pass collapses these N identical references into a
        // single `LetPlan` + N `UsePlan`s, so the underlying mask
        // source executes once and chunks fan out via TeeStream.
        // Without CSE this would re-execute the mask N times — the
        // regression that caused this pushdown to be reverted on
        // PR4. CSE + streaming Let make it safe again.
        if !matches!(mask_plan.schema(), DType::Bool(_)) {
            if trace_flow() {
                tracing::debug!(
                    target: "vortex_layout::v2::flow",
                    child_count = self.children.len(),
                    "struct pushdown failed non-bool mask"
                );
            }
            return None;
        }
        let mut new_children = Vec::with_capacity(self.children.len());
        for (child_idx, child) in self.children.iter().enumerate() {
            let Some(absorbed) = Arc::clone(child).try_pushdown_mask(Arc::clone(&mask_plan)) else {
                if trace_flow() {
                    tracing::debug!(
                        target: "vortex_layout::v2::flow",
                        child_idx,
                        child_count = self.children.len(),
                        "struct pushdown failed child rejected"
                    );
                }
                return None;
            };
            if trace_flow() {
                tracing::debug!(
                    target: "vortex_layout::v2::flow",
                    child_idx,
                    child_count = self.children.len(),
                    "struct pushdown child succeeded"
                );
            }
            new_children.push(absorbed);
        }
        if trace_flow() {
            tracing::debug!(
                target: "vortex_layout::v2::flow",
                child_count = self.children.len(),
                "struct pushdown succeeded"
            );
        }
        Some(Arc::new(Self {
            children: new_children,
            field_names: self.field_names.clone(),
            output_dtype: self.output_dtype.clone(),
            // `row_count` stays as the layout's count; partition_stats
            // is an upper bound and the stream emits ≤ this many rows
            // post-filter.
            row_count: self.row_count,
        }))
    }

    fn execute(
        &self,
        row_range: Range<u64>,
        demand: &RowDemand,
        frontier: &OutputFrontier,

        ctx: &ScanCtx,
    ) -> VortexResult<SendableArrayStream> {
        if self.output_dtype.is_nullable() {
            // Nullable structs need a validity child; that wiring lives in
            // StructLayout::plan and isn't plumbed through StructPlan yet.
            vortex_bail!("StructPlan does not yet support nullable structs");
        }

        if bool_var("VORTEX_V2_VALUE_TREE_ROW_DEMAND") && !bool_var("VORTEX_V2_DISABLE_ROW_DEMAND")
        {
            let children = self.children.clone();
            let field_names = self.field_names.clone();
            let output_dtype = self.output_dtype.clone();
            let dtype = output_dtype.clone();
            let demand = demand.clone();
            let frontier = frontier.clone();
            let ctx = ctx.clone();
            let stream = try_stream! {
                let demanded_rows = if bool_var("VORTEX_V2_ROW_DEMAND_RANGE_PULL") {
                    demand.cardinality_uncached(row_range.clone()).await?
                } else {
                    demand.cardinality(row_range.clone()).await?
                };
                if demanded_rows == 0 {
                    let len = usize::try_from(row_range.end - row_range.start)?;
                    yield default_array(&output_dtype, len);
                    return;
                }

                let mut child_streams = Vec::with_capacity(children.len());
                if trace_flow() {
                    tracing::debug!(
                        target: "vortex_layout::v2::flow",
                        row_start = row_range.start,
                        row_end = row_range.end,
                        child_count = children.len(),
                        "struct execute"
                    );
                }
                for child in &children {
                    child_streams.push(child.execute(row_range.clone(), &demand, &frontier, &ctx)?);
                }

                let names: FieldNames = FieldNames::from(field_names.as_slice());
                let mut aligned =
                    Box::pin(AlignedArrayStream::new_labeled(child_streams, ctx.session().handle(), "struct"));
                while let Some(result) = aligned.next().await {
                    let arrays = result?;
                    let len = arrays.first().map_or(0, |a| a.len());
                    if trace_flow() {
                        tracing::debug!(
                            target: "vortex_layout::v2::flow",
                            len,
                            field_count = arrays.len(),
                            "struct emit"
                        );
                    }
                    yield StructArray::try_new(
                        names.clone(),
                        arrays,
                        len,
                        Validity::NonNullable,
                    )?
                    .into_array();
                }
            };
            return Ok(Box::pin(ArrayStreamAdapter::new(dtype, stream)));
        }

        let mut child_streams = Vec::with_capacity(self.children.len());
        if trace_flow() {
            tracing::debug!(
                target: "vortex_layout::v2::flow",
                row_start = row_range.start,
                row_end = row_range.end,
                child_count = self.children.len(),
                "struct execute"
            );
        }
        for child in &self.children {
            child_streams.push(child.execute(row_range.clone(), demand, frontier, ctx)?);
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

        let aligned =
            AlignedArrayStream::new_labeled(child_streams, ctx.session().handle(), "struct");
        let zipped = aligned.map(move |result| {
            let arrays = result?;
            let len = arrays.first().map_or(0, |a| a.len());
            if trace_flow() {
                tracing::debug!(
                    target: "vortex_layout::v2::flow",
                    len,
                    field_count = arrays.len(),
                    "struct emit"
                );
            }
            Ok(
                StructArray::try_new(names.clone(), arrays, len, Validity::NonNullable)?
                    .into_array(),
            )
        });

        Ok(Box::pin(ArrayStreamAdapter::new(dtype, zipped)))
    }
}
