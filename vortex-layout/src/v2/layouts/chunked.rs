// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;
use std::fmt;
use std::ops::Range;
use std::sync::Arc;

use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::dtype::DType;
use vortex_array::expr::Expression;
use vortex_error::VortexResult;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

use crate::v2::layout::ChildRelationship;
use crate::v2::layout::Layout;
use crate::v2::layout::LayoutChild;
use crate::v2::layout::LayoutId;
use crate::v2::layout::LayoutVTable;
use crate::v2::scan::planner::ComputeArgs;
use crate::v2::scan::planner::NodeId;
use crate::v2::scan::planner::NodeOpts;
use crate::v2::scan::planner::PlanBuilder;
use crate::v2::scan::planner::SplitPlanner;
use crate::v2::scan::planner::SplitPlannerRef;
use crate::v2::selection::Selection;

/// The chunked layout vtable.
#[derive(Clone)]
pub struct Chunked;

/// Metadata for a chunked layout, storing cumulative row offsets for each chunk.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ChunkedMetadata {
    /// Cumulative row offsets. Has `num_children + 1` entries, starting with 0.
    pub chunk_offsets: Vec<u64>,
}

impl fmt::Display for ChunkedMetadata {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ChunkedMetadata({} chunks)",
            self.chunk_offsets.len().saturating_sub(1)
        )
    }
}

impl LayoutVTable for Chunked {
    type Metadata = ChunkedMetadata;

    fn id(&self) -> LayoutId {
        LayoutId::new_ref("vortex.chunked")
    }

    fn deserialize_metadata(
        _metadata: &[u8],
        _dtype: &DType,
        _row_count: u64,
        children: &[LayoutChild],
        _array_ctx: &ReadContext,
    ) -> VortexResult<Self::Metadata> {
        // Derive cumulative chunk offsets from child row counts.
        let mut chunk_offsets = Vec::with_capacity(children.len() + 1);
        chunk_offsets.push(0);
        let mut offset = 0u64;
        for child in children {
            offset += child.row_count();
            chunk_offsets.push(offset);
        }
        Ok(ChunkedMetadata { chunk_offsets })
    }

    fn child_dtype(layout: &Layout<Self>, _child_idx: usize) -> &DType {
        // All children have the same dtype (homogeneous chunks).
        layout.dtype()
    }

    fn child_relationship(layout: &Layout<Self>, child_idx: usize) -> ChildRelationship {
        ChildRelationship::RowOffset(layout.metadata().chunk_offsets[child_idx])
    }

    fn prepare(
        layout: &Layout<Self>,
        expr: &Expression,
        selection: &Selection,
        row_offset: Option<u64>,
        row_splits: &mut BTreeSet<u64>,
        session: &VortexSession,
    ) -> VortexResult<SplitPlannerRef> {
        let offsets = &layout.metadata().chunk_offsets;
        let num_chunks = layout.num_children();
        let mut children = Vec::new();

        for chunk_idx in 0..num_chunks {
            let chunk_start = offsets[chunk_idx];
            let chunk_end = offsets[chunk_idx + 1];
            let chunk_range = chunk_start..chunk_end;

            // Skip chunks that don't overlap with the selection.
            if !selection.overlaps(&chunk_range) {
                continue;
            }

            // Translate the selection to chunk-local coordinates.
            let local_selection = selection.slice(&chunk_range);

            // Derive the child's global row offset from ours.
            let relationship = Self::child_relationship(layout, chunk_idx);
            let child_offset = relationship.child_row_offset(row_offset);

            let child = layout.child(chunk_idx)?;
            let planner =
                child.prepare(expr, &local_selection, child_offset, row_splits, session)?;

            children.push((chunk_range, relationship, planner));
        }

        Ok(Arc::new(ChunkedSplitPlanner {
            dtype: layout.dtype().clone(),
            children,
        }))
    }
}

/// A split planner for the chunked layout.
struct ChunkedSplitPlanner {
    dtype: DType,
    children: Vec<(Range<u64>, ChildRelationship, SplitPlannerRef)>,
}

impl SplitPlanner for ChunkedSplitPlanner {
    fn plan_split(
        &self,
        row_range: &Range<u64>,
        selection: NodeId,
        builder: &mut PlanBuilder,
    ) -> VortexResult<NodeId> {
        // Find children that overlap with this row range.
        let overlapping: Vec<_> = self
            .children
            .iter()
            .filter(|(chunk_range, ..)| ranges_overlap(row_range, chunk_range))
            .collect();

        match overlapping.len() {
            0 => {
                // No overlapping children — return an empty array.
                let dtype = self.dtype.clone();
                builder.create_node(NodeOpts {
                    label: "Empty",
                    inputs: &[],
                    segments: vec![],
                    lifetime: builder.row_range_lifetime(row_range.clone()),
                    compute: move |_args: ComputeArgs| Ok(Canonical::empty(&dtype).into_array()),
                })
            }
            1 => {
                // Single child — translate row_range to local and delegate.
                let (chunk_range, relationship, planner) = overlapping[0];
                let local_start = row_range.start.saturating_sub(chunk_range.start);
                let local_end = row_range.end.min(chunk_range.end) - chunk_range.start;
                let mut child_builder = builder.step_into(relationship);
                planner.plan_split(&(local_start..local_end), selection, &mut child_builder)
            }
            _ => {
                // Multiple children — plan each and concatenate.
                let mut child_outputs = Vec::with_capacity(overlapping.len());
                for (chunk_range, relationship, planner) in &overlapping {
                    let local_start = row_range.start.max(chunk_range.start) - chunk_range.start;
                    let local_end = row_range.end.min(chunk_range.end) - chunk_range.start;
                    let mut child_builder = builder.step_into(relationship);
                    let child_output = planner.plan_split(
                        &(local_start..local_end),
                        selection,
                        &mut child_builder,
                    )?;
                    child_outputs.push(child_output);
                }
                let dtype = self.dtype.clone();
                builder.create_node(NodeOpts {
                    label: "Concat",
                    inputs: &child_outputs,
                    segments: vec![],
                    lifetime: builder.row_range_lifetime(row_range.clone()),
                    compute: move |args: ComputeArgs| {
                        Ok(ChunkedArray::try_new(args.inputs, dtype)?.into_array())
                    },
                })
            }
        }
    }
}

/// Check if two ranges overlap.
fn ranges_overlap(a: &Range<u64>, b: &Range<u64>) -> bool {
    a.start < b.end && b.start < a.end
}
