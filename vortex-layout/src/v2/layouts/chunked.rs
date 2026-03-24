// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;
use std::fmt;
use std::ops::Range;
use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::dtype::DType;
use vortex_array::expr::Expression;
use vortex_error::VortexResult;

use crate::v2::layout::ChildRelationship;
use crate::v2::layout::Layout;
use crate::v2::layout::LayoutId;
use crate::v2::layout::LayoutVTable;
use crate::v2::scan::planner::NodeId;
use crate::v2::scan::planner::NodeInput;
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
    type Plan = ();

    fn id(&self) -> LayoutId {
        LayoutId::new_ref("vortex.chunked")
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
        row_splits: &mut BTreeSet<u64>,
    ) -> VortexResult<SplitPlannerRef> {
        let offsets = &layout.metadata().chunk_offsets;
        let num_chunks = layout.num_children();
        let mut children = Vec::new();

        for chunk_idx in 0..num_chunks {
            let chunk_start = offsets[chunk_idx];
            let chunk_end = offsets[chunk_idx + 1];
            let chunk_range = chunk_start..chunk_end;

            // Skip chunks that don't overlap with the selection.
            if !selection_overlaps(selection, &chunk_range) {
                continue;
            }

            // Translate the selection to chunk-local coordinates.
            let local_selection = translate_selection(selection, chunk_start, chunk_end);

            // Step into the child's coordinate space.
            let child = layout.child(chunk_idx)?;
            let planner = child.prepare(expr, &local_selection, row_splits)?;

            // Translate any row splits added by the child back to global coordinates.
            // We collect and re-insert since the child adds splits in its local space.
            let global_splits: Vec<u64> = row_splits
                .range(0..chunk_end.saturating_sub(chunk_start))
                .map(|&s| s + chunk_start)
                .collect();
            let local_splits: Vec<u64> = row_splits
                .range(0..chunk_end.saturating_sub(chunk_start))
                .copied()
                .collect();
            for s in &local_splits {
                row_splits.remove(s);
            }
            for s in global_splits {
                row_splits.insert(s);
            }

            // Register the chunk boundary as a split point.
            row_splits.insert(chunk_end);

            children.push((chunk_range, planner));
        }

        // Ensure the start boundary is registered.
        if let Some(first_offset) = offsets.first() {
            row_splits.insert(*first_offset);
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
    children: Vec<(Range<u64>, SplitPlannerRef)>,
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
            .filter(|(chunk_range, _)| ranges_overlap(row_range, chunk_range))
            .collect();

        match overlapping.len() {
            0 => {
                // No overlapping children — return an empty array.
                let dtype = self.dtype.clone();
                builder.create_node(NodeOpts {
                    inputs: &[],
                    segments: vec![],
                    lifetime: builder.row_range_lifetime(row_range.clone()),
                    compute: move |_inputs: Vec<NodeInput>| {
                        Ok(Canonical::empty(&dtype).into_array())
                    },
                })
            }
            1 => {
                // Single child — translate row_range to local and delegate.
                let (chunk_range, planner) = overlapping[0];
                let local_start = row_range.start.saturating_sub(chunk_range.start);
                let local_end = row_range.end.min(chunk_range.end) - chunk_range.start;
                planner.plan_split(&(local_start..local_end), selection, builder)
            }
            _ => {
                // Multiple children — plan each and concatenate.
                let mut child_outputs = Vec::with_capacity(overlapping.len());
                for (chunk_range, planner) in &overlapping {
                    let local_start = row_range.start.max(chunk_range.start) - chunk_range.start;
                    let local_end = row_range.end.min(chunk_range.end) - chunk_range.start;
                    let child_output =
                        planner.plan_split(&(local_start..local_end), selection, builder)?;
                    child_outputs.push(child_output);
                }
                let dtype = self.dtype.clone();
                builder.create_node(NodeOpts {
                    inputs: &child_outputs,
                    segments: vec![],
                    lifetime: builder.row_range_lifetime(row_range.clone()),
                    compute: move |inputs: Vec<NodeInput>| {
                        let chunks: Vec<ArrayRef> =
                            inputs.into_iter().map(|i| i.into_array()).collect();
                        Ok(ChunkedArray::try_new(chunks, dtype)?.into_array())
                    },
                })
            }
        }
    }
}

/// Check if a selection overlaps with a given range.
///
/// TODO: implement precise overlap checking for non-All selection variants.
fn selection_overlaps(_selection: &Selection, _range: &Range<u64>) -> bool {
    // Conservative: assume all chunks may overlap. Precise checks for IncludeByIndex,
    // ExcludeByIndex, and Roaring variants can be added later.
    true
}

/// Check if two ranges overlap.
fn ranges_overlap(a: &Range<u64>, b: &Range<u64>) -> bool {
    a.start < b.end && b.start < a.end
}

/// Translate a selection to chunk-local coordinates.
///
/// TODO: implement precise translation for non-All selection variants.
fn translate_selection(selection: &Selection, _chunk_start: u64, _chunk_end: u64) -> Selection {
    match selection {
        Selection::All => Selection::All,
        // Conservative: pass through to all chunks. Precise index translation can be added later.
        _ => selection.clone(),
    }
}
