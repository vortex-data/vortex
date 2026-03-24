// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;
use std::fmt;
use std::ops::Range;
use std::sync::Arc;

use vortex_array::dtype::DType;
use vortex_array::dtype::TryFromBytes;
use vortex_array::expr::Expression;
use vortex_array::expr::stats::Stat;
use vortex_array::stats::stats_from_bitset_bytes;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;

use crate::v2::layout::ChildRelationship;
use crate::v2::layout::Layout;
use crate::v2::layout::LayoutChild;
use crate::v2::layout::LayoutId;
use crate::v2::layout::LayoutRef;
use crate::v2::layout::LayoutVTable;
use crate::v2::scan::planner::NodeId;
use crate::v2::scan::planner::NodeInput;
use crate::v2::scan::planner::NodeOpts;
use crate::v2::scan::planner::PlanBuilder;
use crate::v2::scan::planner::SplitPlanner;
use crate::v2::scan::planner::SplitPlannerRef;
use crate::v2::selection::Selection;

/// The zoned layout vtable.
#[derive(Clone)]
pub struct Zoned;

/// Metadata for a zoned layout.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ZonedMetadata {
    /// The number of data rows per zone.
    pub zone_len: u64,
    /// Present statistics
    pub present_stats: Arc<[Stat]>,
}

impl fmt::Display for ZonedMetadata {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ZonedMetadata(zone_len={})", self.zone_len)
    }
}

impl LayoutVTable for Zoned {
    type Metadata = ZonedMetadata;
    type Plan = ();

    fn id(&self) -> LayoutId {
        LayoutId::new_ref("vortex.stats")
    }

    fn deserialize_metadata(
        metadata: &[u8],
        _dtype: &DType,
        _row_count: u64,
        _children: &[LayoutChild],
    ) -> VortexResult<ZonedMetadata> {
        let zone_len = u32::try_from_le_bytes(&metadata[0..4])? as u64;
        let present_stats: Arc<[Stat]> = stats_from_bitset_bytes(&metadata[4..]).into();
        Ok(ZonedMetadata {
            zone_len,
            present_stats,
        })
    }

    fn child_dtype(layout: &Layout<Self>, child_idx: usize) -> &DType {
        match child_idx {
            // Child 0 is the data child, same dtype as parent.
            0 => layout.dtype(),
            // Child 1 is the zone map, dtype derived from data statistics.
            // For now, return the parent dtype as a placeholder.
            1 => layout.dtype(),
            _ => vortex_panic!("Zoned layout has only 2 children, got index {child_idx}"),
        }
    }

    fn child_relationship(layout: &Layout<Self>, child_idx: usize) -> ChildRelationship {
        match child_idx {
            // Data child is in the same row space as the parent.
            0 => ChildRelationship::RowOffset(0),
            // Zone map is auxiliary data scoped to the parent's full row range.
            1 => ChildRelationship::Auxiliary(0..layout.row_count()),
            _ => vortex_panic!("Zoned layout has only 2 children, got index {child_idx}"),
        }
    }

    fn prepare(
        layout: &Layout<Self>,
        expr: &Expression,
        selection: &Selection,
        row_splits: &mut BTreeSet<u64>,
    ) -> VortexResult<SplitPlannerRef> {
        let zone_len = layout.metadata().zone_len;
        let nzones = layout.row_count().div_ceil(zone_len);

        // Prepare the data child with the original expression and selection.
        // Only the data child contributes row split boundaries.
        let _data_rel = Self::child_relationship(layout, 0);
        let data_child = layout.data_child()?;
        let data_planner = data_child.prepare(expr, selection, row_splits)?;

        // TODO(ngates): derive pruning predicate via expr.stat_falsification(...)
        // For now, skip zone map optimization and just delegate to the data child.
        //
        // When implemented, prepare the zone map child once with the full zone range.
        // The zone map is read in its entirety (0..nzones) and shared across all splits.
        let zone_map_planner = None;
        let _zm_child = layout.zone_map_child()?;
        let _zm_rel = Self::child_relationship(layout, 1);
        let _zm_selection = Selection::All;
        // let mut zm_builder = builder.step_into(&zm_rel);
        // let zm_planner = zm_child.prepare(&pruning_expr, &zm_selection, ..., &mut zm_builder)?;

        Ok(Arc::new(ZonedSplitPlanner {
            zone_len,
            row_count: layout.row_count(),
            _nzones: nzones,
            data_planner,
            zone_map_planner,
        }))
    }
}

impl Layout<Zoned> {
    /// Returns the data child of the zoned layout.
    pub fn data_child(&self) -> VortexResult<LayoutRef> {
        self.child(0)
    }

    /// Returns the zone map child of the zoned layout.
    pub fn zone_map_child(&self) -> VortexResult<LayoutRef> {
        self.child(1)
    }
}

struct ZonedSplitPlanner {
    zone_len: u64,
    row_count: u64,
    _nzones: u64,
    data_planner: SplitPlannerRef,
    zone_map_planner: Option<SplitPlannerRef>,
}

impl ZonedSplitPlanner {
    /// Maps a data row range to the corresponding zone index range.
    fn zone_range(&self, row_range: &Range<u64>) -> Range<u64> {
        let zone_start = row_range.start / self.zone_len;
        let zone_end = row_range.end.div_ceil(self.zone_len);
        zone_start..zone_end
    }
}

impl SplitPlanner for ZonedSplitPlanner {
    fn plan_split(
        &self,
        row_range: &Range<u64>,
        selection: NodeId,
        builder: &mut PlanBuilder,
    ) -> VortexResult<NodeId> {
        // Always plan the data child split.
        let data_output = self
            .data_planner
            .plan_split(row_range, selection, builder)?;

        let Some(zone_map_planner) = &self.zone_map_planner else {
            // No pruning predicate, just return the data output directly.
            return Ok(data_output);
        };

        // Map the data row range to zone indices and plan the zone map read.
        let zone_range = self.zone_range(row_range);
        let zm_output = zone_map_planner.plan_split(&zone_range, selection, builder)?;

        // Create a compute node that:
        // 1. Evaluates the pruning predicate on the zone map rows.
        // 2. Expands zone-level bits to row-level mask (each zone bit covers zone_len rows).
        // 3. Intersects with the data output — returns data if not pruned, empty if pruned.
        let zone_len = self.zone_len;
        let row_count = self.row_count;
        builder.create_node(NodeOpts {
            inputs: &[zm_output, data_output],
            segments: vec![],
            lifetime: builder.row_range_lifetime(row_range.clone()),
            compute: move |mut inputs: Vec<NodeInput>| {
                let _zm_array = inputs.remove(0).into_array();
                let data_array = inputs.remove(0).into_array();
                // TODO: evaluate pruning predicate on zone map result,
                // expand zone-level bits to row-level mask (each zone covers zone_len rows),
                // intersect with data — filter or return empty if pruned.
                let _zone_len = zone_len;
                let _row_count = row_count;
                // For now, return data unfiltered.
                Ok(data_array)
            },
        })
    }
}
