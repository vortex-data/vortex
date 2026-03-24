// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub(crate) mod plan;

mod output;
mod split;

use std::collections::BTreeMap;
use std::collections::BTreeSet;

use futures::FutureExt;
use futures::Stream;
use futures::stream::StreamExt;
use vortex_array::expr::Expression;
use vortex_error::VortexResult;

use self::plan::SplitPlan;
use self::split::SplitId;
use self::split::SplitRange;
use self::split::form_splits;
use crate::segments::SegmentSource;
use crate::v2::layout::LayoutRef;
use crate::v2::layout::RowSelection;
use crate::v2::planner::SplitPlannerRef;

/// Configuration for a scan.
pub struct ScanConfig {
    min_split_rows: u64,
    max_split_rows: u64,
}

impl Default for ScanConfig {
    fn default() -> Self {
        Self {
            min_split_rows: 2_000,
            max_split_rows: 1_000_000, // Hmmm
        }
    }
}

/// A scan over a layout that produces an ordered stream of arrays.
///
/// The scan takes a [`LayoutRef`], expression, and selection, forms splits from row boundaries,
/// builds per-split execution plans, fetches segments concurrently, executes compute nodes, and
/// emits results in split order.
pub struct Scan {
    project_planner: SplitPlannerRef,
    filter_planner: Option<SplitPlannerRef>,
    splits: Vec<SplitRange>,

    next_split_idx: usize,
    active_splits: BTreeMap<SplitId, SplitPlan>,
    config: ScanConfig,
}

impl Scan {
    /// Creates a new scan for the given layout and expression.
    pub fn try_new(
        layout: &LayoutRef,
        projection: &Expression,
        filter: Option<&Expression>,
        selection: &RowSelection,
        config: ScanConfig,
    ) -> VortexResult<Self> {
        // Create the row splits set, this is populated in the `prepare` call to give us hints
        // on how to partition the scan.
        let mut row_splits = BTreeSet::new();
        row_splits.insert(0);
        row_splits.insert(layout.row_count());

        // Create split planners for the project / filter expressions.
        let project_planner = layout.prepare(projection, selection, &mut row_splits)?;
        let filter_planner = filter
            .map(|f| layout.prepare(f, selection, &mut row_splits))
            .transpose()?;

        // We now figure out the row splits for executing the scan. This allows us to have some
        // amount of parallelism.
        let splits = form_splits(
            &row_splits,
            layout.row_count(),
            config.min_split_rows,
            config.max_split_rows,
        );

        //

        Ok(Self {
            project_planner,
            filter_planner,
            splits,
            next_split_idx: 0,
            active_splits: BTreeMap::new(),
            config,
        })
    }
}
