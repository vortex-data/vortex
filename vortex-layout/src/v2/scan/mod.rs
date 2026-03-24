// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub(crate) mod plan;

mod output;
pub mod planner;
mod split;

use std::collections::BTreeMap;
use std::collections::BTreeSet;

use futures::FutureExt;
use futures::Stream;
use futures::stream::StreamExt;
use planner::PlanBuilder;
use planner::SplitPlannerRef;
use vortex_array::IntoArray;
use vortex_array::expr::Expression;
use vortex_error::VortexResult;

use self::plan::Plan;
use self::split::SplitId;
use self::split::SplitRange;
use self::split::form_splits;
use crate::segments::SegmentSource;
use crate::v2::layout::LayoutRef;
use crate::v2::scan::planner::NodeId;
use crate::v2::selection::Selection;

/// Configuration for a scan.
pub struct ScanConfig {
    /// The minimum number of rows in each split.
    min_split_rows: u64,
    /// The maximum number of rows in each split.
    max_split_rows: u64,

    /// How far planning should run ahead of I/O. This configuration determines how many
    /// splits to plan before launching I/O. It is used to ensure that we don't have to
    /// plan all splits up-front before we start processing the query since this can be
    /// reasonably expensive for very large files.
    ///
    /// The plan-ahead window is measured in terms of the number of bytes of planned but
    /// not-yet-scheduled segment reads.
    ///
    /// `None` implies all splits should be planned up-front.
    plan_ahead_kb: Option<u64>,
}

impl Default for ScanConfig {
    fn default() -> Self {
        Self {
            min_split_rows: 2_000,
            max_split_rows: 1_000_000,
            plan_ahead_kb: Some(2 * 1024 * 1024), // 2 GB
        }
    }
}

/// A scan over a [`LayoutRef`] that produces an ordered stream of arrays.
///
/// The scan takes a [`LayoutRef`], expression, and selection, forms splits from row boundaries,
/// builds per-split execution plans, fetches segments concurrently, executes compute nodes, and
/// emits results in split order.
pub struct Scan {
    project_planner: SplitPlannerRef,
    filter_planner: Option<SplitPlannerRef>,
    splits: Vec<SplitRange>,
    config: ScanConfig,
    state: State,
}

struct State {
    next_split_to_plan: usize,
    plan: Plan,
    plan_ahead_kb: u64,
    active_splits: BTreeMap<SplitId, NodeId>,
}

impl Scan {
    /// Creates a new scan for the given layout and expression.
    pub fn try_new(
        layout: &LayoutRef,
        projection: &Expression,
        filter: Option<&Expression>,
        selection: &Selection,
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
            selection,
            layout.row_count(),
            config.min_split_rows,
            config.max_split_rows,
        );

        Ok(Self {
            project_planner,
            filter_planner,
            splits,
            config,
            state: State {
                next_split_to_plan: 0,
                plan: Plan::new(),
                plan_ahead_kb: 0,
                active_splits: BTreeMap::new(),
            },
        })
    }

    /// The main function that drives progress of the scan scheduler.
    fn make_progress(&mut self) -> VortexResult<()> {
        // 1. Fill windows - ensure as many I/O requests have been launched as possible.
        //    In order to get sufficient visibility into upcoming I/O, but without having to do
        //    all planning up-front, we make sure to fill our "plan-ahead" window.
        self.fill_plan_ahead()?;

        // 2. Collect together any outstanding I/O request

        // 2. Re-prioritize

        Ok(())
    }

    /// Make progress on a single split.
    fn make_progress_on_split(&mut self, split_id: SplitId) -> VortexResult<()> {
        todo!()
    }

    /// Plan splits until the plan-ahead window is full.
    fn fill_plan_ahead(&mut self) -> VortexResult<()> {
        loop {
            if self.state.next_split_to_plan == self.splits.len() {
                return Ok(());
            }
            if self
                .config
                .plan_ahead_kb
                .is_some_and(|window| self.state.plan_ahead_kb > window)
            {
                return Ok(());
            }

            let split_range = self.splits[self.state.next_split_to_plan];

            let result_node = {
                let mut plan_builder = PlanBuilder::new(&mut self.state.plan);

                // Start with the initial row selection.
                let mut selection =
                    plan_builder.create_node_resolved(split_range.mask.into_array());

                // Map through the filter planner
                if let Some(filter_planner) = &self.filter_planner {
                    selection = filter_planner.plan_split(
                        &split_range.row_range,
                        selection,
                        &mut plan_builder,
                    )?;
                }

                // And plan the projection
                self.project_planner.plan_split(
                    &split_range.row_range,
                    selection,
                    &mut plan_builder,
                )?
            };

            self.state.next_split_to_plan += 1;
            self.state.active_splits.insert(split_range.id, result_node);

            // We initialize state for the split's nodes.
            loop {
                // self.state.plan.node_inputs();
            }
        }
    }

    /// Callback from the I/O subsystem when a particular segment is resolved.
    fn on_segment_read(&self) {}
}
