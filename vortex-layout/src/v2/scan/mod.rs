// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub(crate) mod plan;

mod output;
mod split;

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::sync::Arc;

use futures::FutureExt;
use futures::Stream;
use futures::stream::FuturesUnordered;
use futures::stream::StreamExt;
use vortex_array::ArrayRef;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;

use self::output::OutputQueue;
use self::plan::SplitPlan;
use self::split::SplitId;
use self::split::SplitRange;
use self::split::form_splits;
use crate::segments::SegmentSource;
use crate::v2::layout::LayoutRef;
use crate::v2::layout::RowSelection;
use crate::v2::planner::PlanBuilder;
use crate::v2::planner::SplitPlannerRef;
use crate::v2::planner::SplitSelection;

/// Configuration for a scan.
pub struct ScanConfig {
    /// Maximum number of concurrently active splits.
    pub window_size: usize,
    /// Minimum number of rows per split (small intervals are coalesced).
    pub min_split_rows: u64,
    /// Maximum number of rows per split (large intervals are subdivided).
    pub max_split_rows: u64,
}

impl Default for ScanConfig {
    fn default() -> Self {
        Self {
            window_size: 8,
            min_split_rows: 1024,
            max_split_rows: 1_000_000,
        }
    }
}

#[derive(PartialEq, Eq)]
enum ScanState {
    Running,
    Draining,
    Done,
}

/// A scan over a layout that produces an ordered stream of arrays.
///
/// The scan takes a [`LayoutRef`], expression, and selection, forms splits from row boundaries,
/// builds per-split execution plans, fetches segments concurrently, executes compute nodes, and
/// emits results in split order.
pub struct Scan {
    planner: SplitPlannerRef,
    segment_source: Arc<dyn SegmentSource>,
    splits: Vec<SplitRange>,
    next_split_idx: usize,
    active_splits: BTreeMap<SplitId, SplitPlan>,
    output_queue: OutputQueue,
    config: ScanConfig,
    state: ScanState,
}

impl Scan {
    /// Creates a new scan for the given layout and expression.
    pub fn try_new(
        layout: &LayoutRef,
        expr: &vortex_array::expr::Expression,
        selection: &RowSelection,
        config: ScanConfig,
    ) -> VortexResult<Self> {
        let mut row_splits = BTreeSet::new();
        let mut builder = PlanBuilder::new();
        let planner = layout.prepare(expr, selection, &mut row_splits, &mut builder)?;

        let total_row_count = layout.row_count();
        let segment_source = layout.segment_source().clone();

        let splits = form_splits(
            &row_splits,
            total_row_count,
            config.min_split_rows,
            config.max_split_rows,
        );
        let total_splits = u32::try_from(splits.len()).unwrap_or(u32::MAX);

        Ok(Self {
            planner,
            segment_source,
            splits,
            next_split_idx: 0,
            active_splits: BTreeMap::new(),
            output_queue: OutputQueue::new(total_splits),
            config,
            state: ScanState::Running,
        })
    }

    /// Returns the next batch from the scan, or `None` when complete.
    pub async fn next_batch(&mut self) -> VortexResult<Option<ArrayRef>> {
        loop {
            // Drain any ready outputs.
            for (_, result) in self.output_queue.drain_ready() {
                if let Some(array) = result {
                    return Ok(Some(array));
                }
            }

            if self.state == ScanState::Done {
                return Ok(None);
            }

            if self.state == ScanState::Draining {
                if self.output_queue.is_complete() {
                    self.state = ScanState::Done;
                }
                return Ok(None);
            }

            // Fill the window with new splits.
            self.fill_window()?;

            if self.active_splits.is_empty() {
                self.state = ScanState::Draining;
                continue;
            }

            // Process all active splits: fetch segments and execute compute.
            self.process_window().await?;

            // Move completed splits to the output queue.
            self.collect_completed();

            // If no more work, transition to draining.
            if self.active_splits.is_empty() && self.next_split_idx >= self.splits.len() {
                self.state = ScanState::Draining;
            }
        }
    }

    /// Converts this scan into a stream of arrays.
    pub fn into_stream(mut self) -> impl Stream<Item = VortexResult<ArrayRef>> {
        async_stream::try_stream! {
            while let Some(batch) = self.next_batch().await? {
                yield batch;
            }
        }
    }

    /// Fills the active split window up to `window_size`.
    fn fill_window(&mut self) -> VortexResult<()> {
        while self.active_splits.len() < self.config.window_size
            && self.next_split_idx < self.splits.len()
        {
            let split = &self.splits[self.next_split_idx];
            let selection = SplitSelection::new();
            let mut builder = PlanBuilder::new();
            let root =
                self.planner
                    .plan_split(split.row_range.clone(), &selection, &mut builder)?;
            let mut plan = builder.take_plan();
            plan.set_root(root);
            plan.finalize();
            self.active_splits.insert(split.id, plan);
            self.next_split_idx += 1;
        }
        Ok(())
    }

    /// Fetches all pending segments across active splits and executes ready compute nodes.
    async fn process_window(&mut self) -> VortexResult<()> {
        // Execute any already-ready nodes (e.g., nodes with no dependencies).
        Self::execute_all_ready(&mut self.active_splits)?;

        // Collect all pending segment reads across all active splits.
        let segment_source = self.segment_source.clone();
        let requests: Vec<(SplitId, crate::segments::SegmentId)> = self
            .active_splits
            .iter()
            .flat_map(|(&split_id, plan)| {
                plan.pending_segment_ids()
                    .into_iter()
                    .map(move |seg_id| (split_id, seg_id))
            })
            .collect();

        if requests.is_empty() {
            return Ok(());
        }

        // Create concurrent futures for all segment fetches.
        let futures: FuturesUnordered<_> = requests
            .into_iter()
            .map(|(split_id, seg_id)| {
                let segment_source = segment_source.clone();
                async move {
                    let handle = segment_source.request(seg_id).await?;
                    let buffer = handle.try_into_host_sync()?;
                    VortexResult::Ok((split_id, seg_id, buffer))
                }
                .boxed()
            })
            .collect();

        // Deliver buffers as they arrive.
        futures::pin_mut!(futures);
        while let Some(result) = futures.next().await {
            let (split_id, seg_id, buffer) = result?;
            if let Some(plan) = self.active_splits.get_mut(&split_id) {
                plan.complete_segment(seg_id, buffer);
            }
        }

        // Execute any nodes that became ready.
        Self::execute_all_ready(&mut self.active_splits)?;

        Ok(())
    }

    /// Moves completed split plans to the output queue.
    fn collect_completed(&mut self) {
        let completed: Vec<SplitId> = self
            .active_splits
            .iter()
            .filter(|(_, plan)| plan.is_complete())
            .map(|(&id, _)| id)
            .collect();

        for id in completed {
            let mut plan = self.active_splits.remove(&id).unwrap_or_else(|| {
                vortex_panic!("completed split {:?} not found in active_splits", id)
            });
            let output = plan.take_output();
            self.output_queue.push(id, output);
        }
    }

    /// Executes all ready nodes in all active plans until no more are ready.
    fn execute_all_ready(active_splits: &mut BTreeMap<SplitId, SplitPlan>) -> VortexResult<()> {
        for plan in active_splits.values_mut() {
            loop {
                let ready = plan.ready_nodes();
                if ready.is_empty() {
                    break;
                }
                for node_id in ready {
                    plan.execute_node(node_id)?;
                }
            }
        }
        Ok(())
    }
}
