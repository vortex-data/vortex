// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub(crate) mod plan;

mod output;
pub mod planner;
pub mod shim;
mod split;

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::sync::Arc;

use planner::PlanBuilder;
use planner::SplitPlannerRef;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::expr::Expression;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexError;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use self::output::OutputQueue;
use self::plan::Plan;
use self::split::SplitRange;
use self::split::form_splits;
use crate::segments::SegmentId;
use crate::segments::SegmentSource;
use crate::v2::layout::LayoutRef;
use crate::v2::scan::planner::ComputeArgs;
use crate::v2::scan::planner::ComputeFn;
use crate::v2::scan::planner::NodeId;
use crate::v2::scan::planner::NodeOpts;
use crate::v2::scan::split::SplitId;
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
/// The scan is a pure state machine: it never performs I/O or computation itself. Instead,
/// the external driver calls [`actions`](Self::actions) to get work items, performs them,
/// and reports completions via [`event`](Self::post_event).
pub struct Scan {
    project_planner: SplitPlannerRef,
    filter_planner: Option<SplitPlannerRef>,
    splits: Vec<SplitRange>,
    config: ScanConfig,
    session: VortexSession,
    state: State,
}

struct State {
    next_split_to_plan: usize,
    plan: Plan,
    plan_ahead_kb: u64,
    active_splits: BTreeMap<SplitId, NodeId>,
    output_queue: OutputQueue,

    /// Maps a `ReadId` back to (node_id, segment_slot) so we can resolve the right input.
    read_dispatch: HashMap<ReadId, (NodeId, usize)>,
    /// Maps a `ComputeId` back to the node that was dispatched.
    compute_dispatch: HashMap<ComputeId, NodeId>,
    /// Eagerly-dispatched segment reads accumulated during planning.
    pending_reads: Vec<ScanAction>,

    next_read_id: u32,
    next_compute_id: u32,
    /// Tracks the plan length before each planning round so we can find newly-added nodes.
    nodes_before_plan: usize,
}

impl Scan {
    /// Creates a new scan for the given layout and expression.
    pub fn try_new(
        layout: &LayoutRef,
        projection: &Expression,
        filter: Option<&Expression>,
        selection: &Selection,
        config: ScanConfig,
        session: &VortexSession,
    ) -> VortexResult<Self> {
        // Create the row splits set, this is populated in the `prepare` call to give us hints
        // on how to partition the scan.
        let mut row_splits = BTreeSet::new();
        row_splits.insert(0);
        row_splits.insert(layout.row_count());

        // Create split planners for the project / filter expressions.
        let project_planner = layout.prepare(projection, selection, &mut row_splits, session)?;
        let filter_planner = filter
            .map(|f| layout.prepare(f, selection, &mut row_splits, session))
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

        let total_splits = splits.len() as u32;

        Ok(Self {
            project_planner,
            filter_planner,
            splits,
            config,
            session: session.clone(),
            state: State {
                next_split_to_plan: 0,
                plan: Plan::new(),
                plan_ahead_kb: 0,
                active_splits: BTreeMap::new(),
                output_queue: OutputQueue::new(total_splits),
                read_dispatch: HashMap::new(),
                compute_dispatch: HashMap::new(),
                pending_reads: Vec::new(),
                next_read_id: 0,
                next_compute_id: 0,
                nodes_before_plan: 0,
            },
        })
    }

    /// Returns the pending actions that the driver should execute.
    ///
    /// The driver loop should:
    /// 1. Call `actions()` to get work items.
    /// 2. For each `ReadSegment` / `Compute`, perform the work and call `event()` with the result.
    /// 3. For each `Emit`, forward the result to the consumer.
    /// 4. On `Done`, stop.
    pub fn actions(&mut self) -> VortexResult<Vec<ScanAction>> {
        // Ensure we have planned enough splits to fill the plan-ahead window.
        self.fill_plan_ahead()?;

        let mut actions = Vec::new();

        // 1. Drain eagerly-dispatched segment reads from planning.
        actions.append(&mut self.state.pending_reads);

        // 2. Dispatch compute for any Ready nodes.
        // Collect ready node IDs first to avoid borrow conflict.
        let ready: Vec<NodeId> = self
            .state
            .plan
            .ready_nodes_in_range(0..self.state.plan.len())
            .collect();

        for node_id in ready {
            if self.state.plan.node_has_compute(node_id) {
                let (compute, segments, inputs) = self.state.plan.take_compute(node_id)?;
                let compute_id = ComputeId(self.state.next_compute_id);
                self.state.next_compute_id += 1;
                self.state.compute_dispatch.insert(compute_id, node_id);
                actions.push(ScanAction::Compute {
                    compute_id,
                    compute,
                    segments,
                    inputs,
                    session: self.session.clone(),
                });
            }
        }

        // 3. Drain any ready split results from the output queue.
        for (split_id, result) in self.state.output_queue.drain_ready() {
            self.state.active_splits.remove(&split_id);
            actions.push(ScanAction::Emit { split_id, result });
        }

        // 4. Check if the scan is complete.
        if self.state.output_queue.is_complete() && self.state.active_splits.is_empty() {
            actions.push(ScanAction::Done);
        }

        Ok(actions)
    }

    /// Reports an event from the driver back to the scan.
    pub fn post_event(&mut self, event: ScanEvent) -> VortexResult<()> {
        match event {
            ScanEvent::SegmentReady { read_id, buffer } => {
                let (node_id, slot) = self
                    .state
                    .read_dispatch
                    .remove(&read_id)
                    .expect("Unknown read_id");
                self.state.plan.resolve_segment(node_id, slot, buffer);

                // If all deps resolved and node has no compute, complete it immediately
                // with the single buffer. Otherwise it becomes Ready for compute dispatch.
                if self.state.plan.node_pending_deps(node_id) == 0
                    && !self.state.plan.node_has_compute(node_id)
                {
                    // Segment-only node with no compute: shouldn't normally happen
                    // since add_node always requires compute. But handle gracefully.
                }
                // If deps hit 0, resolve_segment already transitions to Ready.
                // Propagation to dependents happens when the node completes (after compute).
            }
            ScanEvent::ComputeReady { compute_id, result } => {
                let node_id = self
                    .state
                    .compute_dispatch
                    .remove(&compute_id)
                    .expect("Unknown compute_id");

                self.state.plan.complete_node(node_id, result.clone());

                // If this node is a split root, push its result to the output queue.
                // We check if any active split points to this node.
                let mut completed_split = None;
                for (&split_id, &root_node) in &self.state.active_splits {
                    if root_node.as_usize() == node_id.as_usize() {
                        completed_split = Some(split_id);
                        break;
                    }
                }
                if let Some(split_id) = completed_split {
                    self.state.output_queue.push(split_id, Some(result.clone()));
                }

                // Propagate to downstream dependents.
                let dependents: Vec<(NodeId, usize)> =
                    self.state.plan.dependents_of(node_id).to_vec();
                for (downstream_id, slot) in dependents {
                    self.state
                        .plan
                        .resolve_input(downstream_id, slot, result.clone());
                }
            }
            ScanEvent::SegmentFailed { error, .. } => {
                return Err(error);
            }
            ScanEvent::ComputeFailed { error, .. } => {
                return Err(error);
            }
        }
        Ok(())
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

            let split_range = self.splits[self.state.next_split_to_plan].clone();
            self.state.nodes_before_plan = self.state.plan.len();

            let result_node = {
                let mut plan_builder = PlanBuilder::new(&mut self.state.plan);

                // Start with the initial row selection.
                let mut selection =
                    plan_builder.create_node_resolved(split_range.mask.into_array());

                // Map through the filter planner.
                if let Some(filter_planner) = &self.filter_planner {
                    let filter_result = filter_planner.plan_split(
                        &split_range.row_range,
                        // FIXME(ngates): note that we always pass down the initial selection.
                        //  We should instead pass separate selection and definition masks.
                        //  That way we can have the filter plan return a refined mask, rather
                        //  than one with lower cardinality.
                        selection,
                        &mut plan_builder,
                    )?;

                    // FIXME(ngates): per the comment above, we currently need to perform a rank
                    //  intersection after the fact on the filter result.
                    selection = plan_builder.create_node(NodeOpts {
                        inputs: &[selection, filter_result],
                        segments: vec![],
                        lifetime: plan_builder.row_range_lifetime(split_range.row_range.clone()),
                        compute: move |mut args: ComputeArgs| {
                            let mut arrays = args.inputs.into_iter();
                            let initial_selection = arrays.next().vortex_expect("missing");
                            let filter_result = arrays.next().vortex_expect("missing");

                            let initial_selection =
                                initial_selection.execute::<Mask>(&mut args.ctx)?;
                            let filter_result = filter_result.execute::<Mask>(&mut args.ctx)?;

                            let refined_selection =
                                initial_selection.intersect_by_rank(&filter_result);
                            Ok(refined_selection.into_array())
                        },
                    })?;
                }

                // And plan the projection.
                self.project_planner.plan_split(
                    &split_range.row_range,
                    selection,
                    &mut plan_builder,
                )?
            };

            self.state.next_split_to_plan += 1;
            self.state.active_splits.insert(split_range.id, result_node);

            // Eagerly dispatch segment reads for newly-added nodes.
            let new_range = self.state.nodes_before_plan..self.state.plan.len();
            for i in new_range {
                let node_id = NodeId::new(i);
                let seg_count = self.state.plan.node_segment_count(node_id);
                if seg_count > 0 {
                    let segments = self.state.plan.take_segment_requests(node_id)?;
                    for (slot, seg) in segments.into_iter().enumerate() {
                        let read_id = ReadId(self.state.next_read_id);
                        self.state.next_read_id += 1;
                        self.state.read_dispatch.insert(read_id, (node_id, slot));
                        self.state.pending_reads.push(ScanAction::ReadSegment {
                            read_id,
                            source: seg.source,
                            segment_id: seg.segment_id,
                        });
                    }
                    // TODO: accumulate segment byte estimates into plan_ahead_kb.
                }
            }
        }
    }
}

/// Identifies a segment read dispatched to the driver.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ReadId(pub(crate) u32);

/// Identifies a compute task dispatched to the driver.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ComputeId(pub(crate) u32);

/// An action the driver must perform on behalf of the scan.
pub enum ScanAction {
    /// Read the segment identified by `segment_id` from `source`, then report back with `read_id`.
    ReadSegment {
        read_id: ReadId,
        source: Arc<dyn SegmentSource>,
        segment_id: SegmentId,
    },
    /// Execute the compute function with the given inputs, then report back with `compute_id`.
    Compute {
        compute_id: ComputeId,
        compute: ComputeFn,
        segments: Vec<ByteBuffer>,
        inputs: Vec<ArrayRef>,
        session: VortexSession,
    },
    /// A split result is ready for the consumer.
    Emit {
        split_id: SplitId,
        result: Option<ArrayRef>,
    },
    /// The scan is complete—all splits have been emitted.
    Done,
}

/// An event the driver reports back to the scan after completing an action.
pub enum ScanEvent {
    /// A segment read completed successfully.
    SegmentReady { read_id: ReadId, buffer: ByteBuffer },
    /// A compute task completed successfully.
    ComputeReady {
        compute_id: ComputeId,
        result: ArrayRef,
    },
    /// A segment read failed.
    SegmentFailed { read_id: ReadId, error: VortexError },
    /// A compute task failed.
    ComputeFailed {
        compute_id: ComputeId,
        error: VortexError,
    },
}
