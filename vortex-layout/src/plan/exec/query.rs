// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::BitAnd;

use bit_vec::BitVec;
use vortex_array::ExecutionCtx;
use vortex_array::VortexSessionExecute;
use vortex_array::builtins::ArrayBuiltins;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::plan::QueryPlan;
use crate::plan::exec::ExecNode;
use crate::plan::exec::NodeState;
use crate::plan::exec::Ready;
use crate::plan::exec::StepCx;
use crate::plan::exec::selection::Selection;
use crate::plan::exec::selection::join;

/// The port the projection feeds. Conjunct `i` feeds port `i + 1`.
const PROJECTION: usize = 0;

/// The selected fraction of a split at or above which a conjunct runs over every row and its
/// result is intersected with the mask, rather than running over the selected rows only.
///
/// Filtering an encoded column to a few rows before comparing is cheaper than comparing every
/// row, but the filter has a cost of its own, so a nearly full mask is not worth applying.
const EXPR_EVAL_THRESHOLD: f64 = 0.2;

/// Evaluates a query's conjuncts one at a time, each under the rows the earlier ones kept, then
/// its projection under the rows that passed them all.
///
/// Each conjunct is a child spawned once the previous one has closed, since its selection is
/// the previous one's result. A conjunct over a sparse mask is spawned under that mask, so the
/// scans beneath it filter their segments before the predicate runs; a conjunct over a dense
/// mask is spawned over every row and intersected afterwards. The projection streams through.
pub(crate) struct QueryNode {
    plan: QueryPlan,
    selection: Selection,
    ctx: ExecutionCtx,
    /// The rows that passed every conjunct evaluated so far.
    mask: Mask,
    /// Conjuncts not yet evaluated.
    remaining: BitVec,
    /// The conjunct being evaluated, and whether it was spawned under the mask or over every
    /// row.
    current: Option<(usize, bool)>,
    projecting: bool,
}

impl QueryNode {
    pub(crate) fn new(plan: QueryPlan, selection: Selection, session: VortexSession) -> Self {
        let conjuncts = plan.conjunct_count();
        Self {
            plan,
            mask: selection.mask().clone(),
            selection,
            ctx: session.create_execution_ctx(),
            remaining: BitVec::from_elem(conjuncts, true),
            current: None,
            projecting: false,
        }
    }

    /// Spawns the next conjunct the scheduler prefers, or the projection when none is left.
    /// Returns `Done` when nothing is selected.
    fn advance(&mut self, cx: &mut StepCx<'_>) -> VortexResult<NodeState> {
        if self.mask.all_false() {
            return Ok(NodeState::Done);
        }
        let next = self
            .plan
            .scheduler()
            .and_then(|scheduler| scheduler.next_conjunct(&self.remaining));
        let rows = self.selection.rows().clone();
        match next {
            Some(index) => {
                self.remaining.set(index, false);
                let under_mask = self.mask.density() < EXPR_EVAL_THRESHOLD;
                let mask = if under_mask {
                    self.mask.clone()
                } else {
                    Mask::new_true(self.mask.len())
                };
                self.current = Some((index, under_mask));
                cx.spawn(index + 1, self.plan.conjunct(index)?, rows, mask);
            }
            None => {
                self.projecting = true;
                cx.spawn(PROJECTION, self.plan.projection()?, rows, self.mask.clone());
            }
        }
        Ok(NodeState::Wait)
    }

    /// Folds the closed conjunct's result into the mask and reports its selectivity.
    fn narrow(&mut self, cx: &mut StepCx<'_>) -> VortexResult<()> {
        let (index, under_mask) = self
            .current
            .take()
            .ok_or_else(|| vortex_err!("Query has no conjunct in progress"))?;
        let conjunct = self.plan.conjunct(index)?;
        let result = join(conjunct.dtype(), cx.input(index + 1).take_all())?
            .fill_null(false)?
            .execute::<Mask>(&mut self.ctx)?;
        let input = self.mask.true_count();
        let mask = std::mem::replace(&mut self.mask, Mask::new_false(0));
        self.mask = if under_mask {
            mask.intersect_by_rank(&result)
        } else {
            mask.bitand(&result)
        };
        if let Some(scheduler) = self.plan.scheduler() {
            scheduler.report_selectivity(index, self.mask.true_count() as f64 / input as f64);
        }
        Ok(())
    }
}

impl ExecNode for QueryNode {
    fn ready(&self) -> Ready {
        if self.projecting {
            Ready::Any
        } else {
            Ready::AllClosed
        }
    }

    fn start(&mut self, cx: &mut StepCx<'_>) -> VortexResult<NodeState> {
        self.advance(cx)
    }

    fn compute(&mut self, cx: &mut StepCx<'_>) -> VortexResult<NodeState> {
        if self.projecting {
            for array in cx.input(PROJECTION).take_all() {
                cx.emit(array);
            }
            if cx.input(PROJECTION).finished() {
                return Ok(NodeState::Done);
            }
            return Ok(NodeState::Wait);
        }
        self.narrow(cx)?;
        self.advance(cx)
    }
}
