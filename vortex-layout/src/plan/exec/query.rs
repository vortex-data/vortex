// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::BitAnd;
use std::ops::Range;

use bit_vec::BitVec;
use vortex_array::ExecutionCtx;
use vortex_array::VortexSessionExecute;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_buffer::BitBufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::plan::Concat;
use crate::plan::Eval;
use crate::plan::Filter;
use crate::plan::Pack;
use crate::plan::PlanRef;
use crate::plan::QueryPlan;
use crate::plan::Take;
use crate::plan::Zoned;
use crate::plan::exec::ExecNode;
use crate::plan::exec::NodeState;
use crate::plan::exec::Ready;
use crate::plan::exec::StepCx;
use crate::plan::exec::selection::Selection;
use crate::plan::exec::selection::join;

/// The port the projection feeds. Conjunct `i` feeds port `i + 1`.
const PROJECTION: usize = 0;

/// The selected fraction of a split at or above which a conjunct runs over every row of the
/// chunks holding a selected row and its result is intersected with the mask, rather than
/// running over the selected rows only.
///
/// Filtering an encoded column to a few rows before comparing is cheaper than comparing every
/// row, but the filter has a cost of its own, so a nearly full mask is not worth applying.
const EXPR_EVAL_THRESHOLD: f64 = 0.2;

/// Prunes a query's rows with the zone statistics of each conjunct's column, evaluates the
/// conjuncts one at a time, each under the rows the earlier ones kept, then streams its
/// projection under the rows that passed them all.
///
/// Each pruning plan and each conjunct is a child spawned once the previous one has closed,
/// since its selection is the previous one's result. A conjunct over a sparse mask is spawned
/// under that mask, so the scans beneath it filter their segments before the predicate runs. A
/// conjunct over a dense mask is spawned over every row of the chunks holding a selected row,
/// and its result is intersected afterwards; chunks with no selected row, such as those the
/// zones pruned, are never read. The projection streams through.
pub(crate) struct QueryNode {
    plan: QueryPlan,
    selection: Selection,
    session: VortexSession,
    ctx: ExecutionCtx,
    /// The rows that passed every pruning plan and conjunct evaluated so far.
    mask: Mask,
    phase: Phase,
    /// Conjuncts not yet evaluated.
    remaining: BitVec,
    /// The pruning plan or conjunct being evaluated, and the rows it was spawned under.
    current: Option<(usize, Spawned)>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// Pruning with the zones of the conjuncts from this index on.
    Pruning(usize),
    Conjuncts,
    Projecting,
}

/// The rows a conjunct was spawned under, so its result can be folded into the mask.
enum Spawned {
    /// The selected rows: the result has one value per selected row.
    Selected,
    /// Every row of the chunks holding a selected row: the result has one value per row this
    /// mask selects.
    Chunks(Mask),
}

impl QueryNode {
    pub(crate) fn new(plan: QueryPlan, selection: Selection, session: VortexSession) -> Self {
        let conjuncts = plan.conjunct_count();
        Self {
            plan,
            mask: selection.mask().clone(),
            selection,
            ctx: session.create_execution_ctx(),
            session,
            phase: Phase::Pruning(0),
            remaining: BitVec::from_elem(conjuncts, true),
            current: None,
        }
    }

    /// The port the pruning plan of conjunct `index` feeds.
    fn pruning_port(&self, index: usize) -> usize {
        1 + self.plan.conjunct_count() + index
    }

    /// Spawns the next conjunct's pruning plan over every row, or moves on to the conjuncts
    /// when no conjunct is left to prune with. Returns `Done` when nothing is selected.
    fn prune(&mut self, cx: &mut StepCx<'_>) -> VortexResult<NodeState> {
        if self.mask.all_false() {
            return Ok(NodeState::Done);
        }
        while let Phase::Pruning(index) = self.phase {
            if index == self.plan.conjunct_count() {
                self.phase = Phase::Conjuncts;
                break;
            }
            self.phase = Phase::Pruning(index + 1);
            if let Some(pruning) = self.plan.pruning(index, &self.session)? {
                self.current = Some((index, Spawned::Selected));
                let rows = self.selection.rows().clone();
                let mask = Mask::new_true(self.mask.len());
                cx.spawn(self.pruning_port(index), pruning, rows, mask);
                return Ok(NodeState::Wait);
            }
        }
        self.advance(cx)
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
                let conjunct = self.plan.conjunct(index)?;
                let (spawned, mask) = if self.mask.density() < EXPR_EVAL_THRESHOLD {
                    (Spawned::Selected, self.mask.clone())
                } else {
                    let chunks = chunks_with_selected_rows(&conjunct, &rows, &self.mask)?;
                    (Spawned::Chunks(chunks.clone()), chunks)
                };
                self.current = Some((index, spawned));
                cx.spawn(index + 1, conjunct, rows, mask);
            }
            None => {
                self.phase = Phase::Projecting;
                cx.spawn(PROJECTION, self.plan.projection()?, rows, self.mask.clone());
            }
        }
        Ok(NodeState::Wait)
    }

    /// Folds the closed pruning plan's result into the mask.
    fn narrow_pruned(&mut self, cx: &mut StepCx<'_>) -> VortexResult<()> {
        let (index, _) = self
            .current
            .take()
            .ok_or_else(|| vortex_err!("Query has no pruning plan in progress"))?;
        let port = self.pruning_port(index);
        let kept = join(
            &DType::Bool(Nullability::NonNullable),
            cx.input(port).take_all(),
        )?
        .execute::<Mask>(&mut self.ctx)?;
        let mask = std::mem::replace(&mut self.mask, Mask::new_false(0));
        self.mask = mask.bitand(&kept);
        Ok(())
    }

    /// Folds the closed conjunct's result into the mask and reports its selectivity.
    fn narrow(&mut self, cx: &mut StepCx<'_>) -> VortexResult<()> {
        let (index, spawned) = self
            .current
            .take()
            .ok_or_else(|| vortex_err!("Query has no conjunct in progress"))?;
        let conjunct = self.plan.conjunct(index)?;
        let result = join(conjunct.dtype(), cx.input(index + 1).take_all())?
            .fill_null(false)?
            .execute::<Mask>(&mut self.ctx)?;
        let input = self.mask.true_count();
        let mask = std::mem::replace(&mut self.mask, Mask::new_false(0));
        self.mask = match spawned {
            Spawned::Selected => mask.intersect_by_rank(&result),
            Spawned::Chunks(chunks) if chunks.all_true() => mask.bitand(&result),
            Spawned::Chunks(chunks) => mask.bitand(&chunks.intersect_by_rank(&result)),
        };
        if let Some(scheduler) = self.plan.scheduler() {
            scheduler.report_selectivity(index, self.mask.true_count() as f64 / input as f64);
        }
        Ok(())
    }
}

impl ExecNode for QueryNode {
    fn ready(&self) -> Ready {
        if self.phase == Phase::Projecting {
            Ready::Any
        } else {
            Ready::AllClosed
        }
    }

    fn start(&mut self, cx: &mut StepCx<'_>) -> VortexResult<NodeState> {
        self.prune(cx)
    }

    fn compute(&mut self, cx: &mut StepCx<'_>) -> VortexResult<NodeState> {
        match self.phase {
            Phase::Pruning(_) => {
                self.narrow_pruned(cx)?;
                self.prune(cx)
            }
            Phase::Conjuncts => {
                self.narrow(cx)?;
                self.advance(cx)
            }
            Phase::Projecting => {
                while let Some(array) = cx.input(PROJECTION).pop() {
                    cx.emit(array);
                }
                if cx.input(PROJECTION).finished() {
                    return Ok(NodeState::Done);
                }
                Ok(NodeState::Wait)
            }
        }
    }
}

/// The mask over `rows` selecting every row of the chunks of `plan` in which `mask` selects a
/// row, and no row of the others.
///
/// A conjunct spawned under it runs over whole chunks, as a dense result is cheaper to
/// intersect with than a dense mask is to filter by, while the chunks nothing selects, such as
/// those the zones pruned, are skipped.
fn chunks_with_selected_rows(plan: &PlanRef, rows: &Range<u64>, mask: &Mask) -> VortexResult<Mask> {
    let len = mask.len();
    let mut starts = Vec::new();
    chunk_starts(plan, rows, 0, &mut starts)?;
    starts.sort_unstable();
    starts.dedup();
    starts.retain(|&start| start > 0 && start < len);
    if starts.is_empty() {
        return Ok(Mask::new_true(len));
    }
    let mut bits = BitBufferMut::with_capacity(len);
    let mut start = 0;
    for end in starts.into_iter().chain([len]) {
        bits.append_n(!mask.slice(start..end).all_false(), end - start);
        start = end;
    }
    Ok(Mask::from_buffer(bits.freeze()))
}

/// Collects where the chunks of `plan` overlapping `rows` begin, as positions from the start
/// of the outermost range, descending through the operators that keep their child's row domain.
/// `offset` is where `rows` begins in that frame.
fn chunk_starts(
    plan: &PlanRef,
    rows: &Range<u64>,
    offset: usize,
    starts: &mut Vec<usize>,
) -> VortexResult<()> {
    if let Some(concat) = plan.as_opt::<Concat>() {
        let offsets = concat.row_offsets();
        let first = offsets
            .partition_point(|&o| o <= rows.start)
            .saturating_sub(1);
        let end = offsets.partition_point(|&o| o < rows.end);
        for index in first..end {
            let chunk_start = offsets[index];
            let chunk_end = offsets
                .get(index + 1)
                .copied()
                .unwrap_or_else(|| concat.row_count());
            let local = rows.start.max(chunk_start)..rows.end.min(chunk_end);
            if local.start >= local.end {
                continue;
            }
            let position = offset + usize::try_from(local.start - rows.start)?;
            starts.push(position);
            chunk_starts(
                &concat.child_required(index)?,
                &(local.start - chunk_start..local.end - chunk_start),
                position,
                starts,
            )?;
        }
        return Ok(());
    }
    let children: Vec<PlanRef> = if let Some(eval) = plan.as_opt::<Eval>() {
        vec![eval.child_plan()?]
    } else if let Some(filter) = plan.as_opt::<Filter>() {
        vec![filter.child_plan()?]
    } else if let Some(zoned) = plan.as_opt::<Zoned>() {
        zoned.data_plan()?.into_iter().collect()
    } else if let Some(take) = plan.as_opt::<Take>() {
        vec![take.codes()?]
    } else if plan.is::<Pack>() {
        plan.children().iter().collect::<VortexResult<_>>()?
    } else {
        return Ok(());
    };
    for child in &children {
        chunk_starts(child, rows, offset, starts)?;
    }
    Ok(())
}
