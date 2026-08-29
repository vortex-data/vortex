// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::VecDeque;
use std::ops::BitAnd;
use std::ops::Range;

use vortex_array::VortexSessionExecute;
use vortex_array::expr::BoundExpression;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::Mask;

use crate::node::ActivationRows;
use crate::node::ActivationTarget;
use crate::node::ChildPoll;
use crate::node::ExecCx;
use crate::node::ExecNode;
use crate::node::ExecPoll;
use crate::node::NodeId;
use crate::node::NodeState;
use crate::node::PlanCx;
use crate::node::PlanItem;
use crate::node::PlanPoll;
use crate::node::PushBatch;
use crate::node::PushCx;
use crate::node::RetireCx;
use crate::node::StageOutput;
use crate::node::Value;
use crate::node::ValueBatch;
use crate::nodes::EXPR_EVAL_THRESHOLD;

/// One conjunct: the subtree producing its input, and the predicate applied to that input.
pub struct ConjunctSlot {
    /// The node producing the fields this predicate reads.
    pub input: NodeId,
    /// The predicate, bound to the input subtree's output dtype.
    pub predicate: BoundExpression,
}

/// How the conjuncts of one filter relate to each other.
///
/// This is the whole of the cascade-versus-parallel policy: the operators are identical, only
/// the demand each conjunct sees differs.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ConjunctMode {
    /// Each conjunct sees the mask the previous one produced, and an all-false mask ends the
    /// morsel early. Fewer rows read; a serial dependency between conjuncts.
    Cascade,
    /// Every conjunct sees the incoming mask, and the results are intersected. More rows read;
    /// no dependency between conjuncts.
    Parallel,
}

/// The demand spine: predicate evaluations feeding one intersection.
pub struct ConjunctExec {
    slots: Vec<ConjunctSlot>,
    push_predicates: Vec<Option<BoundExpression>>,
    mode: ConjunctMode,

    // Per-morsel state.
    range: Range<u64>,
    plan_cursor: usize,
    plan_started: bool,
    exec_cursor: usize,
    incoming: Option<Mask>,
    mask: Option<Mask>,
    done: bool,
    children: Vec<NodeId>,
    push_cursor: u64,
    push_heads: Vec<VecDeque<PendingMask>>,
    push_ended: Vec<bool>,
    push_saw_batch: Vec<bool>,
    push_received: Vec<u64>,
    cascade_active: usize,
    cascade_stage: VecDeque<PendingMask>,
    cascade_ready: VecDeque<PendingMask>,
    push_output_credit: bool,
    cascade_skipped_from: Option<usize>,
}

struct PendingMask {
    batch: PushBatch,
    last: bool,
}

impl ConjunctExec {
    /// Build a conjunct node.
    pub fn new(slots: Vec<ConjunctSlot>, mode: ConjunctMode) -> Self {
        let push_predicates = vec![None; slots.len()];
        Self::new_with_push_predicates(slots, push_predicates, mode)
    }

    pub(crate) fn new_with_push_predicates(
        slots: Vec<ConjunctSlot>,
        push_predicates: Vec<Option<BoundExpression>>,
        mode: ConjunctMode,
    ) -> Self {
        debug_assert_eq!(slots.len(), push_predicates.len());
        let children = slots.iter().map(|slot| slot.input).collect();
        Self {
            slots,
            push_predicates,
            mode,
            range: 0..0,
            plan_cursor: 0,
            plan_started: false,
            exec_cursor: 0,
            incoming: None,
            mask: None,
            done: false,
            children,
            push_cursor: 0,
            push_heads: Vec::new(),
            push_ended: Vec::new(),
            push_saw_batch: Vec::new(),
            push_received: Vec::new(),
            cascade_active: 0,
            cascade_stage: VecDeque::new(),
            cascade_ready: VecDeque::new(),
            push_output_credit: true,
            cascade_skipped_from: None,
        }
    }

    /// Evaluate one conjunct under `incoming`, returning the refined mask.
    fn eval(
        &self,
        idx: usize,
        incoming: &Mask,
        cx: &mut ExecCx<'_>,
    ) -> VortexResult<ChildPoll<Mask>> {
        let slot = &self.slots[idx];

        // The regime switch: over a sparse mask, filter first and correct by rank; over a dense
        // one, evaluate the whole range and intersect. Same choice the V1 flat reader makes.
        let sparse = incoming.density() < EXPR_EVAL_THRESHOLD;
        let child_demand = if sparse {
            incoming.clone()
        } else {
            Mask::new_true(incoming.len())
        };

        let array = match cx.child_array(slot.input, child_demand)? {
            ChildPoll::Value(array) => array,
            ChildPoll::Blocked(waits) => return Ok(ChildPoll::Blocked(waits)),
            ChildPoll::Done => {
                return Err(vortex_err!(
                    "conjunct input {} produced no value",
                    slot.input
                ));
            }
        };
        let array = array.apply_bound(&slot.predicate)?;
        let mut ctx = cx.session().create_execution_ctx();
        let predicate_mask = array.null_as_false().execute(&mut ctx)?;

        Ok(ChildPoll::Value(if sparse {
            incoming.intersect_by_rank(&predicate_mask)
        } else {
            incoming.bitand(&predicate_mask)
        }))
    }
}

impl ExecNode for ConjunctExec {
    fn push_profile_kind(&self) -> crate::node::PushProfileKind {
        crate::node::PushProfileKind::Conjunct
    }

    fn reset(&mut self, range: Range<u64>) {
        self.range = range;
        self.plan_cursor = 0;
        self.plan_started = false;
        self.exec_cursor = 0;
        self.incoming = None;
        self.mask = None;
        self.done = false;
        self.push_cursor = self.range.start;
        let width = self.slots.len();
        self.push_heads.truncate(width);
        self.push_heads.resize_with(width, VecDeque::new);
        self.push_heads.iter_mut().for_each(VecDeque::clear);
        self.push_ended.resize(width, false);
        self.push_ended.fill(false);
        self.push_saw_batch.resize(width, false);
        self.push_saw_batch.fill(false);
        self.push_received.resize(width, self.range.start);
        self.push_received.fill(self.range.start);
        self.cascade_active = 0;
        self.cascade_stage.clear();
        self.cascade_ready.clear();
        self.push_output_credit = true;
        self.cascade_skipped_from = None;
    }

    fn next_plan(&mut self, cx: &mut PlanCx<'_>) -> VortexResult<PlanPoll> {
        // Emit-once planning: every conjunct's IO is named up front, whatever the mode. Under
        // cascade a later conjunct may turn out not to be needed, but a use is named before its
        // demand is known — refining it after emission is P2's cancellation path, not a reason
        // to defer naming it here.
        while self.plan_cursor < self.slots.len() {
            if cx.out_of_budget() {
                return Ok(PlanPoll::Item(PlanItem::Plan));
            }
            let fresh = !self.plan_started;
            self.plan_started = true;
            if cx.plan_child(
                self.slots[self.plan_cursor].input,
                self.range.clone(),
                fresh,
            )? {
                self.plan_cursor += 1;
                self.plan_started = false;
            } else {
                return Ok(PlanPoll::Item(PlanItem::Plan));
            }
        }
        Ok(PlanPoll::Complete)
    }

    fn execute(&mut self, cx: &mut ExecCx<'_>) -> VortexResult<ExecPoll> {
        if self.done {
            return Ok(ExecPoll::Done);
        }
        if self.incoming.is_none() {
            let incoming = cx.demand().clone();
            self.mask = Some(incoming.clone());
            self.incoming = Some(incoming);
        }

        while self.exec_cursor < self.slots.len() {
            let eval_demand = match self.mode {
                ConjunctMode::Cascade => self.mask.as_ref(),
                ConjunctMode::Parallel => self.incoming.as_ref(),
            }
            .vortex_expect("execution masks initialized")
            .clone();
            if self.mode == ConjunctMode::Cascade && eval_demand.all_false() {
                cx.stats().conjuncts_short_circuited +=
                    (self.slots.len() - self.exec_cursor) as u64;
                self.exec_cursor = self.slots.len();
                break;
            }

            match self.eval(self.exec_cursor, &eval_demand, cx)? {
                ChildPoll::Value(refined) => {
                    if self.mode == ConjunctMode::Parallel {
                        self.mask = Some(
                            self.mask
                                .take()
                                .vortex_expect("execution mask initialized")
                                .bitand(&refined),
                        );
                    } else {
                        self.mask = Some(refined);
                    }
                    self.exec_cursor += 1;
                }
                ChildPoll::Blocked(waits) => return Ok(ExecPoll::Blocked(waits)),
                ChildPoll::Done => {
                    return Err(vortex_err!(
                        "conjunct {} produced no value",
                        self.exec_cursor
                    ));
                }
            }
        }

        let mask = self.mask.take().vortex_expect("execution mask initialized");
        self.incoming = None;
        self.done = true;

        Ok(ExecPoll::Value(ValueBatch {
            coverage: self.range.clone(),
            value: Value::Mask(mask),
        }))
    }

    #[inline]
    fn push_input(
        &mut self,
        port: crate::node::InputPort,
        batch: PushBatch,
        last_for_input: bool,
        cx: &mut PushCx<'_>,
        out: &mut StageOutput,
    ) -> VortexResult<NodeState> {
        if let Some(state) = accept_conjunct_input(self, port, batch, last_for_input, cx, out)? {
            return Ok(state);
        }
        drain_parallel(self, out)
    }

    #[inline]
    fn push_end(
        &mut self,
        port: crate::node::InputPort,
        _cx: &mut PushCx<'_>,
        out: &mut StageOutput,
    ) -> VortexResult<NodeState> {
        accept_conjunct_end(self, port)?;
        drain_push(self, out)
    }

    #[inline]
    fn push_resume(
        &mut self,
        _cx: &mut PushCx<'_>,
        out: &mut StageOutput,
    ) -> VortexResult<NodeState> {
        drain_push(self, out)
    }

    #[inline]
    fn push_credit(
        &mut self,
        _cx: &mut PushCx<'_>,
        out: &mut StageOutput,
    ) -> VortexResult<NodeState> {
        if self.push_output_credit {
            return Err(vortex_err!("conjunct received duplicate output credit"));
        }
        self.push_output_credit = true;
        drain_push(self, out)
    }

    fn retire(&mut self, cx: &mut RetireCx<'_>) {
        for &child in &self.children {
            cx.retire_child(child);
        }
    }

    fn children(&self) -> &[NodeId] {
        &self.children
    }
}

fn accept_conjunct_end(node: &mut ConjunctExec, port: crate::node::InputPort) -> VortexResult<()> {
    let idx = port.index();
    if idx >= node.slots.len() {
        return Err(vortex_err!("conjunct received unknown slot port {idx}"));
    }
    if node.push_ended[idx] {
        return Err(vortex_err!("conjunct slot {idx} ended more than once"));
    }
    if node.push_saw_batch[idx] {
        return Err(vortex_err!(
            "conjunct slot {idx} used End after emitting a batch"
        ));
    }
    let accounted = node.push_received[idx];
    if accounted != node.range.end {
        return Err(vortex_err!(
            "conjunct slot {idx} ended without accounting for {accounted}..{}",
            node.range.end
        ));
    }
    node.push_ended[idx] = true;
    Ok(())
}

fn accept_conjunct_input(
    node: &mut ConjunctExec,
    port: crate::node::InputPort,
    batch: PushBatch,
    last_for_input: bool,
    cx: &mut PushCx<'_>,
    out: &mut StageOutput,
) -> VortexResult<Option<NodeState>> {
    let idx = port.index();
    if idx >= node.slots.len() {
        return Err(vortex_err!("conjunct received unknown slot port {idx}"));
    }
    if node.mode == ConjunctMode::Cascade
        && node
            .cascade_skipped_from
            .is_some_and(|first_skipped| idx >= first_skipped)
    {
        if !batch.selection.all_false() {
            return Err(vortex_err!(
                "skipped cascade slot {idx} received a non-empty activation"
            ));
        }
        if batch.coverage.start != node.push_received[idx]
            || batch.coverage.end > node.range.end
            || (last_for_input && batch.coverage.end != node.range.end)
            || (!last_for_input && batch.coverage.end == node.range.end)
        {
            return Err(vortex_err!(
                "skipped cascade slot {idx} produced invalid coverage {:?}",
                batch.coverage
            ));
        }
        node.push_received[idx] = batch.coverage.end;
        node.push_ended[idx] = last_for_input;
        node.push_saw_batch[idx] = true;
        if !last_for_input {
            out.push_consumed(port);
        }
        return Ok(Some(NodeState::NeedInput));
    }
    if node.push_ended[idx] {
        return Err(vortex_err!(
            "conjunct slot {idx} produced data after ending"
        ));
    }
    if (node.mode == ConjunctMode::Parallel || node.slots.len() == 1)
        && !node.push_heads[idx].is_empty()
    {
        return Err(vortex_err!(
            "conjunct slot {idx} exceeded its one-batch edge credit"
        ));
    }
    if node.mode == ConjunctMode::Cascade && idx != node.cascade_active {
        return Err(vortex_err!(
            "cascade received slot {idx} while slot {} is active",
            node.cascade_active
        ));
    }
    if batch.coverage.start != node.push_received[idx] || batch.coverage.end > node.range.end {
        return Err(vortex_err!(
            "conjunct slot {idx} produced non-contiguous coverage {:?}; expected {}",
            batch.coverage,
            node.push_received[idx]
        ));
    }
    if last_for_input && batch.coverage.end != node.range.end {
        return Err(vortex_err!(
            "conjunct slot {idx} ended at {}, expected {}",
            batch.coverage.end,
            node.range.end
        ));
    }
    if !last_for_input && batch.coverage.end == node.range.end {
        return Err(vortex_err!(
            "conjunct slot {idx} covered its full range without a final marker"
        ));
    }
    node.push_received[idx] = batch.coverage.end;
    node.push_ended[idx] = last_for_input;
    node.push_saw_batch[idx] = true;
    let PushBatch {
        coverage,
        selection,
        materialized,
        value,
    } = batch;
    let array = value.into_array()?;
    let predicate = node.push_predicates[idx]
        .as_ref()
        .unwrap_or(&node.slots[idx].predicate);
    let array = array.apply_bound(predicate)?;
    let mut ctx = cx.session().create_execution_ctx();
    let predicate = array.null_as_false().execute(&mut ctx)?;
    if predicate.len() != materialized.true_count() {
        return Err(vortex_err!(
            "conjunct {idx} predicate length {} does not match dense input length {}",
            predicate.len(),
            materialized.true_count()
        ));
    }
    let refined = refine_materialized_predicate(&selection, &materialized, predicate);
    let pushed =
        PushBatch::from_validated_parts(coverage, selection, materialized, Value::Mask(refined));
    if node.slots.len() == 1 {
        let pending = PendingMask {
            batch: pushed,
            last: last_for_input,
        };
        if node.push_output_credit {
            return Ok(Some(emit_single(node, pending, out)?));
        }
        node.push_heads[idx].push_back(pending);
        return Ok(Some(NodeState::NeedInput));
    }
    if node.mode == ConjunctMode::Cascade {
        if !last_for_input {
            out.push_consumed(port);
        }
        node.cascade_stage.push_back(PendingMask {
            batch: pushed,
            last: last_for_input,
        });
        if !last_for_input {
            return Ok(Some(NodeState::NeedInput));
        }
        let all_false = node
            .cascade_stage
            .iter()
            .all(|pending| matches!(&pending.batch.value, Value::Mask(mask) if mask.all_false()));
        if idx + 1 == node.slots.len() {
            node.cascade_ready.extend(node.cascade_stage.drain(..));
            node.cascade_active = node.slots.len();
        } else if all_false {
            cx.stats().conjuncts_short_circuited += (node.slots.len() - idx - 1) as u64;
            node.cascade_skipped_from = Some(idx + 1);
            enqueue_all_false_closure(idx + 1, node.slots.len(), &node.cascade_stage, out);
            node.cascade_ready.extend(node.cascade_stage.drain(..));
            node.cascade_active = node.slots.len();
        } else {
            let materialize_coverage = cascade_materializes_coverage(&node.cascade_stage);
            for pending in node.cascade_stage.drain(..) {
                let Value::Mask(selection) = pending.batch.value else {
                    unreachable!("conjunct evaluation produces a mask")
                };
                let rows = if materialize_coverage {
                    ActivationRows::try_new(
                        selection,
                        Mask::new_true(pending.batch.selection.len()),
                    )?
                } else {
                    ActivationRows::selected(selection)
                };
                out.push_gate(
                    ActivationTarget::PredicateSlot(idx + 1),
                    pending.batch.coverage,
                    rows,
                );
            }
            node.cascade_active += 1;
        }
        return Ok(Some(drain_cascade(node, out)?));
    }
    node.push_heads[idx].push_back(PendingMask {
        batch: pushed,
        last: last_for_input,
    });
    Ok(None)
}

/// Map a predicate result from its dense evaluation domain back into coverage coordinates.
///
/// The executor deliberately has two common regimes. Sparse evaluation materializes exactly the
/// selected rows and needs rank expansion; dense evaluation materializes the whole coverage and
/// needs a bitwise intersection. Keeping these branches exclusive avoids expanding a dense mask
/// only to intersect it immediately afterward. A mixed materialization domain can arise when
/// differently-shaped fragments are coalesced, so retain a direct one-pass fallback for it.
#[inline]
fn refine_materialized_predicate(selection: &Mask, materialized: &Mask, predicate: Mask) -> Mask {
    debug_assert_eq!(selection.len(), materialized.len());
    debug_assert_eq!(predicate.len(), materialized.true_count());
    if materialized.all_true() {
        // Keep `selection` for the outgoing batch and consume the freshly-evaluated predicate.
        // This reuses its owned bitmap rather than allocating another intersection buffer.
        return predicate.bitand(selection);
    }
    // PushBatch guarantees `selection` is a subset of `materialized`, so equal cardinality proves
    // equality without scanning two sliced bitmap views.
    if selection.true_count() == materialized.true_count() {
        return selection.intersect_by_rank(&predicate);
    }

    let mut predicate = predicate.iter();
    Mask::from_iter(
        selection
            .iter()
            .zip(materialized.iter())
            .map(|(selected, materialized)| {
                let evaluated = if materialized {
                    predicate.next().vortex_expect("predicate domain validated")
                } else {
                    false
                };
                selected && evaluated
            }),
    )
}

fn drain_push(node: &mut ConjunctExec, out: &mut StageOutput) -> VortexResult<NodeState> {
    if node.slots.len() == 1 {
        drain_single(node, out)
    } else if node.mode == ConjunctMode::Cascade {
        drain_cascade(node, out)
    } else {
        drain_parallel(node, out)
    }
}

fn drain_single(node: &mut ConjunctExec, out: &mut StageOutput) -> VortexResult<NodeState> {
    if !node.push_output_credit {
        return Ok(NodeState::NeedInput);
    }
    let Some(pending) = node.push_heads[0].pop_front() else {
        return Ok(NodeState::NeedInput);
    };
    emit_single(node, pending, out)
}

fn emit_single(
    node: &mut ConjunctExec,
    pending: PendingMask,
    out: &mut StageOutput,
) -> VortexResult<NodeState> {
    let end = pending.batch.coverage.end;
    if !pending.last {
        out.push_consumed(crate::node::InputPort::new(0)?);
    }
    out.set_batch(pending.batch, pending.last);
    node.push_cursor = end;
    node.push_output_credit = false;
    if pending.last {
        node.done = true;
        return Ok(NodeState::Done);
    }
    Ok(NodeState::NeedInput)
}

fn cascade_materializes_coverage(stage: &VecDeque<PendingMask>) -> bool {
    let selected_rows = stage
        .iter()
        .map(|pending| match &pending.batch.value {
            Value::Mask(mask) => mask.true_count(),
            Value::Array(_) => unreachable!("conjunct evaluation produces a mask"),
        })
        .sum::<usize>();
    let coverage_rows = stage
        .iter()
        .map(|pending| pending.batch.selection.len())
        .sum::<usize>();
    coverage_rows != 0 && (selected_rows as f64 / coverage_rows as f64) >= EXPR_EVAL_THRESHOLD
}

fn drain_parallel(node: &mut ConjunctExec, out: &mut StageOutput) -> VortexResult<NodeState> {
    if node.push_output_credit && node.push_heads.iter().all(|heads| !heads.is_empty()) {
        let end = node
            .push_heads
            .iter()
            .filter_map(|heads| heads.front().map(|head| head.batch.coverage.end))
            .min()
            .ok_or_else(|| vortex_err!("conjunct alignment has no heads"))?;
        let coverage = node.push_cursor..end;
        let mut selection: Option<Mask> = None;
        let mut materialized: Option<Mask> = None;
        let mut results = Vec::with_capacity(node.push_heads.len());
        let mut all_last = true;
        for (idx, heads) in node.push_heads.iter_mut().enumerate() {
            let mut head = heads.pop_front().vortex_expect("conjunct head exists");
            let consumed_all = head.batch.coverage.end == end;
            let prefix = if consumed_all {
                let PendingMask { batch, last } = head;
                all_last &= last;
                if !last {
                    out.push_consumed(crate::node::InputPort::new(idx)?);
                }
                batch
            } else {
                all_last = false;
                let prefix = head.batch.clone().slice(coverage.clone())?;
                let head_end = head.batch.coverage.end;
                head.batch = head.batch.slice(end..head_end)?;
                heads.push_front(head);
                prefix
            };
            let PushBatch {
                selection: prefix_selection,
                materialized: prefix_materialized,
                value,
                ..
            } = prefix;
            if let Some(expected) = selection.as_ref() {
                if expected != &prefix_selection {
                    return Err(vortex_err!(
                        "parallel conjunct slots have different selections over {coverage:?}"
                    ));
                }
            } else {
                selection = Some(prefix_selection);
            }
            if let Some(expected) = materialized.as_ref() {
                if expected != &prefix_materialized {
                    return Err(vortex_err!(
                        "parallel conjunct slots have different materialization domains over {coverage:?}"
                    ));
                }
            } else {
                materialized = Some(prefix_materialized);
            }
            let mask = value.into_mask()?;
            results.push(mask);
        }
        let selection = selection.vortex_expect("conjunct has at least one slot");
        let materialized = materialized.vortex_expect("conjunct has at least one slot");
        if results.is_empty() {
            return Err(vortex_err!("conjunct has no masks"));
        }
        let result = Mask::intersect_owned(results);
        node.push_cursor = end;
        let last = all_last && end == node.range.end;
        out.set_batch(
            PushBatch::from_validated_parts(coverage, selection, materialized, Value::Mask(result)),
            last,
        );
        node.push_output_credit = false;
        if last {
            node.done = true;
            return Ok(NodeState::Done);
        }
    }
    Ok(NodeState::NeedInput)
}

fn enqueue_all_false_closure(
    first_skipped: usize,
    slot_count: usize,
    stage: &VecDeque<PendingMask>,
    out: &mut StageOutput,
) {
    for skipped in first_skipped..slot_count {
        for pending in stage {
            let selection = Mask::new_false(pending.batch.selection.len());
            out.push_gate(
                ActivationTarget::PredicateSlot(skipped),
                pending.batch.coverage.clone(),
                ActivationRows::selected(selection),
            );
        }
    }
}

fn drain_cascade(node: &mut ConjunctExec, out: &mut StageOutput) -> VortexResult<NodeState> {
    if !node.push_output_credit {
        return Ok(NodeState::NeedInput);
    }
    let Some(pending) = node.cascade_ready.pop_front() else {
        return Ok(NodeState::NeedInput);
    };
    node.push_cursor = pending.batch.coverage.end;
    node.push_output_credit = false;
    out.set_batch(pending.batch, pending.last);
    if pending.last {
        node.done = true;
        return Ok(NodeState::Done);
    }
    Ok(NodeState::NeedInput)
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::expr::root;
    use vortex_error::VortexResult;
    use vortex_mask::Mask;

    use super::ConjunctExec;
    use super::ConjunctMode;
    use super::ConjunctSlot;
    use super::PendingMask;
    use super::cascade_materializes_coverage;
    use super::drain_single;
    use super::emit_single;
    use super::enqueue_all_false_closure;
    use super::refine_materialized_predicate;
    use crate::node::ActivationTarget;
    use crate::node::ExecNode;
    use crate::node::NodeState;
    use crate::node::PushBatch;
    use crate::node::StageOutput;
    use crate::node::StageSideband;
    use crate::node::Value;

    fn pending_mask(
        coverage: std::ops::Range<u64>,
        selected: impl IntoIterator<Item = bool>,
        last: bool,
    ) -> VortexResult<PendingMask> {
        let mask = Mask::from_iter(selected);
        Ok(PendingMask {
            batch: PushBatch::try_new(coverage, Mask::new_true(mask.len()), Value::Mask(mask))?,
            last,
        })
    }

    #[test]
    #[expect(clippy::cognitive_complexity)]
    fn single_slot_streams_fragments_and_retains_only_under_backpressure() -> VortexResult<()> {
        for mode in [ConjunctMode::Cascade, ConjunctMode::Parallel] {
            let predicate = root().bind(&DType::Bool(Nullability::NonNullable))?;
            let mut node = ConjunctExec::new(
                vec![ConjunctSlot {
                    input: 0,
                    predicate,
                }],
                mode,
            );
            node.reset(0..6);
            let mut output = StageOutput::default();

            let state = emit_single(
                &mut node,
                pending_mask(0..2, [true, false], false)?,
                &mut output,
            )?;
            assert!(matches!(state, NodeState::NeedInput));
            let (first, terminal) = output.take_batch().expect("first fragment is forwarded");
            assert!(!terminal);
            assert_eq!(first.coverage, 0..2);
            assert_eq!(first.value.into_mask()?, Mask::from_iter([true, false]));
            assert!(matches!(
                output.take_sideband(),
                Some(StageSideband::Consumed(port)) if port.index() == 0
            ));
            assert!(node.cascade_stage.is_empty());

            node.push_heads[0].push_back(pending_mask(2..4, [false, true], false)?);
            assert!(matches!(
                drain_single(&mut node, &mut output)?,
                NodeState::NeedInput
            ));
            assert!(output.take_batch().is_none());
            assert_eq!(node.push_heads[0].len(), 1);

            node.push_output_credit = true;
            assert!(matches!(
                drain_single(&mut node, &mut output)?,
                NodeState::NeedInput
            ));
            let (second, terminal) = output.take_batch().expect("retained fragment is drained");
            assert!(!terminal);
            assert_eq!(second.coverage, 2..4);
            assert_eq!(second.value.into_mask()?, Mask::from_iter([false, true]));
            assert!(matches!(
                output.take_sideband(),
                Some(StageSideband::Consumed(port)) if port.index() == 0
            ));

            node.push_output_credit = true;
            assert!(matches!(
                emit_single(
                    &mut node,
                    pending_mask(4..6, [true, true], true)?,
                    &mut output,
                )?,
                NodeState::Done
            ));
            let (last, terminal) = output.take_batch().expect("terminal fragment is forwarded");
            assert!(terminal);
            assert_eq!(last.coverage, 4..6);
            assert!(output.take_sideband().is_none());
            assert!(node.done);
        }
        Ok(())
    }

    #[test]
    fn all_false_stage_closes_every_skipped_slot_without_evaluation() -> VortexResult<()> {
        // Closure depends only on already-evaluated masks; skipped slot expressions are not
        // accepted by this helper and therefore cannot be executed on the short-circuit path.
        let mut stage = VecDeque::new();
        for coverage in [0..2, 2..5] {
            let len = usize::try_from(coverage.end - coverage.start)
                .map_err(|_| vortex_error::vortex_err!("test coverage exceeds usize"))?;
            stage.push_back(PendingMask {
                batch: PushBatch::try_new(
                    coverage,
                    Mask::new_true(len),
                    Value::Mask(Mask::new_false(len)),
                )?,
                last: false,
            });
        }

        let mut output = StageOutput::default();
        enqueue_all_false_closure(1, 4, &stage, &mut output);
        let mut activations = Vec::new();
        while let Some(sideband) = output.take_sideband() {
            if let StageSideband::Gate {
                target: ActivationTarget::PredicateSlot(slot),
                coverage,
                rows,
            } = sideband
            {
                activations.push((slot, coverage, rows.logical().clone()));
            }
        }
        assert_eq!(activations.len(), 6);
        for skipped in 1..4 {
            let spans = activations
                .iter()
                .filter(|(slot, ..)| *slot == skipped)
                .map(|(_, coverage, selection)| (coverage.clone(), selection.all_false()))
                .collect::<Vec<_>>();
            assert_eq!(spans, [(0..2, true), (2..5, true)]);
        }
        Ok(())
    }

    #[test]
    fn cascade_materialization_threshold_uses_complete_fragmented_stage() -> VortexResult<()> {
        for (selected, expected) in [
            (0, false),
            (199, false),
            (200, true),
            (201, true),
            (1000, true),
        ] {
            let mut stage = VecDeque::new();
            for (start, end) in [(0, 113), (113, 509), (509, 1000)] {
                let mask = Mask::from_iter((start..end).map(|row| row < selected));
                stage.push_back(PendingMask {
                    batch: PushBatch::try_new(
                        u64::try_from(start)?..u64::try_from(end)?,
                        Mask::new_true(end - start),
                        Value::Mask(mask),
                    )?,
                    last: end == 1000,
                });
            }
            assert_eq!(
                cascade_materializes_coverage(&stage),
                expected,
                "selected={selected}"
            );
        }
        Ok(())
    }

    #[test]
    fn predicate_refinement_matches_threshold_regimes_across_fragments() {
        for selected_rows in [0, 199, 200, 201, 1000] {
            let selected = Mask::from_iter((0..1000).map(|row| row < selected_rows));
            let materialized = if selected_rows < 200 {
                selected.clone()
            } else {
                Mask::new_true(1000)
            };
            for (start, end) in [(0, 113), (113, 509), (509, 1000)] {
                let selected = selected.slice(start..end);
                let materialized = materialized.slice(start..end);
                let dense_predicate =
                    Mask::from_iter(materialized.iter().zip(start..end).filter_map(
                        |(materialized, row)| materialized.then_some(row.is_multiple_of(3)),
                    ));
                let actual =
                    refine_materialized_predicate(&selected, &materialized, dense_predicate);
                let expected = Mask::from_iter(
                    (start..end).map(|row| row < selected_rows && row.is_multiple_of(3)),
                );
                assert_eq!(
                    actual, expected,
                    "selected={selected_rows}, span={start}..{end}"
                );
            }
        }
    }

    #[test]
    fn predicate_refinement_handles_mixed_materialization_domain() {
        let selected = Mask::from_iter([true, false, false, true, false, false, true, false]);
        let materialized = Mask::from_iter([true, true, false, true, false, true, true, false]);
        // Predicate values correspond to materialized rows 0, 1, 3, 5, and 6.
        let predicate = Mask::from_iter([false, true, true, true, false]);

        assert_eq!(
            refine_materialized_predicate(&selected, &materialized, predicate),
            Mask::from_iter([false, false, false, true, false, false, false, false])
        );
    }
}
