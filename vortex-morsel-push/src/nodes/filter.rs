// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::VecDeque;
use std::ops::Range;

use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::dtype::DType;
use vortex_array::expr::BoundExpression;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::Mask;

use crate::io::IoPriority;
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
use crate::nodes::PushBatching;

/// The root of a morsel: refine the demand with the filter, then project under it.
pub struct FilterExec {
    predicate: Option<NodeId>,
    projection: NodeId,
    projection_expr: BoundExpression,
    output_dtype: DType,
    push_batching: PushBatching,

    // Per-morsel state.
    range: Range<u64>,
    plan_stage: u8,
    plan_started: bool,
    mask: Option<Mask>,
    done: bool,
    children: Vec<NodeId>,
    push_cursor: u64,
    /// Authoritative mask fragments awaiting projection. Unlike general input batches, these are
    /// consumed into compact coverage-domain state before predicate credit is returned.
    push_predicate: VecDeque<PredicateFragment>,
    /// Full-range predicate mask retained after a morsel-batched predicate reaches terminal.
    /// The same mask is sent through the projection Gate and later validates the terminal batch.
    push_predicate_received: u64,
    push_predicate_rows: u64,
    push_projection: VecDeque<PendingInput>,
    push_predicate_ended: bool,
    push_projection_ended: bool,
    push_projection_saw_batch: bool,
    push_output_credit: bool,
}

struct PendingInput {
    batch: PushBatch,
    last: bool,
}

struct PredicateFragment {
    coverage: Range<u64>,
    mask: Mask,
}

impl FilterExec {
    /// Build a filter node.
    pub fn new(
        predicate: Option<NodeId>,
        projection: NodeId,
        projection_expr: BoundExpression,
        output_dtype: DType,
    ) -> Self {
        Self::new_with_push_batching(
            predicate,
            projection,
            projection_expr,
            output_dtype,
            PushBatching::Streaming,
        )
    }

    pub(crate) fn new_with_push_batching(
        predicate: Option<NodeId>,
        projection: NodeId,
        projection_expr: BoundExpression,
        output_dtype: DType,
        push_batching: PushBatching,
    ) -> Self {
        let children = predicate.into_iter().chain([projection]).collect();
        Self {
            predicate,
            projection,
            projection_expr,
            output_dtype,
            push_batching,
            range: 0..0,
            plan_stage: 0,
            plan_started: false,
            mask: None,
            done: false,
            children,
            push_cursor: 0,
            push_predicate: VecDeque::new(),
            push_predicate_received: 0,
            push_predicate_rows: 0,
            push_projection: VecDeque::new(),
            push_predicate_ended: false,
            push_projection_ended: false,
            push_projection_saw_batch: false,
            push_output_credit: true,
        }
    }
}

impl ExecNode for FilterExec {
    fn push_profile_kind(&self) -> crate::node::PushProfileKind {
        crate::node::PushProfileKind::Filter
    }

    fn reset(&mut self, range: Range<u64>) {
        self.range = range;
        self.plan_stage = 0;
        self.plan_started = false;
        self.mask = None;
        self.done = false;
        self.push_cursor = self.range.start;
        self.push_predicate.clear();
        self.push_predicate_received = self.range.start;
        self.push_predicate_rows = 0;
        self.push_projection.clear();
        self.push_predicate_ended = self.predicate.is_none();
        self.push_projection_ended = false;
        self.push_projection_saw_batch = false;
        self.push_output_credit = true;
    }

    fn next_plan(&mut self, cx: &mut PlanCx<'_>) -> VortexResult<PlanPoll> {
        loop {
            let child = match (self.plan_stage, self.predicate) {
                (0, Some(predicate)) => predicate,
                (0, None) | (1, _) => self.projection,
                _ => return Ok(PlanPoll::Complete),
            };
            if cx.out_of_budget() {
                return Ok(PlanPoll::Item(PlanItem::Plan));
            }
            let fresh = !self.plan_started;
            self.plan_started = true;
            let priority = if self.predicate.is_some() && self.plan_stage > 0 {
                IoPriority::Speculative
            } else {
                IoPriority::Required
            };
            if cx.plan_child_with_priority(child, self.range.clone(), fresh, priority)? {
                self.plan_stage += if self.plan_stage == 0 && self.predicate.is_none() {
                    2
                } else {
                    1
                };
                self.plan_started = false;
            } else {
                return Ok(PlanPoll::Item(PlanItem::Plan));
            }
        }
    }

    fn execute(&mut self, cx: &mut ExecCx<'_>) -> VortexResult<ExecPoll> {
        if self.done {
            return Ok(ExecPoll::Done);
        }
        if self.mask.is_none() {
            let demand = cx.demand().clone();
            let mask = match self.predicate {
                Some(predicate) => match cx.child_mask(predicate, demand)? {
                    ChildPoll::Value(mask) => mask,
                    ChildPoll::Blocked(waits) => return Ok(ExecPoll::Blocked(waits)),
                    ChildPoll::Done => {
                        return Err(vortex_err!("filter predicate produced no value"));
                    }
                },
                None => demand,
            };

            if mask.all_false() {
                self.done = true;
                cx.stats().morsels_empty += 1;
                return Ok(ExecPoll::Value(ValueBatch {
                    coverage: self.range.clone(),
                    value: Value::Array(Canonical::empty(&self.output_dtype).into_array()),
                }));
            }
            self.mask = Some(mask);
        }

        // The projection subtree executes only for surviving rows. A sealed-empty chunk avoids
        // cloning and decoding its projection tickets, although planning may have prefetched them.
        let mask = self
            .mask
            .as_ref()
            .vortex_expect("non-empty predicate mask is retained")
            .clone();
        let array = match cx.child_array(self.projection, mask)? {
            ChildPoll::Value(array) => array,
            ChildPoll::Blocked(waits) => return Ok(ExecPoll::Blocked(waits)),
            ChildPoll::Done => return Err(vortex_err!("filter projection produced no value")),
        };
        let array = array.apply_bound(&self.projection_expr)?;
        self.mask = None;
        self.done = true;

        Ok(ExecPoll::Value(ValueBatch {
            coverage: self.range.clone(),
            value: Value::Array(array),
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
        accept_filter_input(self, port, batch, last_for_input, out)?;
        drain_filter(self, cx, out)
    }

    #[inline]
    fn push_end(
        &mut self,
        port: crate::node::InputPort,
        cx: &mut PushCx<'_>,
        out: &mut StageOutput,
    ) -> VortexResult<NodeState> {
        accept_filter_end(self, port)?;
        drain_filter(self, cx, out)
    }

    #[inline]
    fn push_resume(
        &mut self,
        cx: &mut PushCx<'_>,
        out: &mut StageOutput,
    ) -> VortexResult<NodeState> {
        drain_filter(self, cx, out)
    }

    #[inline]
    fn push_credit(
        &mut self,
        cx: &mut PushCx<'_>,
        out: &mut StageOutput,
    ) -> VortexResult<NodeState> {
        if self.push_output_credit {
            return Err(vortex_err!("filter received duplicate output credit"));
        }
        self.push_output_credit = true;
        drain_filter(self, cx, out)
    }

    fn retire(&mut self, cx: &mut RetireCx<'_>) {
        self.mask = None;
        for &child in &self.children {
            cx.retire_child(child);
        }
    }

    fn children(&self) -> &[NodeId] {
        &self.children
    }
}

fn accept_filter_input(
    node: &mut FilterExec,
    port: crate::node::InputPort,
    batch: PushBatch,
    last_for_input: bool,
    out: &mut StageOutput,
) -> VortexResult<()> {
    match port.index() {
        0 if node.predicate.is_some() => {
            let coverage = batch.coverage.clone();
            let selection = batch.value.into_mask()?;
            push_predicate_fragment(
                &mut node.push_predicate,
                &mut node.push_predicate_ended,
                &mut node.push_predicate_received,
                &mut node.push_predicate_rows,
                coverage.clone(),
                selection.clone(),
                last_for_input,
                node.range.start,
                node.range.end,
            )?;
            out.push_gate(
                ActivationTarget::Projection,
                coverage,
                ActivationRows::selected(selection),
            );
            if !last_for_input {
                out.push_consumed(port);
            }
            Ok(())
        }
        1 => {
            push_input(
                &mut node.push_projection,
                &mut node.push_projection_ended,
                batch,
                last_for_input,
                node.push_cursor,
                node.range.end,
                "projection",
            )?;
            node.push_projection_saw_batch = true;
            Ok(())
        }
        port => Err(vortex_err!("filter received unknown input port {port}")),
    }
}

fn accept_filter_end(node: &mut FilterExec, port: crate::node::InputPort) -> VortexResult<()> {
    match port.index() {
        0 if node.predicate.is_some() => {
            if node.push_predicate_ended {
                return Err(vortex_err!("filter predicate ended more than once"));
            }
            if node.push_predicate_received != node.range.start {
                return Err(vortex_err!(
                    "filter predicate used End after emitting a batch"
                ));
            }
            if node.push_predicate_received != node.range.end {
                return Err(vortex_err!(
                    "filter predicate ended without accounting for {}..{}",
                    node.push_predicate_received,
                    node.range.end
                ));
            }
            node.push_predicate_ended = true;
            Ok(())
        }
        1 => {
            if node.push_projection_saw_batch {
                return Err(vortex_err!(
                    "filter projection used End after emitting a batch"
                ));
            }
            end_input(
                &node.push_projection,
                &mut node.push_projection_ended,
                node.push_cursor,
                node.range.end,
                "projection",
            )
        }
        port => Err(vortex_err!("filter received unknown input port {port}")),
    }
}

fn drain_filter(
    node: &mut FilterExec,
    cx: &mut PushCx<'_>,
    out: &mut StageOutput,
) -> VortexResult<NodeState> {
    if node.done {
        return Ok(NodeState::Done);
    }
    if node.push_batching == PushBatching::Morsel {
        return drain_filter_morsel(node, cx, out);
    }
    if node.predicate.is_none() {
        if node.push_output_credit
            && let Some(input) = node.push_projection.pop_front()
        {
            let batch = apply_projection(input.batch, &node.projection_expr, cx)?;
            node.push_cursor = batch.coverage.end;
            if !input.last {
                out.push_consumed(crate::node::InputPort::new(1)?);
            }
            out.set_batch(batch, input.last);
            node.push_output_credit = false;
            if input.last {
                node.done = true;
                return Ok(NodeState::Done);
            }
        }
        return Ok(NodeState::NeedInput);
    }
    if node.push_output_credit
        && !node.push_predicate.is_empty()
        && !node.push_projection.is_empty()
    {
        let end = node
            .push_predicate
            .front()
            .vortex_expect("predicate head exists")
            .coverage
            .end
            .min(
                node.push_projection
                    .front()
                    .vortex_expect("projection head exists")
                    .batch
                    .coverage
                    .end,
            );
        let coverage = node.push_cursor..end;
        let mut predicate = node
            .push_predicate
            .pop_front()
            .vortex_expect("predicate head exists");
        let projection = node
            .push_projection
            .pop_front()
            .vortex_expect("projection head exists");
        if predicate.coverage.start != node.push_cursor
            || projection.batch.coverage.start != node.push_cursor
        {
            return Err(vortex_err!(
                "filter inputs are not aligned at coverage cursor {}",
                node.push_cursor
            ));
        }
        let predicate_all = predicate.coverage.end == end;
        let projection_all = projection.batch.coverage.end == end;
        let projection_last = projection.last;
        let mut projection_batch = Some(projection.batch);
        let projection_prefix = if projection_all {
            projection_batch
                .take()
                .vortex_expect("projection batch exists")
        } else {
            projection_batch
                .as_ref()
                .vortex_expect("projection batch exists")
                .clone()
                .slice(coverage.clone())?
        };
        let prefix_len = usize::try_from(end - predicate.coverage.start)
            .map_err(|_| vortex_err!("predicate prefix length exceeds usize"))?;
        let mask = predicate.mask.slice(0..prefix_len);
        projection_selection_matches_predicate(
            &projection_prefix.selection,
            &projection_prefix.materialized,
            &mask,
        )?;
        let array = projection_prefix.value.into_array()?;
        let consumed_rows = end - coverage.start;
        let projected =
            PushBatch::from_validated_parts(coverage, mask.clone(), mask, Value::Array(array));
        let projected = apply_projection(projected, &node.projection_expr, cx)?;
        let last =
            projection_all && projection_last && node.push_predicate_ended && end == node.range.end;
        node.push_cursor = end;
        node.push_predicate_rows = node
            .push_predicate_rows
            .checked_sub(consumed_rows)
            .ok_or_else(|| vortex_err!("predicate retained-row accounting underflow"))?;
        if projection_all && !projection_last {
            out.push_consumed(crate::node::InputPort::new(1)?);
        }
        out.set_batch(projected, last);
        node.push_output_credit = false;
        if !predicate_all {
            predicate.coverage.start = end;
            predicate.mask = predicate.mask.slice(prefix_len..);
            node.push_predicate.push_front(predicate);
        }
        if !projection_all {
            let batch = projection_batch
                .take()
                .vortex_expect("projection batch exists");
            let head_end = batch.coverage.end;
            node.push_projection.push_front(PendingInput {
                batch: batch.slice(end..head_end)?,
                last: projection_last,
            });
        }
        if last {
            node.done = true;
            return Ok(NodeState::Done);
        }
    }
    Ok(NodeState::NeedInput)
}

fn drain_filter_morsel(
    node: &mut FilterExec,
    cx: &mut PushCx<'_>,
    out: &mut StageOutput,
) -> VortexResult<NodeState> {
    if !node.push_output_credit || !node.push_predicate_ended || !node.push_projection_ended {
        return Ok(NodeState::NeedInput);
    }
    let Some(projection) = node.push_projection.pop_front() else {
        return Ok(NodeState::NeedInput);
    };
    if !projection.last || projection.batch.coverage != node.range {
        return Err(vortex_err!(
            "morsel-batched filter projection must be one full-range terminal batch"
        ));
    }
    let predicate = if node.push_predicate.is_empty() {
        Mask::new_true(
            usize::try_from(node.range.end - node.range.start)
                .map_err(|_| vortex_err!("morsel-batched filter coverage length exceeds usize"))?,
        )
    } else {
        Mask::concat(node.push_predicate.iter().map(|fragment| &fragment.mask))?
    };
    if predicate.len()
        != usize::try_from(node.range.end - node.range.start)
            .map_err(|_| vortex_err!("morsel-batched filter coverage length exceeds usize"))?
    {
        return Err(vortex_err!(
            "morsel-batched predicate fragments do not cover the full range"
        ));
    }
    projection_selection_matches_predicate(
        &projection.batch.selection,
        &projection.batch.materialized,
        &predicate,
    )?;
    let projected = apply_projection(projection.batch, &node.projection_expr, cx)?;
    node.push_predicate.clear();
    node.push_predicate_rows = 0;
    node.push_cursor = node.range.end;
    out.set_batch(projected, true);
    node.push_output_credit = false;
    node.done = true;
    Ok(NodeState::Done)
}

#[inline]
fn projection_selection_matches_predicate(
    projection: &Mask,
    materialized: &Mask,
    predicate: &Mask,
) -> VortexResult<()> {
    #[cfg(debug_assertions)]
    if projection != predicate || materialized != predicate {
        return Err(vortex_err!(
            "filter projection selection does not match its authoritative predicate mask"
        ));
    }
    #[cfg(not(debug_assertions))]
    let _ = (projection, materialized, predicate);
    Ok(())
}

fn apply_projection(
    batch: PushBatch,
    expression: &BoundExpression,
    _cx: &mut PushCx<'_>,
) -> VortexResult<PushBatch> {
    let PushBatch {
        coverage,
        selection,
        materialized,
        value,
    } = batch;
    let array = value.into_array()?.apply_bound(expression)?;
    Ok(PushBatch::from_validated_parts(
        coverage,
        selection,
        materialized,
        Value::Array(array),
    ))
}

#[allow(clippy::too_many_arguments)]
fn push_predicate_fragment(
    queue: &mut VecDeque<PredicateFragment>,
    ended: &mut bool,
    received: &mut u64,
    retained_rows: &mut u64,
    coverage: Range<u64>,
    mask: Mask,
    last: bool,
    range_start: u64,
    range_end: u64,
) -> VortexResult<()> {
    if *ended {
        return Err(vortex_err!("filter predicate produced data after ending"));
    }
    if coverage.start != *received || coverage.end > range_end {
        return Err(vortex_err!(
            "filter predicate produced non-contiguous coverage {coverage:?}; expected {received}"
        ));
    }
    if last && coverage.end != range_end {
        return Err(vortex_err!(
            "filter predicate ended at {}, expected {range_end}",
            coverage.end
        ));
    }
    if !last && coverage.end == range_end {
        return Err(vortex_err!(
            "filter predicate covered its full range without a final marker"
        ));
    }
    let rows = coverage.end - coverage.start;
    let next_retained = retained_rows
        .checked_add(rows)
        .ok_or_else(|| vortex_err!("predicate retained-row accounting overflow"))?;
    if next_retained > range_end - range_start {
        return Err(vortex_err!(
            "filter predicate retained {next_retained} rows beyond morsel capacity {}",
            range_end - range_start
        ));
    }
    *received = coverage.end;
    *retained_rows = next_retained;
    *ended = last;
    queue.push_back(PredicateFragment { coverage, mask });
    Ok(())
}

fn push_input(
    queue: &mut VecDeque<PendingInput>,
    ended: &mut bool,
    batch: PushBatch,
    last: bool,
    cursor: u64,
    expected_end: u64,
    name: &str,
) -> VortexResult<()> {
    if *ended {
        return Err(vortex_err!("filter {name} produced data after ending"));
    }
    if !queue.is_empty() {
        return Err(vortex_err!(
            "filter {name} exceeded its one-batch edge credit"
        ));
    }
    let expected_start = queue
        .back()
        .map_or(cursor, |input| input.batch.coverage.end);
    if batch.coverage.start != expected_start || batch.coverage.end > expected_end {
        return Err(vortex_err!(
            "filter {name} produced non-contiguous coverage {:?}; expected {expected_start}",
            batch.coverage
        ));
    }
    if last && batch.coverage.end != expected_end {
        return Err(vortex_err!(
            "filter {name} ended at {}, expected {expected_end}",
            batch.coverage.end
        ));
    }
    if !last && batch.coverage.end == expected_end {
        return Err(vortex_err!(
            "filter {name} covered its full range without a final marker"
        ));
    }
    *ended = last;
    queue.push_back(PendingInput { batch, last });
    Ok(())
}

fn end_input(
    queue: &VecDeque<PendingInput>,
    ended: &mut bool,
    cursor: u64,
    expected_end: u64,
    name: &str,
) -> VortexResult<()> {
    let accounted = queue
        .back()
        .map_or(cursor, |input| input.batch.coverage.end);
    if accounted != expected_end {
        return Err(vortex_err!(
            "filter {name} ended without accounting for {accounted}..{expected_end}"
        ));
    }
    *ended = true;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::Arc;

    use futures::FutureExt;
    use vortex_array::Canonical;
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::expr::root;
    use vortex_buffer::BitBuffer;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;
    use vortex_layout::segments::SegmentFuture;
    use vortex_layout::segments::SegmentId;
    use vortex_layout::segments::SegmentSource;
    use vortex_mask::Mask;
    use vortex_session::VortexSession;

    use super::FilterExec;
    use super::projection_selection_matches_predicate;
    use super::push_input;
    use super::push_predicate_fragment;
    use crate::cells::SharedCells;
    use crate::io::IoPlane;
    use crate::io::IoService;
    use crate::node::ExecNode;
    use crate::node::InputPort;
    use crate::node::NodeState;
    use crate::node::PushBatch;
    use crate::node::PushCx;
    use crate::node::StageOutput;
    use crate::node::StageSideband;
    use crate::node::Value;
    use crate::nodes::PushBatching;
    use crate::stats::ScanStats;

    struct NoIo;

    impl SegmentSource for NoIo {
        fn request(&self, _id: SegmentId) -> SegmentFuture {
            async { Err(vortex_err!("unexpected test IO")) }.boxed()
        }
    }

    #[test]
    fn projection_selection_requires_equal_aligned_and_misaligned_masks() {
        let aligned = Mask::from_iter([true, false, true, true, false, false, true]);
        assert!(projection_selection_matches_predicate(&aligned, &aligned, &aligned).is_ok());

        let predicate = Mask::from_buffer(
            BitBuffer::from_iter([false, true, false, true, true, false, false, true, false])
                .slice(1..8),
        );
        let projection = Mask::from_buffer(
            BitBuffer::from_iter([
                false, false, true, false, true, true, false, false, true, false,
            ])
            .slice(2..9),
        );
        assert!(
            projection_selection_matches_predicate(&projection, &projection, &predicate).is_ok()
        );

        let mismatch = Mask::from_iter([true, false, true, false, false, false, true]);
        assert!(projection_selection_matches_predicate(&mismatch, &mismatch, &predicate).is_err());
        assert!(
            projection_selection_matches_predicate(
                &predicate,
                &Mask::new_true(predicate.len()),
                &predicate,
            )
            .is_err()
        );
    }

    #[test]
    fn rejects_non_final_full_coverage_inputs() -> VortexResult<()> {
        let mut predicates = VecDeque::new();
        let mut predicate_ended = false;
        let mut received = 0;
        let mut retained = 0;
        assert!(
            push_predicate_fragment(
                &mut predicates,
                &mut predicate_ended,
                &mut received,
                &mut retained,
                0..4,
                Mask::new_false(4),
                false,
                0,
                4,
            )
            .is_err()
        );

        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let batch = PushBatch::try_new(
            0..4,
            Mask::new_false(4),
            Value::Array(Canonical::empty(&dtype).into_array()),
        )?;
        let mut projection = VecDeque::new();
        let mut projection_ended = false;
        assert!(
            push_input(
                &mut projection,
                &mut projection_ended,
                batch,
                false,
                0,
                4,
                "projection",
            )
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn accepts_one_explicit_projection_end() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let expression = root().bind(&dtype)?;
        let mut filter = FilterExec::new(None, 0, expression, dtype);
        filter.reset(0..0);

        let io = IoPlane::new(IoService::new(Arc::new(NoIo)));
        let cells = SharedCells::disabled();
        let session = VortexSession::empty();
        let mut stats = ScanStats::default();
        let mut cx = PushCx::new(&io, &cells, &session, &mut stats);
        let mut output = StageOutput::default();

        let state = filter.push_end(InputPort::new(1)?, &mut cx, &mut output)?;

        assert!(matches!(state, NodeState::NeedInput));
        assert!(output.is_empty());
        Ok(())
    }

    #[test]
    fn morsel_filter_keeps_fragment_gates_and_emits_one_sparse_batch() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let expression = root().bind(&dtype)?;
        let mut filter =
            FilterExec::new_with_push_batching(Some(0), 1, expression, dtype, PushBatching::Morsel);
        filter.reset(0..4);

        let io = IoPlane::new(IoService::new(Arc::new(NoIo)));
        let cells = SharedCells::disabled();
        let session = VortexSession::empty();
        let mut stats = ScanStats::default();
        let mut cx = PushCx::new(&io, &cells, &session, &mut stats);
        let mut output = StageOutput::default();

        for (coverage, predicate, last) in [
            (0..2, Mask::from_iter([true, false]), false),
            (2..4, Mask::from_iter([false, true]), true),
        ] {
            let len = predicate.len();
            let batch = PushBatch::try_new(
                coverage.clone(),
                Mask::new_true(len),
                Value::Mask(predicate.clone()),
            )?;
            let _state =
                filter.push_input(InputPort::new(0)?, batch, last, &mut cx, &mut output)?;
            assert!(output.take_batch().is_none());
            assert!(matches!(
                output.take_sideband(),
                Some(StageSideband::Gate {
                    target: crate::node::ActivationTarget::Projection,
                    coverage: gate_coverage,
                    rows,
                }) if gate_coverage == coverage && rows.logical() == &predicate
            ));
            if !last {
                assert!(matches!(
                    output.take_sideband(),
                    Some(StageSideband::Consumed(port)) if port.index() == 0
                ));
            }
            assert!(output.take_sideband().is_none());
        }

        let selection = Mask::from_iter([true, false, false, true]);
        let projection = PushBatch::try_new(
            0..4,
            selection.clone(),
            Value::Array(PrimitiveArray::from_iter([10i32, 40]).into_array()),
        )?;
        let state =
            filter.push_input(InputPort::new(1)?, projection, true, &mut cx, &mut output)?;
        assert!(matches!(state, NodeState::Done));
        let (batch, last) = output
            .take_batch()
            .ok_or_else(|| vortex_err!("morsel filter emitted no batch"))?;
        assert!(last);
        assert_eq!(batch.coverage, 0..4);
        assert_eq!(batch.selection, selection);
        assert_eq!(batch.selection, batch.materialized);
        assert_eq!(batch.value.into_array()?.len(), 2);

        filter.reset(4..6);
        assert!(filter.push_predicate.is_empty());
        assert!(filter.push_projection.is_empty());
        assert_eq!(filter.push_predicate_rows, 0);
        Ok(())
    }

    #[test]
    fn morsel_filter_all_false_is_one_typed_empty_batch() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let expression = root().bind(&dtype)?;
        let mut filter = FilterExec::new_with_push_batching(
            Some(0),
            1,
            expression,
            dtype.clone(),
            PushBatching::Morsel,
        );
        filter.reset(0..3);

        let io = IoPlane::new(IoService::new(Arc::new(NoIo)));
        let cells = SharedCells::disabled();
        let session = VortexSession::empty();
        let mut stats = ScanStats::default();
        let mut cx = PushCx::new(&io, &cells, &session, &mut stats);
        let mut output = StageOutput::default();
        let predicate =
            PushBatch::try_new(0..3, Mask::new_true(3), Value::Mask(Mask::new_false(3)))?;
        let _state =
            filter.push_input(InputPort::new(0)?, predicate, true, &mut cx, &mut output)?;
        assert!(matches!(
            output.take_sideband(),
            Some(StageSideband::Gate {
                target: crate::node::ActivationTarget::Projection,
                coverage,
                rows,
            }) if coverage == (0..3) && rows.logical().all_false()
        ));
        assert!(output.take_sideband().is_none());

        let projection = PushBatch::try_new(
            0..3,
            Mask::new_false(3),
            Value::Array(Canonical::empty(&dtype).into_array()),
        )?;
        let state =
            filter.push_input(InputPort::new(1)?, projection, true, &mut cx, &mut output)?;
        assert!(matches!(state, NodeState::Done));
        let (batch, last) = output
            .take_batch()
            .ok_or_else(|| vortex_err!("all-false morsel filter emitted no batch"))?;
        assert!(last);
        assert!(batch.selection.all_false());
        let array = batch.value.into_array()?;
        assert!(array.is_empty());
        assert_eq!(array.dtype(), &dtype);
        Ok(())
    }

    #[test]
    fn streaming_filter_keeps_one_gate_per_predicate_fragment() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let expression = root().bind(&dtype)?;
        let mut filter = FilterExec::new_with_push_batching(
            Some(0),
            1,
            expression,
            dtype,
            PushBatching::Streaming,
        );
        filter.reset(10..14);

        let io = IoPlane::new(IoService::new(Arc::new(NoIo)));
        let cells = SharedCells::disabled();
        let session = VortexSession::empty();
        let mut stats = ScanStats::default();
        let mut cx = PushCx::new(&io, &cells, &session, &mut stats);
        let mut output = StageOutput::default();

        for (coverage, mask, last) in [
            (10..12, Mask::from_iter([true, false]), false),
            (12..14, Mask::from_iter([false, true]), true),
        ] {
            let batch = PushBatch::try_new(
                coverage.clone(),
                Mask::new_true(mask.len()),
                Value::Mask(mask.clone()),
            )?;
            let _state =
                filter.push_input(InputPort::new(0)?, batch, last, &mut cx, &mut output)?;
            assert!(matches!(
                output.take_sideband(),
                Some(StageSideband::Gate {
                    target: crate::node::ActivationTarget::Projection,
                    coverage: gate_coverage,
                    rows,
                }) if gate_coverage == coverage && rows.logical() == &mask
            ));
            if last {
                assert!(output.take_sideband().is_none());
            } else {
                assert!(matches!(
                    output.take_sideband(),
                    Some(StageSideband::Consumed(port)) if port.index() == 0
                ));
            }
        }
        Ok(())
    }
}
