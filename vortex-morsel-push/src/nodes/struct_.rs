// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::VecDeque;
use std::ops::Range;
use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::FieldNames;
use vortex_array::validity::Validity;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::Mask;

use crate::node::ActivationRows;
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

/// Struct is almost nothing: identity edges to each field, then a zip.
///
/// Every field is planned and executed under the *same* demand — the identity map means sharing
/// the demand handle rather than transforming it.
pub struct StructExec {
    names: FieldNames,
    children: Arc<[NodeId]>,
    push_batching: PushBatching,
    push_passthrough: bool,

    // Per-morsel state.
    range: Range<u64>,
    plan_cursor: usize,
    plan_started: bool,
    exec_cursor: usize,
    fields: Vec<ArrayRef>,
    done: bool,
    push_cursor: u64,
    push_heads: Vec<VecDeque<PendingField>>,
    push_ended: Vec<bool>,
    push_saw_batch: Vec<bool>,
    push_output_credit: bool,
    push_field_cursors: Vec<u64>,
    push_field_parts: Vec<Vec<ArrayRef>>,
    push_field_mask_parts: Vec<Vec<Mask>>,
}

struct PendingField {
    batch: PushBatch,
    last: bool,
}

impl StructExec {
    /// Build a struct node over one child per projected field.
    pub fn new(names: FieldNames, children: Arc<[NodeId]>) -> Self {
        Self::new_with_push_batching(names, children, PushBatching::Streaming, false)
    }

    pub(crate) fn new_with_push_batching(
        names: FieldNames,
        children: Arc<[NodeId]>,
        push_batching: PushBatching,
        push_passthrough: bool,
    ) -> Self {
        debug_assert_eq!(names.len(), children.len());
        debug_assert!(!push_passthrough || children.len() == 1);
        Self {
            names,
            children,
            push_batching,
            push_passthrough,
            range: 0..0,
            plan_cursor: 0,
            plan_started: false,
            exec_cursor: 0,
            fields: Vec::new(),
            done: false,
            push_cursor: 0,
            push_heads: Vec::new(),
            push_ended: Vec::new(),
            push_saw_batch: Vec::new(),
            push_output_credit: true,
            push_field_cursors: Vec::new(),
            push_field_parts: Vec::new(),
            push_field_mask_parts: Vec::new(),
        }
    }
}

impl ExecNode for StructExec {
    fn push_profile_kind(&self) -> crate::node::PushProfileKind {
        crate::node::PushProfileKind::Struct
    }

    fn reset(&mut self, range: Range<u64>) {
        self.range = range;
        self.plan_cursor = 0;
        self.plan_started = false;
        self.exec_cursor = 0;
        self.fields.clear();
        self.done = false;
        self.push_cursor = self.range.start;
        let width = self.children.len();
        self.push_heads.truncate(width);
        self.push_heads.resize_with(width, VecDeque::new);
        self.push_heads.iter_mut().for_each(VecDeque::clear);
        self.push_ended.resize(width, false);
        self.push_ended.fill(false);
        self.push_saw_batch.resize(width, false);
        self.push_saw_batch.fill(false);
        self.push_output_credit = true;
        self.push_field_cursors.resize(width, self.range.start);
        self.push_field_cursors.fill(self.range.start);
        self.push_field_parts.truncate(width);
        self.push_field_parts.resize_with(width, Vec::new);
        self.push_field_parts.iter_mut().for_each(Vec::clear);
        self.push_field_mask_parts.truncate(width);
        self.push_field_mask_parts.resize_with(width, Vec::new);
        self.push_field_mask_parts.iter_mut().for_each(Vec::clear);
    }

    fn next_plan(&mut self, cx: &mut PlanCx<'_>) -> VortexResult<PlanPoll> {
        while self.plan_cursor < self.children.len() {
            if cx.out_of_budget() {
                return Ok(PlanPoll::Item(PlanItem::Plan));
            }
            let fresh = !self.plan_started;
            self.plan_started = true;
            if cx.plan_child(self.children[self.plan_cursor], self.range.clone(), fresh)? {
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

        let demand = cx.demand().clone();
        let len = demand.true_count();
        if self.fields.capacity() < self.children.len() {
            self.fields
                .reserve(self.children.len().saturating_sub(self.fields.len()));
        }
        while self.exec_cursor < self.children.len() {
            let child = self.children[self.exec_cursor];
            match cx.child_array(child, demand.clone())? {
                ChildPoll::Value(array) => {
                    self.fields.push(array);
                    self.exec_cursor += 1;
                }
                ChildPoll::Blocked(waits) => return Ok(ExecPoll::Blocked(waits)),
                ChildPoll::Done => {
                    return Err(vortex_err!("struct child {child} produced no value"));
                }
            }
        }

        let fields = std::mem::take(&mut self.fields);
        let array = StructArray::try_new(self.names.clone(), fields, len, Validity::NonNullable)?
            .into_array();
        self.done = true;

        Ok(ExecPoll::Value(ValueBatch {
            coverage: self.range.clone(),
            value: Value::Array(array),
        }))
    }

    #[inline]
    fn push_start(
        &mut self,
        span: Range<u64>,
        rows: ActivationRows,
        _cx: &mut PushCx<'_>,
        out: &mut StageOutput,
    ) -> VortexResult<NodeState> {
        if !self.children.is_empty() {
            return Err(vortex_err!(
                "non-empty struct node received a source activation"
            ));
        }
        if span != self.range {
            return Err(vortex_err!(
                "empty struct activation {span:?} does not match its range {:?}",
                self.range
            ));
        }
        let array = StructArray::try_new(
            self.names.clone(),
            Vec::new(),
            rows.materialized().true_count(),
            Validity::NonNullable,
        )?
        .into_array();
        out.set_batch(
            PushBatch::try_new_materialized(span, rows, Value::Array(array))?,
            true,
        );
        self.done = true;
        Ok(NodeState::Done)
    }

    #[inline]
    fn push_input(
        &mut self,
        port: crate::node::InputPort,
        batch: PushBatch,
        last_for_input: bool,
        _cx: &mut PushCx<'_>,
        out: &mut StageOutput,
    ) -> VortexResult<NodeState> {
        if self.push_batching == PushBatching::Morsel && !self.push_passthrough {
            accept_morsel_struct_input(self, port, batch, last_for_input, out)?;
            return drain_morsel_struct(self, out);
        }
        accept_struct_input(self, port, batch, last_for_input)?;
        drain_struct(self, out)
    }

    #[inline]
    fn push_end(
        &mut self,
        port: crate::node::InputPort,
        _cx: &mut PushCx<'_>,
        out: &mut StageOutput,
    ) -> VortexResult<NodeState> {
        if self.push_batching == PushBatching::Morsel && !self.push_passthrough {
            accept_morsel_struct_end(self, port)?;
            return drain_morsel_struct(self, out);
        }
        accept_struct_end(self, port)?;
        drain_struct(self, out)
    }

    #[inline]
    fn push_resume(
        &mut self,
        _cx: &mut PushCx<'_>,
        out: &mut StageOutput,
    ) -> VortexResult<NodeState> {
        drain_struct(self, out)
    }

    #[inline]
    fn push_credit(
        &mut self,
        _cx: &mut PushCx<'_>,
        out: &mut StageOutput,
    ) -> VortexResult<NodeState> {
        if self.push_output_credit {
            return Err(vortex_err!("struct received duplicate output credit"));
        }
        self.push_output_credit = true;
        drain_struct(self, out)
    }

    fn retire(&mut self, cx: &mut RetireCx<'_>) {
        for &child in self.children.iter() {
            cx.retire_child(child);
        }
    }

    fn children(&self) -> &[NodeId] {
        &self.children
    }
}

fn accept_struct_input(
    node: &mut StructExec,
    port: crate::node::InputPort,
    batch: PushBatch,
    last_for_input: bool,
) -> VortexResult<()> {
    let idx = port.index();
    if idx >= node.children.len() {
        return Err(vortex_err!("struct received unknown field port {idx}"));
    }
    if node.push_ended[idx] {
        return Err(vortex_err!("struct field {idx} produced data after ending"));
    }
    if !node.push_heads[idx].is_empty() {
        return Err(vortex_err!(
            "struct field {idx} exceeded its one-batch edge credit"
        ));
    }
    let expected = node.push_cursor;
    if batch.coverage.start != expected || batch.coverage.end > node.range.end {
        return Err(vortex_err!(
            "struct field {idx} produced non-contiguous coverage {:?}; expected {expected}",
            batch.coverage
        ));
    }
    if last_for_input && batch.coverage.end != node.range.end {
        return Err(vortex_err!(
            "struct field {idx} ended at {}, expected {}",
            batch.coverage.end,
            node.range.end
        ));
    }
    if !last_for_input && batch.coverage.end == node.range.end {
        return Err(vortex_err!(
            "struct field {idx} covered its full range without a final marker"
        ));
    }
    node.push_ended[idx] = last_for_input;
    node.push_saw_batch[idx] = true;
    node.push_heads[idx].push_back(PendingField {
        batch,
        last: last_for_input,
    });
    Ok(())
}

fn accept_struct_end(node: &mut StructExec, port: crate::node::InputPort) -> VortexResult<()> {
    let idx = port.index();
    if idx >= node.children.len() {
        return Err(vortex_err!("struct received unknown field port {idx}"));
    }
    if node.push_ended[idx] {
        return Err(vortex_err!("struct field {idx} ended more than once"));
    }
    if node.push_saw_batch[idx] {
        return Err(vortex_err!(
            "struct field {idx} used End after emitting a batch"
        ));
    }
    if node.push_cursor != node.range.end {
        return Err(vortex_err!(
            "struct field {idx} ended without accounting for {}..{}",
            node.push_cursor,
            node.range.end
        ));
    }
    node.push_ended[idx] = true;
    Ok(())
}

fn drain_struct(node: &mut StructExec, out: &mut StageOutput) -> VortexResult<NodeState> {
    if node.done {
        return Ok(NodeState::Done);
    }
    if node.push_passthrough {
        return drain_predicate_passthrough(node, out);
    }
    if node.push_batching == PushBatching::Morsel {
        return drain_morsel_struct(node, out);
    }
    if node.push_output_credit
        && node.push_heads.iter().all(|heads| !heads.is_empty())
        && let Some(aligned) = take_aligned_fields(node, out)?
    {
        let AlignedFields {
            coverage,
            fields,
            selection,
            materialized,
            all_last,
        } = aligned;
        let array = StructArray::try_new(
            node.names.clone(),
            fields,
            materialized.true_count(),
            Validity::NonNullable,
        )?
        .into_array();
        let last = all_last && coverage.end == node.range.end;
        out.set_batch(
            PushBatch::from_validated_parts(coverage, selection, materialized, Value::Array(array)),
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

fn drain_predicate_passthrough(
    node: &mut StructExec,
    out: &mut StageOutput,
) -> VortexResult<NodeState> {
    if !node.push_output_credit {
        return Ok(NodeState::NeedInput);
    }
    let Some(pending) = node.push_heads.get_mut(0).and_then(VecDeque::pop_front) else {
        return Ok(NodeState::NeedInput);
    };
    let end = pending.batch.coverage.end;
    if !pending.last {
        out.push_consumed(crate::node::InputPort::new(0)?);
    }
    let last = pending.last && end == node.range.end;
    out.set_batch(pending.batch, last);
    node.push_cursor = end;
    node.push_output_credit = false;
    if last {
        node.done = true;
        return Ok(NodeState::Done);
    }
    Ok(NodeState::NeedInput)
}

struct AlignedFields {
    coverage: Range<u64>,
    fields: Vec<ArrayRef>,
    selection: Mask,
    materialized: Mask,
    all_last: bool,
}

fn take_aligned_fields(
    node: &mut StructExec,
    out: &mut StageOutput,
) -> VortexResult<Option<AlignedFields>> {
    if node.push_heads.iter().all(|heads| !heads.is_empty()) {
        let end = node
            .push_heads
            .iter()
            .filter_map(|heads| heads.front().map(|head| head.batch.coverage.end))
            .min()
            .ok_or_else(|| vortex_err!("struct alignment has no field heads"))?;
        let coverage = node.push_cursor..end;
        let mut fields = Vec::with_capacity(node.children.len());
        let mut selection = None;
        let mut materialized = None;
        let mut all_last = true;
        for (idx, heads) in node.push_heads.iter_mut().enumerate() {
            let mut head = heads.pop_front().vortex_expect("field head exists");
            let consumed_all = head.batch.coverage.end == end;
            let prefix = if consumed_all {
                let PendingField { batch, last } = head;
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
            if let Some(expected) = selection.as_ref()
                && expected != &prefix.selection
            {
                return Err(vortex_err!(
                    "struct fields have different selections over {coverage:?}"
                ));
            }
            selection.get_or_insert_with(|| prefix.selection.clone());
            align_materialization(&mut materialized, &prefix.materialized, &coverage)?;
            fields.push(prefix.value.into_array()?);
        }
        let selection = selection.vortex_expect("struct has at least one field");
        let materialized = materialized.vortex_expect("struct has at least one field");
        node.push_cursor = end;
        return Ok(Some(AlignedFields {
            coverage,
            fields,
            selection,
            materialized,
            all_last,
        }));
    }
    Ok(None)
}

fn accept_morsel_struct_input(
    node: &mut StructExec,
    port: crate::node::InputPort,
    batch: PushBatch,
    last_for_input: bool,
    out: &mut StageOutput,
) -> VortexResult<()> {
    let idx = port.index();
    if idx >= node.children.len() {
        return Err(vortex_err!("struct received unknown field port {idx}"));
    }
    if node.push_ended[idx] {
        return Err(vortex_err!("struct field {idx} produced data after ending"));
    }
    let expected = node.push_field_cursors[idx];
    if batch.coverage.start != expected || batch.coverage.end > node.range.end {
        return Err(vortex_err!(
            "struct field {idx} produced non-contiguous coverage {:?}; expected {expected}",
            batch.coverage
        ));
    }
    if last_for_input && batch.coverage.end != node.range.end {
        return Err(vortex_err!(
            "struct field {idx} ended at {}, expected {}",
            batch.coverage.end,
            node.range.end
        ));
    }
    if !last_for_input && batch.coverage.end == node.range.end {
        return Err(vortex_err!(
            "struct field {idx} covered its full range without a final marker"
        ));
    }
    if batch.selection != batch.materialized {
        return Err(vortex_err!(
            "morsel-batched projection struct requires identical selection and materialization"
        ));
    }
    let end = batch.coverage.end;
    let PushBatch {
        selection, value, ..
    } = batch;
    node.push_field_parts[idx].push(value.into_array()?);
    node.push_field_mask_parts[idx].push(selection);
    node.push_field_cursors[idx] = end;
    node.push_ended[idx] = last_for_input;
    node.push_saw_batch[idx] = true;
    if !last_for_input {
        out.push_consumed(port);
    }
    Ok(())
}

fn accept_morsel_struct_end(
    node: &mut StructExec,
    port: crate::node::InputPort,
) -> VortexResult<()> {
    let idx = port.index();
    if idx >= node.children.len() {
        return Err(vortex_err!("struct received unknown field port {idx}"));
    }
    if node.push_ended[idx] {
        return Err(vortex_err!("struct field {idx} ended more than once"));
    }
    if node.push_saw_batch[idx] {
        return Err(vortex_err!(
            "struct field {idx} used End after emitting a batch"
        ));
    }
    if node.push_field_cursors[idx] != node.range.end {
        return Err(vortex_err!(
            "struct field {idx} ended without accounting for {}..{}",
            node.push_field_cursors[idx],
            node.range.end
        ));
    }
    node.push_ended[idx] = true;
    Ok(())
}

fn drain_morsel_struct(node: &mut StructExec, out: &mut StageOutput) -> VortexResult<NodeState> {
    if !node.push_output_credit || node.push_ended.iter().any(|ended| !ended) {
        return Ok(NodeState::NeedInput);
    }

    let mut fields = Vec::with_capacity(node.push_field_parts.len());
    let mut authoritative_mask = None;
    let expected_len = usize::try_from(node.range.end - node.range.start)
        .map_err(|_| vortex_err!("struct range is too large"))?;
    for (idx, mask_parts) in node.push_field_mask_parts.iter().enumerate() {
        let mask = Mask::concat(mask_parts.iter())?;
        if mask.len() != expected_len {
            return Err(vortex_err!(
                "struct field {idx} mask covers {} rows, expected {expected_len}",
                mask.len()
            ));
        }
        if let Some(expected) = authoritative_mask.as_ref()
            && expected != &mask
        {
            return Err(vortex_err!(
                "struct fields have different selections over {:?}",
                node.range
            ));
        }
        authoritative_mask.get_or_insert(mask);
    }
    for parts in &mut node.push_field_parts {
        let field = match parts.len() {
            0 => return Err(vortex_err!("projection struct field produced no arrays")),
            1 => parts.pop().vortex_expect("one coalesced field part"),
            _ => {
                let dtype = parts[0].dtype().clone();
                let mut chunks = Vec::with_capacity(parts.len());
                chunks.append(parts);
                ChunkedArray::try_new(chunks, dtype)?.into_array()
            }
        };
        fields.push(field);
    }
    let selection = authoritative_mask.vortex_expect("struct has at least one field");
    let array = StructArray::try_new(
        node.names.clone(),
        fields,
        selection.true_count(),
        Validity::NonNullable,
    )?
    .into_array();
    out.set_batch(
        PushBatch::from_validated_parts(
            node.range.clone(),
            selection.clone(),
            selection,
            Value::Array(array),
        ),
        true,
    );
    node.push_output_credit = false;
    node.done = true;
    Ok(NodeState::Done)
}

fn align_materialization(
    expected: &mut Option<Mask>,
    actual: &Mask,
    coverage: &Range<u64>,
) -> VortexResult<()> {
    if expected.as_ref().is_some_and(|expected| expected != actual) {
        return Err(vortex_err!(
            "struct fields have different materialization domains over {coverage:?}"
        ));
    }
    expected.get_or_insert_with(|| actual.clone());
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_error::VortexResult;
    use vortex_mask::Mask;

    use super::StructExec;
    use super::accept_morsel_struct_input;
    use super::accept_struct_input;
    use super::align_materialization;
    use super::drain_struct;
    use crate::node::ExecNode;
    use crate::node::InputPort;
    use crate::node::PushBatch;
    use crate::node::StageOutput;
    use crate::node::StageSideband;
    use crate::node::Value;
    use crate::nodes::PushBatching;

    fn field_batch(
        coverage: std::ops::Range<u64>,
        mask: Mask,
        values: impl IntoIterator<Item = i32>,
    ) -> VortexResult<PushBatch> {
        PushBatch::try_new(
            coverage,
            mask,
            Value::Array(PrimitiveArray::from_iter(values).into_array()),
        )
    }

    #[test]
    fn struct_rejects_mismatched_materialization_domains() -> VortexResult<()> {
        let mut expected = None;
        align_materialization(
            &mut expected,
            &Mask::from_iter([true, false, true]),
            &(0..3),
        )?;
        assert!(
            align_materialization(
                &mut expected,
                &Mask::from_iter([true, true, false]),
                &(0..3),
            )
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn predicate_passthrough_preserves_array_masks_coverage_and_last() -> VortexResult<()> {
        let mut node = StructExec::new_with_push_batching(
            ["x"].into(),
            Arc::from([0]),
            PushBatching::Streaming,
            true,
        );
        node.reset(0..4);
        let selection = Mask::from_iter([true, false, true, false]);
        let array = PrimitiveArray::from_iter([10i32, 30]).into_array();
        let batch = PushBatch::try_new(0..4, selection.clone(), Value::Array(array.clone()))?;
        let mut out = StageOutput::default();

        accept_struct_input(&mut node, InputPort::new(0)?, batch, true)?;
        assert!(matches!(
            drain_struct(&mut node, &mut out)?,
            crate::node::NodeState::Done
        ));
        let (batch, last) = out
            .take_batch()
            .ok_or_else(|| vortex_error::vortex_err!("passthrough emitted no batch"))?;
        assert!(last);
        assert_eq!(batch.coverage, 0..4);
        assert_eq!(batch.selection, selection);
        assert_eq!(batch.selection, batch.materialized);
        let output = batch.value.into_array()?;
        assert!(vortex_array::ArrayRef::ptr_eq(&array, &output));
        assert!(out.take_sideband().is_none());
        Ok(())
    }

    #[test]
    fn predicate_passthrough_returns_nonterminal_credit_and_last_flag() -> VortexResult<()> {
        let mut node = StructExec::new_with_push_batching(
            ["x"].into(),
            Arc::from([0]),
            PushBatching::Streaming,
            true,
        );
        node.reset(0..4);
        let batch = field_batch(0..2, Mask::new_true(2), [1, 2])?;
        let mut out = StageOutput::default();

        accept_struct_input(&mut node, InputPort::new(0)?, batch, false)?;
        assert!(matches!(
            drain_struct(&mut node, &mut out)?,
            crate::node::NodeState::NeedInput
        ));
        let (_, last) = out
            .take_batch()
            .ok_or_else(|| vortex_error::vortex_err!("passthrough emitted no batch"))?;
        assert!(!last);
        assert!(matches!(
            out.take_sideband(),
            Some(StageSideband::Consumed(port)) if port.index() == 0
        ));
        Ok(())
    }

    #[test]
    #[allow(clippy::cognitive_complexity)]
    fn morsel_batching_accepts_misaligned_fields_and_later_field_finishes_first() -> VortexResult<()>
    {
        let mut node = StructExec::new_with_push_batching(
            ["a", "b"].into(),
            Arc::from([0, 1]),
            PushBatching::Morsel,
            false,
        );
        node.reset(0..6);
        let mut out = StageOutput::default();

        accept_morsel_struct_input(
            &mut node,
            InputPort::new(1)?,
            field_batch(0..1, Mask::from_iter([true]), [10])?,
            false,
            &mut out,
        )?;
        assert!(matches!(
            drain_struct(&mut node, &mut out)?,
            crate::node::NodeState::NeedInput
        ));
        assert!(out.take_batch().is_none());
        assert!(matches!(
            out.take_sideband(),
            Some(StageSideband::Consumed(port)) if port.index() == 1
        ));

        accept_morsel_struct_input(
            &mut node,
            InputPort::new(1)?,
            field_batch(
                1..6,
                Mask::from_iter([false, true, false, true, true]),
                [30, 50, 60],
            )?,
            true,
            &mut out,
        )?;
        assert!(matches!(
            drain_struct(&mut node, &mut out)?,
            crate::node::NodeState::NeedInput
        ));
        assert!(out.take_batch().is_none());
        assert!(out.take_sideband().is_none());

        accept_morsel_struct_input(
            &mut node,
            InputPort::new(0)?,
            field_batch(0..2, Mask::from_iter([true, false]), [1])?,
            false,
            &mut out,
        )?;
        assert!(matches!(
            drain_struct(&mut node, &mut out)?,
            crate::node::NodeState::NeedInput
        ));
        assert!(out.take_batch().is_none());
        assert!(matches!(
            out.take_sideband(),
            Some(StageSideband::Consumed(port)) if port.index() == 0
        ));

        accept_morsel_struct_input(
            &mut node,
            InputPort::new(0)?,
            field_batch(2..4, Mask::from_iter([true, false]), [3])?,
            false,
            &mut out,
        )?;
        assert!(matches!(
            drain_struct(&mut node, &mut out)?,
            crate::node::NodeState::NeedInput
        ));
        assert!(out.take_batch().is_none());
        assert!(matches!(
            out.take_sideband(),
            Some(StageSideband::Consumed(port)) if port.index() == 0
        ));

        accept_morsel_struct_input(
            &mut node,
            InputPort::new(0)?,
            field_batch(4..6, Mask::from_iter([true, true]), [5, 6])?,
            true,
            &mut out,
        )?;
        assert!(matches!(
            drain_struct(&mut node, &mut out)?,
            crate::node::NodeState::Done
        ));
        let (batch, last) = out
            .take_batch()
            .ok_or_else(|| vortex_error::vortex_err!("coalesced struct emitted no batch"))?;
        assert!(last);
        assert_eq!(batch.coverage, 0..6);
        assert_eq!(
            batch.selection,
            Mask::from_iter([true, false, true, false, true, true])
        );
        assert_eq!(batch.selection, batch.materialized);
        assert_eq!(batch.value.into_array()?.len(), 4);

        let field_capacity = node.push_field_parts.capacity();
        node.reset(6..8);
        assert!(node.push_field_parts.iter().all(Vec::is_empty));
        assert!(node.push_field_mask_parts.iter().all(Vec::is_empty));
        assert!(node.push_field_parts.capacity() >= field_capacity);
        Ok(())
    }
}
