// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::VecDeque;
use std::ops::Range;
use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::dtype::DType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::Mask;

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

/// One overlap between the morsel's range and a chunk.
#[derive(Clone, Debug)]
struct Cut {
    chunk: usize,
    /// Rows within the chunk.
    chunk_range: Range<u64>,
    /// The slice of the demand mask that covers this overlap.
    mask_range: Range<usize>,
}

struct PendingBatch {
    batch: PushBatch,
    last: bool,
}

/// Chunked has no runtime existence beyond cutting: it turns one range into per-chunk ranges and
/// wraps the children's outputs back up in chunk order.
///
/// The cut is `partition_point` plus a walk of the overlapping chunks — chunks outside the
/// morsel are arithmetic that never ran, not objects that were created and discarded.
pub struct ChunkedExec {
    chunk_offsets: Arc<[u64]>,
    children: Arc<[NodeId]>,
    dtype: DType,

    // Per-morsel state.
    range: Range<u64>,
    cuts: Vec<Cut>,
    /// Index into `cuts` of the child currently being planned.
    plan_cursor: usize,
    /// Whether `plan_cursor`'s child has already been reset for this morsel.
    plan_started: bool,
    exec_cursor: usize,
    parts: Vec<ArrayRef>,
    done: bool,
    push_next: usize,
    push_received: Vec<u64>,
    push_pending: Vec<VecDeque<PendingBatch>>,
    push_ended: Vec<bool>,
    push_saw_batch: Vec<bool>,
    push_output_credit: bool,
}

impl ChunkedExec {
    /// Build a chunked node from cumulative chunk offsets and one child per chunk.
    pub fn new(chunk_offsets: Arc<[u64]>, children: Arc<[NodeId]>, dtype: DType) -> Self {
        debug_assert_eq!(chunk_offsets.len(), children.len() + 1);
        Self {
            chunk_offsets,
            children,
            dtype,
            range: 0..0,
            cuts: Vec::new(),
            plan_cursor: 0,
            plan_started: false,
            exec_cursor: 0,
            parts: Vec::new(),
            done: false,
            push_next: 0,
            push_received: Vec::new(),
            push_pending: Vec::new(),
            push_ended: Vec::new(),
            push_saw_batch: Vec::new(),
            push_output_credit: true,
        }
    }

    fn cut(&mut self) {
        self.cuts.clear();
        if self.range.is_empty() {
            return;
        }

        let offsets = &self.chunk_offsets;
        let first = offsets
            .partition_point(|&offset| offset <= self.range.start)
            .saturating_sub(1);
        let mut mask_start = 0usize;
        for chunk in first..self.children.len() {
            let chunk_start = offsets[chunk];
            let chunk_end = offsets[chunk + 1];
            if chunk_start >= self.range.end {
                break;
            }
            let overlap_start = self.range.start.max(chunk_start);
            let overlap_end = self.range.end.min(chunk_end);
            if overlap_start >= overlap_end {
                continue;
            }
            let len = usize::try_from(overlap_end - overlap_start)
                .vortex_expect("chunk overlap fits usize");
            self.cuts.push(Cut {
                chunk,
                chunk_range: overlap_start - chunk_start..overlap_end - chunk_start,
                mask_range: mask_start..mask_start + len,
            });
            mask_start += len;
        }
    }

    /// Accept the common one-cut terminal input without queueing and re-emitting the same batch.
    ///
    /// Coverage on push edges is already expressed in root coordinates. Consequently a morsel
    /// wholly contained in one physical chunk needs no mapping here; Chunked only has observable
    /// work when it orders multiple cuts or retains a non-terminal batch behind edge credit.
    pub(crate) fn try_accept_single_cut_terminal(
        &mut self,
        port: crate::node::InputPort,
        batch: &PushBatch,
        last_for_input: bool,
    ) -> VortexResult<bool> {
        let [cut] = self.cuts.as_slice() else {
            return Ok(false);
        };
        if !last_for_input
            || cut.chunk != port.index()
            || self.done
            || self.push_next != 0
            || !self.push_output_credit
            || self.push_ended[0]
            || self.push_saw_batch[0]
            || !self.push_pending[0].is_empty()
            || batch.coverage != self.range
        {
            return Ok(false);
        }

        let expected_end = self.range.start + cut.mask_range.end as u64;
        if self.push_received[0] != self.range.start || expected_end != self.range.end {
            return Ok(false);
        }

        self.push_received[0] = batch.coverage.end;
        self.push_saw_batch[0] = true;
        self.push_ended[0] = true;
        self.push_next = 1;
        self.push_output_credit = false;
        self.done = true;
        Ok(true)
    }
}

impl ExecNode for ChunkedExec {
    fn push_profile_kind(&self) -> crate::node::PushProfileKind {
        crate::node::PushProfileKind::Chunked
    }

    fn reset(&mut self, range: Range<u64>) {
        self.range = range;
        self.plan_cursor = 0;
        self.plan_started = false;
        self.exec_cursor = 0;
        self.parts.clear();
        self.done = false;
        self.cut();
        self.push_next = 0;
        let width = self.cuts.len();
        self.push_received.resize(width, 0);
        for (received, cut) in self.push_received.iter_mut().zip(&self.cuts) {
            *received = self.range.start + cut.mask_range.start as u64;
        }
        self.push_pending.truncate(width);
        self.push_pending.resize_with(width, VecDeque::new);
        self.push_pending.iter_mut().for_each(VecDeque::clear);
        self.push_ended.resize(width, false);
        self.push_ended.fill(false);
        self.push_saw_batch.resize(width, false);
        self.push_saw_batch.fill(false);
        self.push_output_credit = true;
    }

    fn next_plan(&mut self, cx: &mut PlanCx<'_>) -> VortexResult<PlanPoll> {
        while self.plan_cursor < self.cuts.len() {
            if cx.out_of_budget() {
                return Ok(PlanPoll::Item(PlanItem::Plan));
            }
            let cut = self.cuts[self.plan_cursor].clone();
            let fresh = !self.plan_started;
            self.plan_started = true;
            if cx.plan_child(self.children[cut.chunk], cut.chunk_range, fresh)? {
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

        if self.cuts.is_empty() {
            self.done = true;
            return Ok(ExecPoll::Value(ValueBatch {
                coverage: self.range.clone(),
                value: Value::Array(Canonical::empty(&self.dtype).into_array()),
            }));
        }

        let demand = cx.demand().clone();
        if self.parts.capacity() < self.cuts.len() {
            self.parts
                .reserve(self.cuts.len().saturating_sub(self.parts.len()));
        }
        while self.exec_cursor < self.cuts.len() {
            let cut = self.cuts[self.exec_cursor].clone();
            let child_demand = slice_mask(&demand, cut.mask_range);
            let child = self.children[cut.chunk];
            match cx.child_array(child, child_demand)? {
                ChildPoll::Value(array) => {
                    if !array.is_empty() {
                        self.parts.push(array);
                    }
                    self.exec_cursor += 1;
                }
                ChildPoll::Blocked(waits) => return Ok(ExecPoll::Blocked(waits)),
                ChildPoll::Done => {
                    return Err(vortex_err!("chunked child {child} produced no value"));
                }
            }
        }

        let parts = std::mem::take(&mut self.parts);
        let array = match parts.len() {
            0 => Canonical::empty(&self.dtype).into_array(),
            1 => parts.into_iter().next().vortex_expect("one part"),
            _ => {
                let dtype = parts[0].dtype().clone();
                ChunkedArray::try_new(parts, dtype)?.into_array()
            }
        };
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
        _cx: &mut PushCx<'_>,
        out: &mut StageOutput,
    ) -> VortexResult<NodeState> {
        accept_chunked_input(self, port, batch, last_for_input)?;
        drain_chunked(self, out)
    }

    #[inline]
    fn push_end(
        &mut self,
        port: crate::node::InputPort,
        _cx: &mut PushCx<'_>,
        out: &mut StageOutput,
    ) -> VortexResult<NodeState> {
        accept_chunked_end(self, port)?;
        drain_chunked(self, out)
    }

    #[inline]
    fn push_resume(
        &mut self,
        _cx: &mut PushCx<'_>,
        out: &mut StageOutput,
    ) -> VortexResult<NodeState> {
        drain_chunked(self, out)
    }

    #[inline]
    fn push_credit(
        &mut self,
        _cx: &mut PushCx<'_>,
        out: &mut StageOutput,
    ) -> VortexResult<NodeState> {
        if self.push_output_credit {
            return Err(vortex_err!("chunked received duplicate output credit"));
        }
        self.push_output_credit = true;
        drain_chunked(self, out)
    }

    fn retire(&mut self, cx: &mut RetireCx<'_>) {
        for cut in std::mem::take(&mut self.cuts) {
            cx.retire_child(self.children[cut.chunk]);
        }
    }

    fn children(&self) -> &[NodeId] {
        &self.children
    }
}

fn accept_chunked_input(
    node: &mut ChunkedExec,
    port: crate::node::InputPort,
    batch: PushBatch,
    last_for_input: bool,
) -> VortexResult<()> {
    if node.done {
        return Ok(());
    }
    let cut_idx = node
        .cuts
        .iter()
        .position(|cut| cut.chunk == port.index())
        .ok_or_else(|| vortex_err!("chunked received inactive port {}", port.index()))?;
    if node.push_ended[cut_idx] {
        return Err(vortex_err!(
            "chunked input {} produced data after its terminal batch",
            port.index()
        ));
    }
    if !node.push_pending[cut_idx].is_empty() {
        return Err(vortex_err!(
            "chunked input {} exceeded its one-batch edge credit",
            port.index()
        ));
    }
    let expected_end = node.range.start + node.cuts[cut_idx].mask_range.end as u64;
    if batch.coverage.start != node.push_received[cut_idx] || batch.coverage.end > expected_end {
        return Err(vortex_err!(
            "chunked input {} produced non-contiguous coverage {:?}; expected {}..={expected_end}",
            port.index(),
            batch.coverage,
            node.push_received[cut_idx]
        ));
    }
    if last_for_input && batch.coverage.end != expected_end {
        return Err(vortex_err!(
            "chunked input {} ended at {}, expected {expected_end}",
            port.index(),
            batch.coverage.end
        ));
    }
    if !last_for_input && batch.coverage.end == expected_end {
        return Err(vortex_err!(
            "chunked input {} covered its full range without a final marker",
            port.index()
        ));
    }
    node.push_received[cut_idx] = batch.coverage.end;
    node.push_saw_batch[cut_idx] = true;
    node.push_ended[cut_idx] = last_for_input;
    node.push_pending[cut_idx].push_back(PendingBatch {
        batch,
        last: last_for_input,
    });
    Ok(())
}

fn accept_chunked_end(node: &mut ChunkedExec, port: crate::node::InputPort) -> VortexResult<()> {
    if node.done {
        return Ok(());
    }
    let cut_idx = node
        .cuts
        .iter()
        .position(|cut| cut.chunk == port.index())
        .ok_or_else(|| vortex_err!("chunked received inactive port {}", port.index()))?;
    if node.push_ended[cut_idx] {
        return Err(vortex_err!(
            "chunked input {} ended more than once",
            port.index()
        ));
    }
    if node.push_saw_batch[cut_idx] {
        return Err(vortex_err!(
            "chunked input {} used End after emitting a batch",
            port.index()
        ));
    }
    let expected_end = node.range.start + node.cuts[cut_idx].mask_range.end as u64;
    if node.push_received[cut_idx] != expected_end {
        return Err(vortex_err!(
            "chunked input {} ended without accounting for {}..{expected_end}",
            port.index(),
            node.push_received[cut_idx]
        ));
    }
    node.push_ended[cut_idx] = true;
    Ok(())
}

fn drain_chunked(node: &mut ChunkedExec, out: &mut StageOutput) -> VortexResult<NodeState> {
    if node.done {
        return Ok(NodeState::Done);
    }
    stage_chunked_output(node, out)
}

fn stage_chunked_output(node: &mut ChunkedExec, out: &mut StageOutput) -> VortexResult<NodeState> {
    while node.push_output_credit && node.push_next < node.cuts.len() {
        let idx = node.push_next;
        let Some(pending) = node.push_pending[idx].pop_front() else {
            if node.push_ended[idx] {
                node.push_next += 1;
                continue;
            }
            break;
        };
        let node_last = pending.last && idx + 1 == node.cuts.len();
        if !pending.last {
            out.push_consumed(crate::node::InputPort::new(node.cuts[idx].chunk)?);
        }
        out.set_batch(pending.batch, node_last);
        node.push_output_credit = false;
        if pending.last {
            node.push_next += 1;
        }
        if node_last {
            node.done = true;
            return Ok(NodeState::Done);
        }
        return Ok(NodeState::NeedInput);
    }
    if node.push_next == node.cuts.len() {
        out.set_end();
        node.done = true;
        return Ok(NodeState::Done);
    }
    Ok(NodeState::NeedInput)
}

/// Slice a mask, preserving the all-true / all-false fast paths.
pub(crate) fn slice_mask(mask: &Mask, range: Range<usize>) -> Mask {
    if range.start == 0 && range.end == mask.len() {
        return mask.clone();
    }
    mask.slice(range)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_array::Canonical;
    use vortex_array::IntoArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_error::VortexResult;
    use vortex_mask::Mask;

    use super::ChunkedExec;
    use super::PendingBatch;
    use super::stage_chunked_output;
    use crate::node::ExecNode;
    use crate::node::InputPort;
    use crate::node::PushBatch;
    use crate::node::StageOutput;
    use crate::node::Value;

    fn empty_batch(coverage: std::ops::Range<u64>, dtype: &DType) -> VortexResult<PushBatch> {
        let len = usize::try_from(coverage.end - coverage.start)
            .map_err(|_| vortex_error::vortex_err!("test coverage exceeds usize"))?;
        PushBatch::try_new(
            coverage,
            Mask::new_false(len),
            Value::Array(Canonical::empty(dtype).into_array()),
        )
    }

    #[test]
    fn later_child_waits_for_multiple_earlier_batches() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let mut node = ChunkedExec::new(Arc::from([0, 4, 6]), Arc::from([0, 1]), dtype.clone());
        node.reset(0..6);

        node.push_pending[1].push_back(PendingBatch {
            batch: empty_batch(4..6, &dtype)?,
            last: true,
        });
        node.push_ended[1] = true;
        let mut output = StageOutput::default();
        assert!(matches!(
            stage_chunked_output(&mut node, &mut output)?,
            crate::node::NodeState::NeedInput
        ));
        assert!(output.is_empty());

        node.push_pending[0].push_back(PendingBatch {
            batch: empty_batch(0..2, &dtype)?,
            last: false,
        });
        let mut coverage = Vec::new();
        let mut credits = Vec::new();
        drain_staged(&mut node, &mut coverage, &mut credits)?;
        assert_eq!(coverage.len(), 1);
        assert_eq!(coverage[0], 0..2);

        node.push_pending[0].push_back(PendingBatch {
            batch: empty_batch(2..4, &dtype)?,
            last: true,
        });
        node.push_ended[0] = true;
        while !node.done {
            drain_staged(&mut node, &mut coverage, &mut credits)?;
        }
        assert_eq!(coverage, [0..2, 2..4, 4..6]);
        assert_eq!(
            credits,
            [1, 0, 0],
            "only a fully consumed nonterminal head returns credit"
        );
        Ok(())
    }

    #[test]
    fn reset_reuses_queue_storage_and_handles_cut_width_changes() {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let mut node = ChunkedExec::new(Arc::from([0, 4, 6]), Arc::from([0, 1]), dtype);
        node.reset(0..6);
        node.push_pending[0].reserve(8);
        let queues_capacity = node.push_pending.capacity();
        let first_queue_capacity = node.push_pending[0].capacity();
        node.push_ended.fill(true);
        node.push_saw_batch.fill(true);

        node.reset(0..4);

        assert_eq!(node.push_pending.len(), 1);
        assert!(node.push_pending.capacity() >= queues_capacity);
        assert!(node.push_pending[0].capacity() >= first_queue_capacity);
        assert!(node.push_pending[0].is_empty());
        assert_eq!(node.push_received, [0]);
        assert_eq!(node.push_ended, [false]);
        assert_eq!(node.push_saw_batch, [false]);

        node.reset(0..6);

        assert_eq!(node.push_pending.len(), 2);
        assert!(node.push_pending.iter().all(|queue| queue.is_empty()));
        assert_eq!(node.push_received, [0, 4]);
        assert_eq!(node.push_ended, [false, false]);
        assert_eq!(node.push_saw_batch, [false, false]);
    }

    #[test]
    fn single_cut_terminal_batch_can_bypass_queue() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let mut node = ChunkedExec::new(Arc::from([0, 4, 9]), Arc::from([0, 1]), dtype.clone());
        node.reset(4..9);
        let batch = empty_batch(4..9, &dtype)?;

        assert!(node.try_accept_single_cut_terminal(InputPort::new(1)?, &batch, true)?);
        assert!(node.done);
        assert_eq!(node.push_next, 1);
        assert_eq!(node.push_received, [9]);
        assert_eq!(node.push_ended, [true]);
        assert_eq!(node.push_saw_batch, [true]);
        assert!(node.push_pending[0].is_empty());
        assert!(!node.push_output_credit);
        Ok(())
    }

    #[test]
    fn bypass_rejects_partial_and_multi_cut_inputs_without_mutation() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let mut node = ChunkedExec::new(Arc::from([0, 4, 9]), Arc::from([0, 1]), dtype.clone());
        node.reset(0..4);
        let partial = empty_batch(0..2, &dtype)?;
        assert!(!node.try_accept_single_cut_terminal(InputPort::new(0)?, &partial, false)?);
        assert_eq!(node.push_received, [0]);
        assert!(!node.done);

        node.reset(0..9);
        let later = empty_batch(4..9, &dtype)?;
        assert!(!node.try_accept_single_cut_terminal(InputPort::new(1)?, &later, true)?);
        assert_eq!(node.push_received, [0, 4]);
        assert!(node.push_pending.iter().all(|queue| queue.is_empty()));
        assert!(!node.done);
        Ok(())
    }

    #[test]
    fn reset_after_single_cut_bypass_restores_general_ordering_state() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let mut node = ChunkedExec::new(Arc::from([0, 4, 9]), Arc::from([0, 1]), dtype.clone());
        node.reset(0..4);
        let first = empty_batch(0..4, &dtype)?;
        assert!(node.try_accept_single_cut_terminal(InputPort::new(0)?, &first, true)?);

        node.reset(0..9);
        assert_eq!(node.push_next, 0);
        assert_eq!(node.push_received, [0, 4]);
        assert_eq!(node.push_ended, [false, false]);
        assert_eq!(node.push_saw_batch, [false, false]);
        assert!(node.push_pending.iter().all(|queue| queue.is_empty()));
        assert!(node.push_output_credit);
        assert!(!node.done);
        Ok(())
    }

    fn drain_staged(
        node: &mut ChunkedExec,
        coverage: &mut Vec<std::ops::Range<u64>>,
        credits: &mut Vec<usize>,
    ) -> VortexResult<()> {
        let mut output = StageOutput::default();
        let _state = stage_chunked_output(node, &mut output)?;
        let mut consumed = 0;
        while output.take_sideband().is_some() {
            consumed += 1;
        }
        if let Some((batch, _)) = output.take_batch() {
            coverage.push(batch.coverage);
            credits.push(consumed);
            node.push_output_credit = true;
        }
        Ok(())
    }
}
