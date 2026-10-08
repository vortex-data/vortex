// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_error::VortexResult;

use crate::plan::ConcatPlan;
use crate::plan::exec::ExecNode;
use crate::plan::exec::NodeState;
use crate::plan::exec::StepCx;
use crate::plan::exec::selection::Selection;

/// Emits its chunks' rows in chunk order.
///
/// Only chunks overlapping the selection's rows are spawned, one port each in chunk order, and
/// a chunk whose slice of the mask is all false is skipped. Chunks are read independently and
/// may finish in any order; the node emits whatever the current chunk has produced, moves on
/// once that chunk has finished, and leaves what later chunks produced early waiting in their
/// ports. Nothing above it sees the disorder.
pub(crate) struct ConcatNode {
    plan: ConcatPlan,
    selection: Selection,
    /// The port whose chunk is being emitted. Ports are numbered in chunk order.
    current: usize,
    ports: usize,
}

impl ConcatNode {
    pub(crate) fn new(plan: ConcatPlan, selection: Selection) -> Self {
        Self {
            plan,
            selection,
            current: 0,
            ports: 0,
        }
    }

    fn chunk_rows(&self, index: usize) -> Range<u64> {
        let offsets = self.plan.row_offsets();
        let end = offsets
            .get(index + 1)
            .copied()
            .unwrap_or_else(|| self.plan.row_count());
        offsets[index]..end
    }
}

impl ExecNode for ConcatNode {
    fn start(&mut self, cx: &mut StepCx<'_>) -> VortexResult<NodeState> {
        let rows = self.selection.rows().clone();
        let offsets = self.plan.row_offsets();
        // Every split visits a small part of a file; skip the chunks before and after it.
        let first = offsets
            .partition_point(|&offset| offset <= rows.start)
            .saturating_sub(1);
        let end = offsets.partition_point(|&offset| offset < rows.end);
        for index in first..end {
            let chunk = self.chunk_rows(index);
            let local = rows.start.max(chunk.start)..rows.end.min(chunk.end);
            if local.start >= local.end {
                continue;
            }
            let mask = self.selection.slice(&local)?;
            if mask.all_false() {
                continue;
            }
            cx.spawn(
                self.ports,
                self.plan.child_required(index)?,
                local.start - chunk.start..local.end - chunk.start,
                mask,
            );
            self.ports += 1;
        }
        if self.ports == 0 {
            return Ok(NodeState::Done);
        }
        Ok(NodeState::Wait)
    }

    fn compute(&mut self, cx: &mut StepCx<'_>) -> VortexResult<NodeState> {
        while self.current < self.ports {
            let input = cx.input(self.current);
            let arrays = input.take_all();
            let finished = input.finished();
            for array in arrays {
                cx.emit(array);
            }
            if !finished {
                return Ok(NodeState::Wait);
            }
            self.current += 1;
        }
        Ok(NodeState::Done)
    }
}
