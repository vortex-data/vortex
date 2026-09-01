// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

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
use crate::node::PlanCx;
use crate::node::PlanItem;
use crate::node::PlanPoll;
use crate::node::RetireCx;
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
}

impl ExecNode for ChunkedExec {
    fn reset(&mut self, range: Range<u64>) {
        self.range = range;
        self.plan_cursor = 0;
        self.plan_started = false;
        self.exec_cursor = 0;
        self.parts.clear();
        self.done = false;
        self.cut();
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

    fn retire(&mut self, cx: &mut RetireCx<'_>) {
        for cut in std::mem::take(&mut self.cuts) {
            cx.retire_child(self.children[cut.chunk]);
        }
    }

    fn children(&self) -> &[NodeId] {
        &self.children
    }
}

/// Slice a mask, preserving the all-true / all-false fast paths.
pub(crate) fn slice_mask(mask: &Mask, range: Range<usize>) -> Mask {
    if range.start == 0 && range.end == mask.len() {
        return mask.clone();
    }
    mask.slice(range)
}
