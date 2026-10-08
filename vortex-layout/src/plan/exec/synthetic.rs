// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Plans whose exec nodes are built by hand, for exercising the graph without layouts.
//!
//! A [`SyntheticPlan`] carries a closure that builds its node. The graph treats it like any
//! other operator, so a test or benchmark can wire hand-written sources under the real
//! operators (Pack, Concat, Filter, Eval) and drive the whole thing through [`ExecGraph`],
//! reading segments from memory.
//!
//! [`ExecGraph`]: crate::plan::exec::ExecGraph

use std::fmt;
use std::ops::Range;
use std::sync::Arc;

use vortex_array::EmptyMetadata;
use vortex_array::IntoArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_session::registry::CachedId;

use crate::plan::Plan;
use crate::plan::PlanChildren;
use crate::plan::PlanId;
use crate::plan::PlanParts;
use crate::plan::PlanRef;
use crate::plan::PlanVTable;
use crate::plan::exec::ExecContext;
use crate::plan::exec::ExecNode;
use crate::plan::exec::NodeState;
use crate::plan::exec::Ready;
use crate::plan::exec::StepCx;
use crate::segments::SegmentId;

/// Builds the node of a [`SyntheticPlan`] for one execution.
pub type NodeBuilder =
    dyn Fn(Range<u64>, Mask, &ExecContext) -> VortexResult<Box<dyn ExecNode>> + Send + Sync;

/// A plan operator whose node is built by a closure.
#[derive(Clone, Debug)]
pub struct Synthetic;

/// The node builder of a [`Synthetic`] plan.
#[derive(Clone)]
pub struct SyntheticData {
    builder: Arc<NodeBuilder>,
}

impl fmt::Debug for SyntheticData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("SyntheticData")
    }
}

/// A plan whose node is built by a closure.
pub type SyntheticPlan = Plan<Synthetic>;

impl SyntheticPlan {
    /// A plan of `row_count` rows of `dtype`, with `children`, whose node `builder` makes.
    pub fn new(
        dtype: DType,
        row_count: u64,
        children: Vec<PlanRef>,
        builder: impl Fn(Range<u64>, Mask, &ExecContext) -> VortexResult<Box<dyn ExecNode>>
        + Send
        + Sync
        + 'static,
    ) -> Self {
        PlanParts {
            vtable: Synthetic,
            dtype,
            row_count,
            children: children.into(),
            data: SyntheticData {
                builder: Arc::new(builder),
            },
        }
        .into_typed()
    }
}

impl PlanVTable for Synthetic {
    type PlanData = SyntheticData;
    type Metadata = EmptyMetadata;

    fn id(&self) -> PlanId {
        static ID: CachedId = CachedId::new("vortex.plan.synthetic");
        *ID
    }

    fn metadata(_plan: &Plan<Self>) -> Option<Self::Metadata> {
        None
    }

    fn with_children(
        _plan: &Plan<Self>,
        _children: &PlanChildren,
        _data: &mut Self::PlanData,
    ) -> VortexResult<()> {
        Ok(())
    }

    fn exec(
        plan: &Plan<Self>,
        rows: Range<u64>,
        mask: Mask,
        ctx: &ExecContext,
    ) -> VortexResult<Box<dyn ExecNode>> {
        (plan.data().builder)(rows, mask, ctx)
    }
}

/// The dtype every synthetic source produces: its rows' indices, as `u64`.
pub fn row_dtype() -> DType {
    DType::Primitive(PType::U64, Nullability::NonNullable)
}

/// A source that emits its rows' indices, in order, cut into pieces of the given lengths,
/// optionally after a round trip through the IO service and optionally yielding between
/// pieces.
///
/// A selected source emits only the rows its mask selects, as every operator does. A dense
/// source ignores its mask and emits every row of its range, as a bare segment scan does, so
/// a filter plan can sit over it. The indices are the plan's own rows, so
/// a parent that reorders or filters can be checked against a sequence. Piece lengths count
/// emitted rows; the last piece takes whatever remains.
pub struct RowSource {
    /// Remaining indices to emit, in order.
    remaining: Vec<u64>,
    /// Lengths of the pieces still to emit; the last piece takes whatever remains.
    pieces: Vec<usize>,
    /// A segment to request before emitting anything, standing in for a read.
    segment: Option<SegmentId>,
    /// Whether to return `Yield` between pieces instead of emitting them all at once.
    yielding: bool,
    requested: bool,
}

impl RowSource {
    /// A source plan of `row_count` rows that emits pieces of the given lengths.
    pub fn plan(
        row_count: u64,
        pieces: Vec<usize>,
        segment: Option<SegmentId>,
        yielding: bool,
        dense: bool,
    ) -> PlanRef {
        SyntheticPlan::new(row_dtype(), row_count, Vec::new(), move |rows, mask, _| {
            let remaining = if dense || mask.all_true() {
                (rows.start..rows.end).collect()
            } else {
                (rows.start..rows.end)
                    .zip(mask.iter())
                    .filter_map(|(row, selected)| selected.then_some(row))
                    .collect()
            };
            Ok(Box::new(RowSource {
                remaining,
                pieces: pieces.iter().rev().copied().collect(),
                segment,
                yielding,
                requested: false,
            }))
        })
        .into_plan()
    }

    /// Emits the next piece, or everything left when no lengths remain.
    fn emit_next(&mut self, cx: &mut StepCx<'_>) {
        let len = self
            .pieces
            .pop()
            .unwrap_or(self.remaining.len())
            .min(self.remaining.len());
        let piece: Vec<u64> = self.remaining.drain(..len).collect();
        cx.emit(Buffer::from(piece).into_array());
    }

    fn emit(&mut self, cx: &mut StepCx<'_>) -> NodeState {
        if self.yielding {
            self.emit_next(cx);
            if self.remaining.is_empty() {
                return NodeState::Done;
            }
            return NodeState::Yield;
        }
        while !self.remaining.is_empty() {
            self.emit_next(cx);
        }
        NodeState::Done
    }
}

impl ExecNode for RowSource {
    fn start(&mut self, cx: &mut StepCx<'_>) -> VortexResult<NodeState> {
        if self.remaining.is_empty() {
            return Ok(NodeState::Done);
        }
        if let Some(segment) = self.segment {
            self.requested = true;
            cx.request(segment);
            return Ok(NodeState::Wait);
        }
        Ok(self.emit(cx))
    }

    fn compute(&mut self, cx: &mut StepCx<'_>) -> VortexResult<NodeState> {
        if self.requested {
            self.requested = false;
            cx.take_delivery()
                .ok_or_else(|| vortex_err!("RowSource ran without its delivery"))?;
        }
        Ok(self.emit(cx))
    }
}

/// A node with a chosen readiness rule that records the state of its ports every time it runs,
/// then passes its first port through.
///
/// It spawns one child per plan child, each over the node's own rows and selection.
pub struct Probe {
    ready: Ready,
    children: Vec<PlanRef>,
    rows: Range<u64>,
    mask: Mask,
    /// For each compute: whether each port was closed when the node ran.
    pub runs: Arc<parking_lot::Mutex<Vec<Vec<bool>>>>,
}

impl Probe {
    /// A plan over `children` whose node runs under `ready` and logs into `runs`.
    pub fn plan(
        children: Vec<PlanRef>,
        ready: Ready,
        runs: Arc<parking_lot::Mutex<Vec<Vec<bool>>>>,
    ) -> PlanRef {
        let row_count = children.first().map_or(0, |child| child.row_count());
        SyntheticPlan::new(
            row_dtype(),
            row_count,
            children.clone(),
            move |rows, mask, _| {
                Ok(Box::new(Probe {
                    ready,
                    children: children.clone(),
                    rows,
                    mask,
                    runs: Arc::clone(&runs),
                }))
            },
        )
        .into_plan()
    }
}

impl ExecNode for Probe {
    fn ready(&self) -> Ready {
        self.ready
    }

    fn start(&mut self, cx: &mut StepCx<'_>) -> VortexResult<NodeState> {
        for (port, child) in self.children.iter().enumerate() {
            cx.spawn(port, child.clone(), self.rows.clone(), self.mask.clone());
        }
        Ok(NodeState::Wait)
    }

    fn compute(&mut self, cx: &mut StepCx<'_>) -> VortexResult<NodeState> {
        let closed = cx.inputs().iter().map(|input| input.closed()).collect();
        self.runs.lock().push(closed);
        for array in cx.input(0).take_all() {
            cx.emit(array);
        }
        for port in 1..self.children.len() {
            cx.input(port).take_all();
        }
        if cx.all_finished() {
            return Ok(NodeState::Done);
        }
        Ok(NodeState::Wait)
    }
}

/// An in-memory segment store that answers any segment id with an empty buffer, for sources
/// whose read is only a scheduling event.
pub fn empty_segment() -> BufferHandle {
    BufferHandle::new_host(Buffer::<u8>::empty().into_byte_buffer())
}
