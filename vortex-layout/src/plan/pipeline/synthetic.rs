// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Plans whose chains are built by hand, for exercising pipelines without layouts.
//!
//! A [`SyntheticPlan`] carries a closure that compiles it. A scan treats it like any other
//! plan, so a test or benchmark can wire hand-written sources under the real operators and run
//! the whole thing through a [`Scan`](super::Scan), answering reads from memory.

use std::fmt;
use std::ops::Range;
use std::sync::Arc;

use parking_lot::Mutex;
use vortex_array::ArrayRef;
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

use super::Blocked;
use super::Chain;
use super::Compiler;
use super::Cx;
use super::Input;
use super::Operator;
use super::Source;
use super::Step;
use crate::plan::Plan;
use crate::plan::PlanChildren;
use crate::plan::PlanId;
use crate::plan::PlanParts;
use crate::plan::PlanRef;
use crate::plan::PlanVTable;
use crate::segments::SegmentId;

/// Compiles a [`SyntheticPlan`] over some rows, restricted to a mask.
pub type ChainBuilder =
    dyn Fn(Range<u64>, &Mask, &mut Compiler<'_>) -> VortexResult<Option<Chain>> + Send + Sync;

/// A plan operator whose chain is built by a closure.
#[derive(Clone, Debug)]
pub struct Synthetic;

/// The chain builder of a [`Synthetic`] plan.
#[derive(Clone)]
pub struct SyntheticData {
    builder: Arc<ChainBuilder>,
}

impl fmt::Debug for SyntheticData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("SyntheticData")
    }
}

/// A plan whose chain is built by a closure.
pub type SyntheticPlan = Plan<Synthetic>;

impl SyntheticPlan {
    /// A plan of `row_count` rows of `dtype`, with `children`, compiled by `builder`.
    pub fn new(
        dtype: DType,
        row_count: u64,
        children: Vec<PlanRef>,
        builder: impl Fn(Range<u64>, &Mask, &mut Compiler<'_>) -> VortexResult<Option<Chain>>
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

    fn compile(
        plan: &Plan<Self>,
        rows: Range<u64>,
        mask: &Mask,
        compiler: &mut Compiler<'_>,
    ) -> VortexResult<Option<Chain>> {
        (plan.data().builder)(rows, mask, compiler)
    }
}

/// The dtype every synthetic source produces: its rows' indices, as `u64`.
pub fn row_dtype() -> DType {
    DType::Primitive(PType::U64, Nullability::NonNullable)
}

/// A source that emits its rows' indices, in order, cut into pieces of the given lengths, one
/// piece per call, optionally after a round trip through the IO service.
///
/// A selected source emits only the rows its mask selects, as every plan does. A dense source
/// ignores its mask and emits every row of its range, as a bare segment scan does. Piece
/// lengths count emitted rows; the last piece takes whatever remains.
pub struct RowSource {
    /// Remaining indices to emit, in order.
    remaining: Vec<u64>,
    /// Lengths of the pieces still to emit, last first.
    pieces: Vec<usize>,
    /// A segment to read before emitting anything, standing in for a real read.
    segment: Option<SegmentId>,
    waiting: bool,
}

impl RowSource {
    /// A plan of `row_count` rows compiled to a [`RowSource`] emitting pieces of the given
    /// lengths.
    pub fn plan(
        row_count: u64,
        pieces: Vec<usize>,
        segment: Option<SegmentId>,
        dense: bool,
    ) -> PlanRef {
        SyntheticPlan::new(row_dtype(), row_count, Vec::new(), move |rows, mask, _| {
            let remaining: Vec<u64> = if dense || mask.all_true() {
                (rows.start..rows.end).collect()
            } else {
                (rows.start..rows.end)
                    .zip(mask.iter())
                    .filter_map(|(row, selected)| selected.then_some(row))
                    .collect()
            };
            if remaining.is_empty() {
                return Ok(None);
            }
            Ok(Some(Chain::new(RowSource {
                remaining,
                pieces: pieces.iter().rev().copied().collect(),
                segment,
                waiting: false,
            })))
        })
        .into_plan()
    }
}

impl Operator for RowSource {
    fn compute(&mut self, _input: Input, cx: &mut Cx<'_>) -> VortexResult<Step> {
        if self.waiting {
            self.waiting = false;
            cx.take_bytes()
                .ok_or_else(|| vortex_err!("RowSource ran without its bytes"))?;
        }
        if self.remaining.is_empty() {
            return Ok(Step::Finished);
        }
        let len = self
            .pieces
            .pop()
            .unwrap_or(self.remaining.len())
            .min(self.remaining.len());
        let piece: Vec<u64> = self.remaining.drain(..len).collect();
        Ok(Step::More(Buffer::from(piece).into_array()))
    }
}

impl Source for RowSource {
    fn request(&mut self) -> Option<SegmentId> {
        let segment = self.segment.take()?;
        self.waiting = true;
        Some(segment)
    }
}

/// What a [`Probe`] saw of its inlets on one call: for each, the batches queued and whether
/// its writer had closed it.
pub type ProbeRun = Vec<(usize, bool)>;

/// A source over one inlet per child that records what it sees each time it runs, then passes
/// its first inlet through, one batch per call, and drops what the others hold.
pub struct Probe {
    inlets: usize,
    capacity: usize,
    runs: Arc<Mutex<Vec<ProbeRun>>>,
}

impl Probe {
    /// A plan over `children` compiled to a probe whose inlets hold `capacity` batches and
    /// which logs into `runs`.
    pub fn plan(
        children: Vec<PlanRef>,
        capacity: usize,
        runs: Arc<Mutex<Vec<ProbeRun>>>,
    ) -> PlanRef {
        let row_count = children.first().map_or(0, |child| child.row_count());
        SyntheticPlan::new(
            row_dtype(),
            row_count,
            children.clone(),
            move |rows, mask, compiler| {
                let mut chains = Vec::with_capacity(children.len());
                for child in &children {
                    match compiler.compile(child, rows.clone(), mask)? {
                        Some(chain) => chains.push(chain),
                        None => return Ok(None),
                    }
                }
                let probe = Probe {
                    inlets: chains.len(),
                    capacity,
                    runs: Arc::clone(&runs),
                };
                Ok(Some(compiler.join(chains, probe)))
            },
        )
        .into_plan()
    }
}

impl Operator for Probe {
    fn compute(&mut self, _input: Input, cx: &mut Cx<'_>) -> VortexResult<Step> {
        let run: ProbeRun = (0..self.inlets)
            .map(|index| {
                let inlet = cx.inlet(index);
                (inlet.len(), inlet.closed())
            })
            .collect();
        self.runs.lock().push(run);
        for index in 1..self.inlets {
            while cx.inlet(index).take().is_some() {}
        }
        let mut first = cx.inlet(0);
        let batch: Option<ArrayRef> = first.take();
        match batch {
            Some(batch) => Ok(Step::Last(batch)),
            None if first.closed() => {
                for index in 1..self.inlets {
                    if !cx.inlet(index).closed() {
                        return Ok(Step::Blocked(Blocked::Inlet(index)));
                    }
                }
                Ok(Step::Finished)
            }
            None => Ok(Step::Blocked(Blocked::Inlet(0))),
        }
    }
}

impl Source for Probe {
    fn inlet_count(&self) -> usize {
        self.inlets
    }

    fn capacity(&self, _inlet: usize) -> usize {
        self.capacity
    }
}

/// An empty segment, for sources whose read is only a scheduling event.
pub fn empty_segment() -> BufferHandle {
    BufferHandle::new_host(Buffer::<u8>::empty().into_byte_buffer())
}
