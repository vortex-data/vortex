// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compiling plans into pipelines, and counting the readers of each segment so segments read
//! more than once are decoded once.

use std::ops::Range;

use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use super::DEFAULT_CAPACITY;
use super::port::PortId;
use super::port::SourceEnd;
use super::scan::Core;
use super::values::ValueCache;
use crate::execution::operator::Operator;
use crate::execution::operator::Source;
use crate::plan::ConcatPlan;
use crate::plan::PlanRef;
use crate::source::SegmentId;

/// A pipeline under construction: a source and the stages after it, not yet given outlets, so a
/// parent plan can add stages before it becomes a pipeline.
pub struct Chain {
    pub(crate) source: Box<dyn Source>,
    pub(crate) stages: Vec<Box<dyn Operator>>,
    pub(crate) inlets: Vec<PortId>,
}

impl Chain {
    /// A chain of `source` alone.
    pub fn new(source: impl Source + 'static) -> Self {
        Self {
            source: Box::new(source),
            stages: Vec::new(),
            inlets: Vec::new(),
        }
    }

    /// A chain passing the batches of `port` through.
    /// This chain with `stage` after its last stage.
    pub fn with(mut self, stage: impl Operator + 'static) -> Self {
        self.stages.push(Box::new(stage));
        self
    }
}

impl Core {
    /// Compiles `plan` over `rows`, restricted to `mask`. See [`Compiler::compile`].
    pub(crate) fn compile(
        &mut self,
        plan: &PlanRef,
        rows: Range<u64>,
        mask: &Mask,
    ) -> VortexResult<Option<Chain>> {
        Compiler { core: self }.compile(plan, rows, mask)
    }
}

/// What a plan compiles its children through. See
/// [`PlanVTable::compile`](crate::plan::PlanVTable::compile).
pub struct Compiler<'a> {
    pub(crate) core: &'a mut Core,
}

impl Compiler<'_> {
    /// Compiles `plan` over `rows` of its domain, restricted to `mask`, into a chain producing
    /// the rows `mask` selects, in order, or `None` when that is no row.
    pub fn compile(
        &mut self,
        plan: &PlanRef,
        rows: Range<u64>,
        mask: &Mask,
    ) -> VortexResult<Option<Chain>> {
        if rows.start >= rows.end {
            return Ok(None);
        }
        plan.compile(rows, mask, self)
    }

    /// A chain reading `chains` through `source`: each chain becomes a pipeline writing a new
    /// port, inlet `i` of `source` reading chain `i`, holding [`DEFAULT_CAPACITY`] batches.
    pub fn join(&mut self, chains: Vec<Chain>, source: impl Source + 'static) -> Chain {
        self.join_with(chains, |_| DEFAULT_CAPACITY, source)
    }

    /// [`join`](Self::join), with inlet `i` holding `capacity(i)` batches before its sink is
    /// blocked: `usize::MAX` for an inlet the source needs whole. An empty inlet always has
    /// room.
    pub fn join_with(
        &mut self,
        chains: Vec<Chain>,
        capacity: impl Fn(usize) -> usize,
        source: impl Source + 'static,
    ) -> Chain {
        let inlets = chains
            .into_iter()
            .enumerate()
            .map(|(index, chain)| {
                let port = self
                    .core
                    .arena
                    .create(capacity(index), None, SourceEnd::Unclaimed);
                self.core.add_pipeline(chain, smallvec::smallvec![port]);
                port
            })
            .collect();
        Chain {
            source: Box::new(source),
            stages: Vec::new(),
            inlets,
        }
    }

    /// The global row index of the scanned plan's row zero.
    pub fn row_offset(&self) -> u64 {
        self.core.row_offset
    }

    /// The session used for decoding and evaluation.
    pub fn session(&self) -> &VortexSession {
        &self.core.session
    }

    /// Whole values kept between the scans of the split compiling.
    pub(crate) fn values(&self) -> &ValueCache {
        &self.core.values
    }

    /// Records a source's segment dependency while compiling, without issuing IO.
    ///
    /// Source adapters call this for each segment their chain will request so filter prefetching
    /// and projection announcements see the same dependencies as execution.
    pub fn register_segment(&mut self, segment: SegmentId) {
        self.core.segments.push(segment);
    }
}

/// The chunks of `concat` overlapping `rows`: each chunk's index, where it starts, and the rows
/// of `rows` it holds, in the concatenation's domain. Found by binary search, so a split pays
/// for the chunks it overlaps, not for the chunks of the file.
pub(crate) fn overlapping<'a>(
    concat: &'a ConcatPlan,
    rows: &Range<u64>,
) -> impl Iterator<Item = (usize, u64, Range<u64>)> + 'a {
    let offsets = concat.row_offsets();
    let first = offsets
        .partition_point(|&offset| offset <= rows.start)
        .saturating_sub(1);
    let end = offsets.partition_point(|&offset| offset < rows.end);
    let rows = rows.clone();
    (first..end).filter_map(move |index| {
        let start = offsets[index];
        let chunk_end = offsets
            .get(index + 1)
            .copied()
            .unwrap_or_else(|| concat.row_count());
        let local = rows.start.max(start)..rows.end.min(chunk_end);
        (local.start < local.end).then_some((index, start, local))
    })
}
