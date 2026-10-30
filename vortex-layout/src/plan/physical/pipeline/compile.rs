// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compiling plans into pipelines while tracking row-preserving segment domains.

use std::ops::Range;

use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use super::DEFAULT_CAPACITY;
use super::port::PortId;
use super::port::SourceEnd;
use super::scan::Core;
use super::values::ValueCache;
use crate::plan::physical::ConcatPlan;
use crate::plan::physical::PlanRef;
use crate::plan::physical::SegmentScanPlan;
use crate::plan::physical::operator::Operator;
use crate::plan::physical::operator::Source;
use crate::plan::physical::operators::ScanSource;

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
        Compiler {
            core: self,
            segment_origin: Some(0),
        }
        .compile_rows(plan, rows, mask)
    }
}

/// What a plan compiles its children through. See
/// [`PlanVTable::compile`](crate::plan::physical::PlanVTable::compile).
pub struct Compiler<'a> {
    pub(crate) core: &'a mut Core,
    /// Root row corresponding to this domain's row zero, when the mapping is known.
    segment_origin: Option<u64>,
}

impl Compiler<'_> {
    /// Compiles `plan` over `rows` of its domain, restricted to `mask`, into a chain producing
    /// the rows `mask` selects, in order, or `None` when that is no row.
    /// This general entry point makes no assumption that child coordinates preserve root rows.
    pub fn compile(
        &mut self,
        plan: &PlanRef,
        rows: Range<u64>,
        mask: &Mask,
    ) -> VortexResult<Option<Chain>> {
        let origin = self.segment_origin.take();
        let result = self.compile_rows(plan, rows, mask);
        self.segment_origin = origin;
        result
    }

    /// Compiles a child whose coordinates are in the current row domain.
    pub(crate) fn compile_rows(
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

    /// Compiles a concatenated child whose row zero is `offset` in the current domain.
    pub(crate) fn compile_offset(
        &mut self,
        plan: &PlanRef,
        rows: Range<u64>,
        mask: &Mask,
        offset: u64,
    ) -> VortexResult<Option<Chain>> {
        let origin = self.segment_origin;
        self.segment_origin = origin.and_then(|origin| origin.checked_add(offset));
        let result = self.compile_rows(plan, rows, mask);
        self.segment_origin = origin;
        result
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

    /// The rows `rows` of `scan`'s segment, filtered by `filter`, read by the split compiling.
    /// Whole arrays can be reused when this segment has a known root row domain.
    pub(crate) fn scan(
        &mut self,
        scan: &SegmentScanPlan,
        rows: Range<u64>,
        filter: Option<Mask>,
    ) -> VortexResult<Option<Chain>> {
        let filter = match filter {
            Some(filter) if filter.all_false() => return Ok(None),
            Some(filter) if filter.all_true() => None,
            filter => filter,
        };
        let slice = if rows.start == 0 && rows.end == scan.row_count() {
            None
        } else {
            Some(usize::try_from(rows.start)?..usize::try_from(rows.end)?)
        };
        self.core.segments.push(scan.segment_id());
        let cached = self.core.segment_cache.as_ref().and_then(|cache| {
            // Without a filter, a whole segment cannot cross a chunk or projection-morsel
            // boundary. Partial reads still share segments across those boundaries.
            if slice.is_none() && !cache.reuse_whole() {
                return None;
            }
            let start = self.segment_origin?;
            let end = start.checked_add(scan.row_count())?;
            Some(cache.reader(scan.clone().into_plan(), start..end))
        });
        Ok(Some(Chain::new(
            ScanSource::new(scan.clone(), slice, filter).with_cache(cached),
        )))
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
