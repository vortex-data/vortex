// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compiling plans into pipelines, and counting the readers of each segment so segments read
//! more than once are decoded once.

use std::ops::Range;

use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use vortex_error::VortexExpect;
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
use crate::plan::physical::operators::ConcatSource;
use crate::plan::physical::operators::ScanSource;
use crate::plan::physical::operators::SelectStage;
use crate::segments::SegmentId;

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
    pub(crate) fn port(port: PortId) -> Self {
        Self {
            source: Box::new(ConcatSource::new(1)),
            stages: Vec::new(),
            inlets: vec![port],
        }
    }

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
/// [`PlanVTable::compile`](crate::plan::physical::PlanVTable::compile).
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

    /// The rows `rows` of `scan`'s segment, filtered by `filter`, read by the split compiling.
    /// A segment several readers read is decoded once and shared.
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
        let split = self.core.split;
        Ok(Some(match self.claim(scan, split) {
            None => Chain::new(ScanSource::new(scan.clone(), slice, filter)),
            Some(port) => {
                let chain = Chain::port(port);
                if slice.is_none() && filter.is_none() {
                    chain
                } else {
                    chain.with(SelectStage::new(slice, filter))
                }
            }
        }))
    }

    /// Claims split `split`'s port for `scan`'s segment when the segment is shared, building
    /// the pipeline that decodes it for the first claimant with one port per reader in every
    /// split not yet finished, or `None` when this split's reader is its only one.
    fn claim(&mut self, scan: &SegmentScanPlan, split: usize) -> Option<PortId> {
        if split == usize::MAX {
            return None;
        }
        let segment = scan.segment_id();
        let shares = &mut self.core.shares;
        let ports = &mut shares.ports[split];
        if let Some(waiting) = ports.get_mut(&segment) {
            let port = waiting.pop();
            if waiting.is_empty() {
                ports.remove(&segment);
            }
            return port;
        }
        let ranges = shares.shared.remove(&segment)?;
        let mut claimants: SmallVec<[usize; 4]> = SmallVec::new();
        for rows in &ranges {
            claimants.extend(
                shares
                    .overlapping(rows)
                    .filter(|&split| !shares.finished[split] && shares.overlaps(split, rows)),
            );
        }
        if claimants.len() <= 1 {
            return None;
        }
        let mut outlets: SmallVec<[PortId; 1]> = SmallVec::with_capacity(claimants.len());
        let mut mine = None;
        for claimant in claimants {
            let port = self
                .core
                .arena
                .create(usize::MAX, None, SourceEnd::Unclaimed);
            outlets.push(port);
            if claimant == split && mine.is_none() {
                mine = Some(port);
            } else {
                self.core.shares.ports[claimant]
                    .entry(segment)
                    .or_default()
                    .push(port);
            }
        }
        self.core.add_pipeline(
            Chain::new(ScanSource::new(scan.clone(), None, None)),
            outlets,
        );
        mine
    }
}

/// Where each segment read more than once is read from, and the ports of the segments whose
/// decoding pipeline is built.
#[derive(Default)]
pub(crate) struct Shares {
    /// The splits' row ranges, in split order.
    splits: Vec<Range<u64>>,
    /// Whether the splits are sorted and disjoint, so the splits overlapping a range are found
    /// by binary search.
    ordered: bool,
    /// Splits that have finished, and so will claim nothing.
    finished: Vec<bool>,
    /// Segment readers collected before the scan starts. A split may read only a few of a
    /// file's segments, so recording a reader must not allocate by the file-wide segment id.
    reaches: Vec<(SegmentId, Range<u64>)>,
    /// The row ranges of segments read more than once, until their decoding pipeline is built.
    shared: FxHashMap<SegmentId, SmallVec<[Range<u64>; 1]>>,
    /// By split: the unclaimed ports of each shared segment the split reads.
    ports: Vec<FxHashMap<SegmentId, SmallVec<[PortId; 1]>>>,
}

impl Shares {
    pub(crate) fn new(splits: Vec<Range<u64>>) -> Self {
        let ordered = splits.windows(2).all(|pair| pair[0].end <= pair[1].start);
        Self {
            finished: vec![false; splits.len()],
            ports: (0..splits.len()).map(|_| FxHashMap::default()).collect(),
            splits,
            ordered,
            reaches: Vec::new(),
            shared: FxHashMap::default(),
        }
    }

    /// The splits overlapping `rows`.
    fn overlapping(&self, rows: &Range<u64>) -> Range<usize> {
        if self.ordered {
            let first = self.splits.partition_point(|split| split.end <= rows.start);
            let end = self.splits.partition_point(|split| split.start < rows.end);
            return first..end.max(first);
        }
        let mut overlapping = self
            .splits
            .iter()
            .enumerate()
            .filter(|(_, split)| split.start < rows.end && rows.start < split.end)
            .map(|(index, _)| index);
        match overlapping.next() {
            Some(first) => first..overlapping.next_back().unwrap_or(first) + 1,
            None => 0..0,
        }
    }

    fn overlaps(&self, split: usize, rows: &Range<u64>) -> bool {
        let split = &self.splits[split];
        split.start < rows.end && rows.start < split.end
    }

    /// Records the ranges `plan` reads each of its segments for, read over `rows` by every
    /// split overlapping them. Call once per plan a split stage runs, before the scan starts,
    /// then [`retain_shared`](Self::retain_shared).
    pub(crate) fn add(&mut self, plan: &PlanRef, rows: Range<u64>) -> VortexResult<()> {
        let reaches = &mut self.reaches;
        plan.reach(rows, &Reach::Offset(0), &mut |segment, rows| {
            reaches.push((segment, rows))
        })
    }

    /// Counts a single split's readers without lowering chunks its mask excludes.
    pub(crate) fn add_selected(
        &mut self,
        plan: &PlanRef,
        rows: Range<u64>,
        mask: &Mask,
    ) -> VortexResult<()> {
        if mask.all_false() {
            return Ok(());
        }
        if mask.all_true() {
            return self.add(plan, rows);
        }
        let at = Reach::Selected {
            offset: 0,
            rows: rows.clone(),
            mask: mask.clone(),
        };
        let reaches = &mut self.reaches;
        plan.reach(rows, &at, &mut |segment, rows| {
            reaches.push((segment, rows))
        })
    }

    /// Keeps only the segments more than one reader reads. The rest are read by the one
    /// pipeline that needs them.
    pub(crate) fn retain_shared(&mut self) {
        let mut reaches = std::mem::take(&mut self.reaches);
        reaches.sort_unstable_by_key(|(segment, _)| *segment);
        for readers in reaches.chunk_by(|(left, _), (right, _)| left == right) {
            let (segment, rows) = &readers[0];
            // One reader per split overlapping the range: shared when the range crosses a
            // split boundary, or when multiple plan nodes read the segment.
            if readers.len() == 1
                && self
                    .overlapping(rows)
                    .filter(|&split| self.overlaps(split, rows))
                    .nth(1)
                    .is_none()
            {
                continue;
            }
            self.shared.insert(
                *segment,
                readers.iter().map(|(_, rows)| rows.clone()).collect(),
            );
        }
    }

    /// Every port made for a reader that has not claimed it yet.
    #[cfg(debug_assertions)]
    pub(crate) fn waiting(&self) -> impl Iterator<Item = PortId> + '_ {
        self.ports
            .iter()
            .flat_map(|ports| ports.values().flatten().copied())
    }

    /// Marks split `split` finished and returns the ports it never claimed: those of its
    /// readers that never came, as a chunk its mask ruled out or a stage it never ran. Each is
    /// still an outlet of the pipeline that shares it, which must stop pushing into it: a port
    /// has a consumer until it is closed and drained, so the caller detaches each port from
    /// its sink before discarding it. See [`Core::finish_split`].
    pub(crate) fn finish(&mut self, split: usize) -> SmallVec<[PortId; 4]> {
        self.finished[split] = true;
        std::mem::take(&mut self.ports[split])
            .into_values()
            .flatten()
            .collect()
    }
}

/// How a plan's rows map to the rows of the plan a scan's splits range over.
#[derive(Clone, Debug)]
pub enum Reach {
    /// Row `r` is scanned row `r + offset`.
    Offset(u64),
    /// Every row is read for these scanned rows, as a dictionary's values are for its codes.
    Fixed(Range<u64>),
    /// A linear mapping restricted to selected rows in the scan's root domain.
    Selected {
        /// Root row index of this plan's row zero.
        offset: u64,
        /// The root row range the mask covers.
        rows: Range<u64>,
        /// Selected positions relative to `rows.start`.
        mask: Mask,
    },
}

impl Reach {
    /// The scanned rows `rows` are read for.
    pub fn root(&self, rows: &Range<u64>) -> Range<u64> {
        match self {
            Reach::Offset(offset) | Reach::Selected { offset, .. } => {
                rows.start + offset..rows.end + offset
            }
            Reach::Fixed(root) => root.clone(),
        }
    }

    /// The mapping of a child whose row zero is this plan's row `by`.
    pub fn shift(&self, by: u64) -> Self {
        match self {
            Reach::Offset(offset) => Reach::Offset(offset + by),
            Reach::Fixed(root) => Reach::Fixed(root.clone()),
            Reach::Selected { offset, rows, mask } => Reach::Selected {
                offset: offset + by,
                rows: rows.clone(),
                mask: mask.clone(),
            },
        }
    }

    /// The mapping of a child read whole for `rows` of this plan, as dictionary values are.
    pub fn fixed(&self, rows: &Range<u64>) -> Self {
        Reach::Fixed(self.root(rows))
    }

    /// Whether the selection needs any row of this child, before lowering it.
    pub(crate) fn selects(&self, local: &Range<u64>) -> bool {
        let Reach::Selected { offset, rows, mask } = self else {
            return true;
        };
        let start = (local.start + offset).max(rows.start);
        let end = (local.end + offset).min(rows.end);
        if start >= end {
            return false;
        }
        let start = usize::try_from(start - rows.start).vortex_expect("offset fits selection mask");
        let end = usize::try_from(end - rows.start).vortex_expect("offset fits selection mask");
        match mask {
            Mask::AllTrue(_) => true,
            Mask::AllFalse(_) => false,
            Mask::Values(values) => match values.cached_indices() {
                Some(indices) => indices
                    .get(indices.partition_point(|&index| index < start))
                    .is_some_and(|&index| index < end),
                None => values.bit_buffer().count_range(start, end) > 0,
            },
        }
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

/// Unclaimed shared ports, for tests.
#[cfg(test)]
impl Shares {
    pub(crate) fn pending(&self) -> usize {
        self.ports
            .iter()
            .flat_map(|ports| ports.values())
            .map(SmallVec::len)
            .sum()
    }
}
